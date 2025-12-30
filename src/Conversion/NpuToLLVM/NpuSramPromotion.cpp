//======================================================
// src/Conversion/NpuToLLVM/NpuSramPromotion.cpp
// Promotes DRAM subviews to SRAM buffers for computation
//======================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

const int SRAM_SPACE = 1;        // NPU Internal
const int SHARED_DRAM_SPACE = 2; // Interface

// 辅助函数：创建 SRAM 上的 Alloc
Value createSramAlloc(OpBuilder &b, Location loc, Value originalMemRef) {
  auto oldType = mlir::cast<MemRefType>(originalMemRef.getType());

  // 创建新的 MemRefType，使用 SRAM_SPACE (1)
  // 注意：这里我们通常希望 SRAM 分配是连续的 (Identity Layout)，
  // 即使来源是 strided subview，拷贝到 SRAM 后最好变为紧凑布局以利于 NPU 计算。
  MemRefType newType = MemRefType::Builder(oldType)
                           .setMemorySpace(b.getI64IntegerAttr(SRAM_SPACE))
                           .setLayout({});

  return b.create<memref::AllocOp>(loc, newType);
}

struct PromoteToSramPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    // 1. 只处理 NPU Kernel 内部的 Op
    if (!op->hasAttr("npu.target"))
      return failure();

    bool changed = false;
    Location loc = op.getLoc();

    // 获取输入和输出操作数
    SmallVector<Value> newInputs = op.getInputs();
    SmallVector<Value> newOutputs = op.getOutputs();

    // --- 处理 Input Operands (DRAM -> SRAM) ---
    for (unsigned i = 0; i < newInputs.size(); ++i) {
      Value input = newInputs[i];
      auto type = mlir::cast<MemRefType>(input.getType());

      // 只处理位于 Shared DRAM (Space 2) 的内存
      if (type.getMemorySpaceAsInt() == SHARED_DRAM_SPACE) {
        // 在 Op 之前插入 Alloc 和 Copy
        Value sramBuf = createSramAlloc(rewriter, loc, input);
        rewriter.create<memref::CopyOp>(loc, input, sramBuf); // DRAM -> SRAM

        newInputs[i] = sramBuf;
        changed = true;
      }
    }

    // --- 处理 Output Operands (SRAM -> Op -> SRAM -> DRAM) ---
    // 注意：这里假设 Output 也是 Read-Write 或者 Write-Only
    // 如果是 Accumulate (+=)，我们需要先 CopyIn，计算，再 CopyOut。
    // 如果是 Pure Overwrite (=)，理论上不需要 CopyIn，但为了简化逻辑，
    // 这里统一按 Read-Modify-Write 处理，或者根据 linalg 的 payload 逻辑优化。
    // 为了安全起见，我们先做 CopyIn (保留原值) -> Compute -> CopyOut。

    for (unsigned i = 0; i < newOutputs.size(); ++i) {
      Value output = newOutputs[i];
      auto type = mlir::cast<MemRefType>(output.getType());

      if (type.getMemorySpaceAsInt() == SHARED_DRAM_SPACE) {
        Value sramBuf = createSramAlloc(rewriter, loc, output);

        // 【关键修复】检查这个 Output 在计算块(Region)里是否被使用
        // Linalg Generic 的 BlockArgs 顺序是：所有 Inputs, 然后所有 Outputs
        // 所以 Output[i] 对应的 BlockArg 索引是：inputs.size() + i
        Block &block = op.getRegion().front();
        BlockArgument outputArg = block.getArgument(op.getInputs().size() + i);

        // 如果 outputArg 在内部被使用了 (比如 C += ...)，才需要 CopyIn
        // 如果 outputArg 没有被使用 (比如 C = ...)，则不需要 CopyIn
        if (!outputArg.use_empty()) {
          rewriter.create<memref::CopyOp>(loc, output, sramBuf);
        }

        newOutputs[i] = sramBuf;
        changed = true;
      }
    }

    if (!changed)
      return failure();

    // --- 创建新的 Linalg Op ---
    // Clone 原 Op，但使用新的 SRAM 操作数
    auto newOp =
        mlir::cast<linalg::GenericOp>(rewriter.clone(*op.getOperation()));
    newOp.getInputsMutable().assign(newInputs);
    newOp.getOutputsMutable().assign(newOutputs);

    // --- 处理 Output Copy Back (SRAM -> DRAM) ---
    rewriter.setInsertionPointAfter(newOp);
    for (unsigned i = 0; i < op.getOutputs().size(); ++i) {
      Value originalOut = op.getOutputs()[i];
      auto type = mlir::cast<MemRefType>(originalOut.getType());

      if (type.getMemorySpaceAsInt() == SHARED_DRAM_SPACE) {
        Value sramBuf = newOutputs[i]; // 这是我们在上面分配的 sram buf
        rewriter.create<memref::CopyOp>(
            loc, sramBuf, originalOut); // SRAM -> DRAM

        // 如果没有自动 Dealloc Pass，可以在这里插入 dealloc
        // rewriter.create<memref::DeallocOp>(loc, sramBuf);
      }
    }

    // Input 的 Dealloc 也可以在这里做，或者留给 buffer-deallocation pass

    // 替换旧 Op
    rewriter.eraseOp(op);
    return success();
  }
};

struct NpuSramPromotionPass
    : public PassWrapper<NpuSramPromotionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuSramPromotionPass)
  StringRef getArgument() const override { return "npu-sram-promotion"; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    // 只在 NPU Kernel 内部做 Promotion
    if (!func->hasAttr("npu.target"))
      return;

    RewritePatternSet patterns(&getContext());
    patterns.add<PromoteToSramPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuSramPromotionPass() {
  return std::make_unique<NpuSramPromotionPass>();
}