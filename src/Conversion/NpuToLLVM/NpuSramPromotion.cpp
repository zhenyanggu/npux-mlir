//======================================================
// src/Conversion/NpuToLLVM/NpuSramPromotion.cpp
// Promotes DRAM subviews to SRAM buffers for computation
//======================================================

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// 判断操作是否支持原地更新 (Element-wise)
bool isElementWiseInPlace(linalg::GenericOp op) {
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1) return false;

    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 2) return false;

    // Input Map 必须等于 Output Map
    if (maps[0] != maps[1]) return false;

    // 2. 修复：使用 utils::IteratorType::parallel 进行比较
    for (auto type : op.getIteratorTypesArray()) {
        if (type != utils::IteratorType::parallel) return false;
    }

    return true;
}

class PromoteLinalgToSramPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {

    if (!op->hasAttr("npu.target")) return failure();

    Location loc = op.getLoc();
    
    // Key: Operand Index, Value: New SRAM Value
    llvm::SmallDenseMap<unsigned, Value> operandsToReplace;
    bool promotedAny = false;

    // 记录 Inputs 对应的 SRAM Buffer，用于后续复用检查
    // 假设 Input Index -> SRAM Value
    llvm::SmallDenseMap<unsigned, Value> inputSramBuffers;

    // ==========================================
    // 1. Inputs (DRAM -> SRAM)
    // ==========================================
    for (auto item : llvm::enumerate(op.getInputs())) {
      unsigned idx = item.index(); 
      Value currentInput = item.value();
      auto type = dyn_cast<MemRefType>(currentInput.getType());

      if (!type || type.getMemorySpaceAsInt() == 2) {
          // 如果已经是 SRAM，我们也记录下来，万一能复用呢
          if (type && type.getMemorySpaceAsInt() == 2) {
             inputSramBuffers[idx] = currentInput;
          }
          continue;
      }

      auto sramType = MemRefType::get(type.getShape(), type.getElementType(), {}, 2);
      Value sramBuf = rewriter.create<memref::AllocOp>(loc, sramType);

      rewriter.create<memref::CopyOp>(loc, currentInput, sramBuf);

      operandsToReplace[idx] = sramBuf;
      inputSramBuffers[idx] = sramBuf; // 记录刚分配的 Input Buffer
      promotedAny = true;
    }

    // ==========================================
    // 2. Outputs (分配 SRAM 或 复用 Input SRAM)
    // ==========================================
    unsigned inputCount = op.getNumDpsInputs();
    SmallVector<std::pair<Value, Value>> outputCopyBacks; 

    // 检查是否可以原地更新
    bool canReuse = isElementWiseInPlace(op);

    for (auto item : llvm::enumerate(op.getOutputs())) {
      unsigned outputIdxRel = item.index();
      unsigned operandIdx = inputCount + outputIdxRel; 
      Value currentOutput = item.value();
      auto type = dyn_cast<MemRefType>(currentOutput.getType());

      if (!type || type.getMemorySpaceAsInt() == 2) {
        continue;
      }

      Value sramBuf;

      // === In-Place Logic ===
      // 如果只有一个 Input 且满足 Element-wise，尝试复用 Input[0] 的 Buffer
      if (canReuse && outputIdxRel == 0 && inputSramBuffers.count(0)) {
          Value inputBuf = inputSramBuffers[0];
          auto inType = cast<MemRefType>(inputBuf.getType());
          
          // 再次确认 Size/Shape 是否匹配（严格来说 maps 相等意味着 shape 兼容，但安全起见）
          if (inType == MemRefType::get(type.getShape(), type.getElementType(), {}, 2)) {
              sramBuf = inputBuf;
               llvm::errs() << "[SramPromotion] Reusing Input Buffer for Output!\n";
          }
      }
      // ======================

      // 无法复用，分配新的
      if (!sramBuf) {
          auto sramType = MemRefType::get(type.getShape(), type.getElementType(), {}, 2);
          sramBuf = rewriter.create<memref::AllocOp>(loc, sramType);
      }

      operandsToReplace[operandIdx] = sramBuf;
      outputCopyBacks.push_back({sramBuf, currentOutput});
      promotedAny = true;
    }

    if (!promotedAny) {
      return failure();
    }

    // ==========================================
    // 3. Clone & Replace
    // ==========================================
    Operation *newOp = rewriter.clone(*op.getOperation());

    for (auto it : operandsToReplace) {
      newOp->setOperand(it.first, it.second);
    }

    // ==========================================
    // 4. Copy Back
    // ==========================================
    rewriter.setInsertionPointAfter(newOp);

    for (auto pair : outputCopyBacks) {
      rewriter.create<memref::CopyOp>(loc, pair.first, pair.second);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct SramPromotionPass : public PassWrapper<SramPromotionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SramPromotionPass)

  StringRef getArgument() const override { return "npu-sram-promotion"; }
  StringRef getDescription() const override { return "Promote DRAM to SRAM with In-Place optimization"; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<PromoteLinalgToSramPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuSramPromotionPass() {
  return std::make_unique<SramPromotionPass>();
}