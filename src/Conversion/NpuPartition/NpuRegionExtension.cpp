//===============================================
// src/Conversion/NpuPartition/NpuRegionExtension.cpp
// this file extends the NPU execute_region boundary
// by sinking producers  and fusing consumers
// into the region, realizing zero-copy optimization.
//===============================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

//=============================================================================
// Helper 1: 安全移动 Op 及其依赖 (Clone Empty, Move Op)
//=============================================================================
static void moveOpAndDepsIntoRegion(Operation *op, Block *targetBlock,
    bool moveToStart, Operation *boundary, PatternRewriter &rewriter) {

  if (moveToStart) {
    rewriter.moveOpBefore(op, targetBlock, targetBlock->begin());
  } else {
    assert(boundary && "Boundary must be provided for consumers");
    rewriter.moveOpBefore(op, boundary);
  }

  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    for (OpOperand &initOperand : dpsOp.getDpsInitsMutable()) {
      Operation *defOp = initOperand.get().getDefiningOp();

      // 如果 Empty 定义在外部
      if (defOp && isa<tensor::EmptyOp>(defOp) &&
          defOp->getBlock() != targetBlock) {

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(targetBlock);

        // 克隆进内部 (插在最顶端，也就是 Op 的前面)
        Operation *newEmpty = rewriter.clone(*defOp);

        // 更新 Op 使用新的 Empty
        rewriter.modifyOpInPlace(
            op, [&] { initOperand.set(newEmpty->getResult(0)); });
      }
    }
  }
}

//=============================================================================
// Helper 2: Constant Input Copy
//=============================================================================
static void injectCopyForConstants(Operation *op, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  for (OpOperand &operand : op->getOpOperands()) {
    Value inputVal = operand.get();
    Operation *defOp = inputVal.getDefiningOp();

    if (defOp && defOp->getName().getStringRef() == "onnx.Constant") {
      auto tensorType = cast<RankedTensorType>(inputVal.getType());
      Location loc = op->getLoc();

      Value emptyBuffer = rewriter.create<tensor::EmptyOp>(
          loc, tensorType.getShape(), tensorType.getElementType());
      auto copyOp = rewriter.create<linalg::CopyOp>(loc, inputVal, emptyBuffer);

      rewriter.modifyOpInPlace(op, [&] {
        op->setOperand(operand.getOperandNumber(), copyOp.getResult(0));
      });
    }
  }
}

//=============================================================================
// Helper 3: Output Isolation (当无法融合时，插入 Copy)
//=============================================================================
static void injectInternalOutputCopy(
    scf::YieldOp yieldOp, unsigned resultIdx, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(yieldOp);

  Value originalVal = yieldOp.getOperand(resultIdx);

  // 如果已经是 Copy 了，就不用再处理
  if (auto defOp = originalVal.getDefiningOp()) {
    if (isa<linalg::CopyOp>(defOp))
      return;
  }

  auto tensorType = cast<RankedTensorType>(originalVal.getType());
  Location loc = yieldOp.getLoc();

  // 1. 内部 Alloc
  Value emptyBuffer = rewriter.create<tensor::EmptyOp>(
      loc, tensorType.getShape(), tensorType.getElementType());

  // 2. 内部 Copy
  auto copyOp = rewriter.create<linalg::CopyOp>(loc, originalVal, emptyBuffer);

  // 3. 修改 Yield 返回 Copy 的结果
  rewriter.modifyOpInPlace(
      yieldOp, [&] { yieldOp.setOperand(resultIdx, copyOp.getResult(0)); });
}

//=============================================================================
// Pattern
//=============================================================================
struct ExtendRegionBoundary : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern<scf::ExecuteRegionOp>::OpRewritePattern;

  static constexpr StringRef kProcessedAttr = "npu.region_extended";

  LogicalResult matchAndRewrite(
      scf::ExecuteRegionOp execOp, PatternRewriter &rewriter) const override {

    Block *regionBlock = &execOp.getRegion().front();
    bool changed = false;

    if (execOp->hasAttr(kProcessedAttr)) {
      return failure();
    }

    //=========================================================================
    // Part 1. Input Sinking (生产者下沉)
    //=========================================================================
    for (Operation &op : llvm::make_early_inc_range(*regionBlock)) {
      if (isa<linalg::CopyOp>(op))
        continue;

      for (Value operand : op.getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp || defOp->getBlock() != execOp->getBlock())
          continue;

        if (defOp->getName().getStringRef() == "onnx.Constant") {
          injectCopyForConstants(&op, rewriter);
          changed = true;
        } else {
          // Input Sinking: 移到最前
          moveOpAndDepsIntoRegion(defOp, regionBlock, true, nullptr, rewriter);
          changed = true;
        }
      }
    }

    //=========================================================================
    // Part 2. Output Fusion (消费者融合) - 逻辑已恢复并修复
    //=========================================================================
    Operation *terminator = regionBlock->getTerminator();
    auto yieldOp = cast<scf::YieldOp>(terminator);

    // 遍历每一个 ExecuteRegion 的结果
    for (OpResult result : execOp.getResults()) {
      unsigned resultIdx = result.getResultNumber();

      bool canFuse = true;
      Operation *user = nullptr;

      // 检查融合条件：
      // 1. 只有一个 User
      // 2. User 和 execOp 在同一层 Block
      // 3. User 也是单输出 (简化处理)
      if (!result.hasOneUse()) {
        canFuse = false;
      } else {
        user = *result.getUsers().begin();
        if (user->getBlock() != execOp->getBlock())
          canFuse = false;
        if (user->getNumResults() != 1)
          canFuse = false;

        // 遇到 Terminator (如 Return) 不能融合
        if (user->hasTrait<OpTrait::IsTerminator>())
          canFuse = false;
      }

      // [Branch A] 无法融合 -> 隔离 (Isolation)
      if (!canFuse) {
        // 检查是否已经隔离过 (避免死循环)
        Value yieldOperand = yieldOp.getOperand(resultIdx);
        if (auto defOp = yieldOperand.getDefiningOp()) {
          if (isa<linalg::CopyOp>(defOp))
            continue; // 已经是 Copy，跳过
        }

        injectInternalOutputCopy(yieldOp, resultIdx, rewriter);
        changed = true; // <--- 关键修复：标记为已修改
        continue;
      }

      // [Branch B] 执行融合 (Fusion)
      // user 现在确定要被移进来了

      Value internalYieldVal = yieldOp.getOperand(resultIdx);
      Type newResultType = user->getResult(0).getType();

      moveOpAndDepsIntoRegion(
          user, regionBlock, /*moveToStart=*/false, yieldOp, rewriter);

      // 把 User 的输入从 "Region外部结果" 改为 "Region内部源头"
      rewriter.modifyOpInPlace(
          user, [&] { user->replaceUsesOfWith(result, internalYieldVal); });

      rewriter.modifyOpInPlace(execOp, [&] { result.setType(newResultType); });

      // 此时 User 还没连到 Yield，只有外部有人在用 User。
      // 我们把外部对 User 的引用，全部改为引用 Region 的 Result。
      rewriter.replaceAllUsesWith(user->getResult(0), result);

      // 现在我们安全地把 User 连给 Yield。
      // 因为 replaceAllUsesWith 已经跑过了，这一步建立的新连接不会被误替换。
      rewriter.modifyOpInPlace(
          yieldOp, [&] { yieldOp.setOperand(resultIdx, user->getResult(0)); });

      changed = true;
    }

    if (changed) {
      rewriter.modifyOpInPlace(execOp,
          [&] { execOp->setAttr(kProcessedAttr, rewriter.getUnitAttr()); });
      return success();
    }

    return failure();
  }
};

struct NpuRegionExtensionPass
    : public PassWrapper<NpuRegionExtensionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuRegionExtensionPass)
  llvm::StringRef getArgument() const override {
    return "npu-region-extension";
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ExtendRegionBoundary>(context);
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true).enableFolding(true);

    if (failed(applyPatternsGreedily(
            getOperation().getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuRegionExtensionPass() {
  return std::make_unique<NpuRegionExtensionPass>();
}
