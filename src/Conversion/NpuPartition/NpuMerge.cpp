//===============================================
// src/Conversion/NpuPartition/NpuMerge.cpp
// this file merges regions of NPU-executable operations
// that can be executed together into a single region
//===============================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

//=============================================================================
// Pattern: MergeAdjacentExecuteRegions (保持不变)
//=============================================================================
struct MergeAdjacentExecuteRegions
    : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern<scf::ExecuteRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      scf::ExecuteRegionOp opA, PatternRewriter &rewriter) const override {
    // 1. 获取下一个 Operation
    Operation *nextOp = opA->getNextNode();
    if (!nextOp)
      return failure();

    auto opB = dyn_cast<scf::ExecuteRegionOp>(nextOp);
    if (!opB)
      return failure();

    Block *blockA = &opA.getRegion().front();
    Block *blockB = &opB.getRegion().front();

    auto yieldA = cast<scf::YieldOp>(blockA->getTerminator());
    auto yieldB = cast<scf::YieldOp>(blockB->getTerminator());

    // 2. 处理依赖 (SSA Remapping)
    for (auto it : llvm::zip(opA.getResults(), yieldA.getOperands())) {
      Value resultA = std::get<0>(it);
      Value yieldValA = std::get<1>(it);

      for (auto &use : llvm::make_early_inc_range(resultA.getUses())) {
        if (opB->isAncestor(use.getOwner())) {
          use.set(yieldValA);
        }
      }
    }

    // 3. 决定新的 Result (Smart Merge)
    SmallVector<Value> newYieldOperands;
    SmallVector<Type> newResultTypes;

    int newResultIndex = 0;
    SmallVector<int> mapResultAtoNewIdx(opA.getNumResults(), -1);

    // 3.1 Check Op A
    for (auto it : llvm::enumerate(opA.getResults())) {
      unsigned idx = it.index();
      Value oldRes = it.value();
      Value internalVal = yieldA.getOperand(idx);

      if (!oldRes.use_empty()) {
        newYieldOperands.push_back(internalVal);
        newResultTypes.push_back(oldRes.getType());
        mapResultAtoNewIdx[idx] = newResultIndex++;
      }
    }

    // 3.2 Check Op B
    for (Value val : yieldB.getOperands()) {
      newYieldOperands.push_back(val);
      newResultTypes.push_back(val.getType());
      newResultIndex++;
    }

    // 4. Create new Op
    auto newOp =
        rewriter.create<scf::ExecuteRegionOp>(opA.getLoc(), newResultTypes);
    Block *newBlock = rewriter.createBlock(&newOp.getRegion());

    // 5. Splice
    rewriter.eraseOp(yieldA);
    newBlock->getOperations().splice(newBlock->end(), blockA->getOperations());
    newBlock->getOperations().splice(newBlock->end(), blockB->getOperations());

    // 6. Update Yield
    auto newYieldOp = cast<scf::YieldOp>(newBlock->getTerminator());
    rewriter.modifyOpInPlace(
        newYieldOp, [&] { newYieldOp->setOperands(newYieldOperands); });

    // 7. Replace Uses
    for (unsigned i = 0; i < opA.getNumResults(); ++i) {
      int newIdx = mapResultAtoNewIdx[i];
      if (newIdx != -1) {
        opA.getResult(i).replaceAllUsesWith(newOp.getResult(newIdx));
      }
    }

    int offset = newResultIndex - opB.getNumResults();
    for (unsigned i = 0; i < opB.getNumResults(); ++i) {
      opB.getResult(i).replaceAllUsesWith(newOp.getResult(offset + i));
    }

    // 8. Cleanup
    rewriter.eraseOp(opB);
    rewriter.eraseOp(opA);

    return success();
  }
};

//=============================================================================
// Pass Definition
//=============================================================================

// 修改为针对 func::FuncOp，因为 applyPatternsGreedily 需要 Region，
// 而 FuncOp 保证有一个 Body Region。
struct NpuMergePass
    : public PassWrapper<NpuMergePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuMergePass)
  llvm::StringRef getArgument() const override { return "npu-merge"; }
  llvm::StringRef getDescription() const override {
    return "Merge adjacent NPU ExecuteRegion operations.";
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<MergeAdjacentExecuteRegions>(context);

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true)
        .enableFolding(true)
        .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Aggressive);

    if (failed(applyPatternsGreedily(
            getOperation().getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuMergePass() {
  return std::make_unique<NpuMergePass>();
}