//=============================================================================
// Plan profitable forward fusion chains and wrap them in npux.fusion_group.
//=============================================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "src/Conversion/NpuTiling/FusionAnalysis/NpuFusionPlanner.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

static void collectMovableDependencies(Operation *op, Operation *anchor,
    Block *block, llvm::SmallPtrSetImpl<Operation *> &moveSet) {
  if (!op || op->getBlock() != block)
    return;

  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp || defOp->getBlock() != block)
      continue;
    if (defOp->isBeforeInBlock(anchor))
      continue;
    if (!moveSet.insert(defOp).second)
      continue;
    collectMovableDependencies(defOp, anchor, block, moveSet);
  }
}

struct NpuFusionPlanningPass
    : public PassWrapper<NpuFusionPlanningPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuFusionPlanningPass)

  StringRef getArgument() const override { return "npu-fusion-plan"; }
  StringRef getDescription() const override {
    return "Analyze forward fusion chains and wrap them in npux.fusion_group.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    PatternRewriter rewriter(&getContext());
    FusionCostEvaluator evaluator;

    for (Block &block : func.getBody()) {
      SmallVector<Operation *> originalOps;
      originalOps.reserve(block.getOperations().size());
      for (Operation &op : block.getOperations())
        originalOps.push_back(&op);

      for (Operation *op : originalOps) {
        if (op->getBlock() != &block)
          continue;

        auto seed = dyn_cast<linalg::GenericOp>(op);
        if (!seed || !npux::isCandidateSeedOp(seed))
          continue;

        FailureOr<npux::FusionCursor> cursor =
            npux::buildFusionCursorFromSeed(seed, evaluator);
        if (failed(cursor) || cursor->chainOps.empty())
          continue;

        rewriter.setInsertionPoint(seed);
        auto group = rewriter.create<npux::FusionGroupOp>(
            seed.getLoc(), cursor->tail->getResultTypes(), ValueRange{});
        npux::serializeFusionCursorToGroup(*cursor, group, rewriter);

        Block *groupBlock = new Block();
        group.getBodyRegion().push_back(groupBlock);
        rewriter.setInsertionPointToStart(groupBlock);

        llvm::SmallPtrSet<Operation *, 16> moveSet;
        for (linalg::GenericOp chainOp : cursor->chainOps) {
          moveSet.insert(chainOp.getOperation());
          collectMovableDependencies(
              chainOp.getOperation(), seed.getOperation(), &block, moveSet);
        }

        for (Operation *candidate : originalOps) {
          if (candidate->getBlock() != &block)
            continue;
          if (!moveSet.contains(candidate))
            continue;
          candidate->moveBefore(groupBlock, groupBlock->end());
        }

        auto yield = rewriter.create<npux::GroupYieldOp>(
            cursor->tail.getLoc(), cursor->tail->getResults());

        rewriter.replaceUsesWithIf(cursor->tail->getResults(), group.getResults(),
            [&](OpOperand &use) {
              return use.getOwner()->getParentRegion() != &group.getBodyRegion();
            });

        (void)yield;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuFusionPlanningPass() {
  return std::make_unique<NpuFusionPlanningPass>();
}
