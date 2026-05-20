//=============================================================================
// Materialize tiled fusion groups produced by the planning pass.
//=============================================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Conversion/NpuTiling/FusionAnalysis/NpuFusionPlanner.hpp"
#include "src/Conversion/NpuTiling/Strategy/Dispatcher.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

struct NpuFusionMaterializePass : public PassWrapper<NpuFusionMaterializePass,
                                     OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuFusionMaterializePass)

  StringRef getArgument() const override { return "npu-fusion-materialize"; }
  StringRef getDescription() const override {
    return "Materialize tiled fusion groups and insert DMA-aware lowering.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    PatternRewriter rewriter(context);

    for (Block &block : func.getBody()) {
      SmallVector<npux::FusionGroupOp> groups;
      for (Operation &op : block.getOperations()) {
        if (auto group = dyn_cast<npux::FusionGroupOp>(&op))
          groups.push_back(group);
      }

      for (npux::FusionGroupOp group : groups) {
        if (group->getBlock() != &block)
          continue;

        FailureOr<npux::FusionCursor> cursor =
            npux::deserializeFusionCursorFromGroup(group);
        if (failed(cursor))
          continue;

        Block &groupBlock = group.getBodyRegion().front();
        auto yieldOp = dyn_cast<npux::GroupYieldOp>(groupBlock.getTerminator());
        if (!yieldOp)
          continue;

        SmallVector<Value> yieldedValues(
            yieldOp.getOperands().begin(), yieldOp.getOperands().end());

        SmallVector<Operation *> bodyOps;
        for (Operation &op : groupBlock) {
          if (!isa<npux::GroupYieldOp>(op))
            bodyOps.push_back(&op);
        }

        for (Operation *bodyOp : bodyOps)
          bodyOp->moveBefore(group);

        rewriter.setInsertionPoint(group);
        rewriter.replaceOp(group, yieldedValues);

        rewriter.setInsertionPoint(cursor->seed);
        if (failed(npux::tileSeedOp(*cursor, rewriter))) {
          cursor->seed.emitRemark()
              << "NpuFusionMaterialize skipped unsupported seed "
              << cursor->seed->getAttrOfType<StringAttr>("library_call");
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuFusionMaterializePass() {
  return std::make_unique<NpuFusionMaterializePass>();
}
