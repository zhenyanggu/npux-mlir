//=============================================================
// src/Conversion/NpuTiling/MaxPool.cpp
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "npu-tiling"
using namespace mlir;
using namespace npux;

namespace {

/// Query the cost model for the new 4D output-space maxpool op.
static SmallVector<int64_t> getMaxPoolTileSizes(npucore::MaxPoolOp op) {
  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  return costModel.getOptimalTileSizes(op);
}

/// Tile npucore.maxpool directly on its 4D output iteration space.
struct NpuMaxPoolTilingPattern : public OpRewritePattern<npucore::MaxPoolOp> {
  using OpRewritePattern<npucore::MaxPoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::MaxPoolOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.tiled"))
      return failure();
    if (!op.hasTensorSemantics())
      return failure();

    SmallVector<int64_t> rawTileSizes = getMaxPoolTileSizes(op);
    if (rawTileSizes.empty())
      return failure();

    auto tilingInterfaceOp = cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizes =
        getAsOpFoldResult(rewriter.getI64ArrayAttr(rawTileSizes));

    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);
    if (failed(tilingResult))
      return failure();

    for (Operation *tiledOp : tilingResult->tiledOps)
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
    for (auto loop : tilingResult->loops)
      loop->setAttr("npu.target", rewriter.getStringAttr("npu"));

    auto loops = tilingResult->loops;
    SmallVector<Value> finalResults = tilingResult->replacements;
    for (int i = loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp)
        continue;

      scf::ForOp partialIteration;
      LogicalResult status =
          scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);
      if (succeeded(status)) {
        partialIteration->setAttr("npu.peeled_tail", rewriter.getUnitAttr());
        if (i == 0)
          finalResults = partialIteration->getResults();
      }
    }

    rewriter.replaceOp(op, finalResults);
    return success();
  }
};

} // namespace

void npux::populateMaxPoolTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuMaxPoolTilingPattern>(context);
}
