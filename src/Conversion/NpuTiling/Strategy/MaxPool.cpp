#include "src/Conversion/NpuTiling/Strategy/MaxPool.hpp"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

using namespace mlir;

namespace {

// Rewrites tiled maxpool bodies back to the 4D form expected by the backend.
static void normalizeTiledMaxPoolOps(
    ArrayRef<Operation *> tiledOps, PatternRewriter &rewriter) {
  for (Operation *tiledOp : tiledOps) {
    auto genericOp = dyn_cast<linalg::GenericOp>(tiledOp);
    if (!genericOp || genericOp.getNumDpsInputs() != 2)
      continue;

    rewriter.setInsertionPoint(genericOp);
    Value realInput = genericOp.getDpsInputOperand(0)->get();
    Value outInit = genericOp.getDpsInitOperand(0)->get();
    SmallVector<utils::IteratorType> iteratorTypes(
        4, utils::IteratorType::parallel);
    auto n = rewriter.getAffineDimExpr(0);
    auto c = rewriter.getAffineDimExpr(1);
    auto oh = rewriter.getAffineDimExpr(2);
    auto ow = rewriter.getAffineDimExpr(3);
    auto inputMap = AffineMap::get(
        4, 0, {n, c, oh * 2, ow * 2}, rewriter.getContext());
    auto outputMap = rewriter.getMultiDimIdentityMap(4);

    auto new4DOp = rewriter.create<linalg::GenericOp>(
        genericOp.getLoc(), outInit.getType(), ValueRange{realInput},
        ValueRange{outInit}, ArrayRef<AffineMap>{inputMap, outputMap},
        iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
          b.create<linalg::YieldOp>(loc, args[0]);
        });
    new4DOp->setAttr("library_call", rewriter.getStringAttr("npu_maxpool"));
    new4DOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    new4DOp->setAttr("npu.tiled", rewriter.getUnitAttr());
    if (auto lookupAttr =
            genericOp->getAttrOfType<IntegerAttr>("__npux.fusion_lookup_id")) {
      new4DOp->setAttr("__npux.fusion_lookup_id", lookupAttr);
    }
    rewriter.replaceOp(genericOp, new4DOp.getResults());
  }
}

} // namespace

LogicalResult npux::tileMaxPoolWithRoot(
    const FusionCursor &cursor, PatternRewriter &rewriter) {
  return tileFusedChainOp(cursor.seed, cursor.tail, cursor.tailTileSizes,
      rewriter, cursor.seedInputDmaAnalyses, cursor.rootOutputDmaAnalysis,
      normalizeTiledMaxPoolOps);
}

LogicalResult npux::tileMaxPoolOp(
    linalg::GenericOp op, PatternRewriter &rewriter) {
  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  SmallVector<int64_t> rawTileSizes = costModel.getOptimalTileSizes(op);

  auto tilingInterfaceOp = cast<TilingInterface>(op.getOperation());
  SmallVector<OpFoldResult> tileSizes =
      getAsOpFoldResult(rewriter.getI64ArrayAttr(rawTileSizes));
  scf::SCFTilingOptions options;
  options.setTileSizes(tileSizes);

  FailureOr<scf::SCFTilingResult> tilingResult =
      scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

  normalizeTiledMaxPoolOps(tilingResult->tiledOps, rewriter);

  for (auto loop : tilingResult->loops)
    loop->setAttr("npu.target", rewriter.getStringAttr("npu"));

  SmallVector<Value> finalResults = tilingResult->replacements;
  for (int i = tilingResult->loops.size() - 1; i >= 0; --i) {
    auto loopOp = dyn_cast<scf::ForOp>(tilingResult->loops[i].getOperation());
    scf::ForOp partialIteration;
    if (succeeded(scf::peelForLoopAndSimplifyBounds(
            rewriter, loopOp, partialIteration))) {
      partialIteration->setAttr("npu.peeled_tail", rewriter.getUnitAttr());
      if (i == 0)
        finalResults = partialIteration->getResults();
    }
  }

  rewriter.replaceOp(op, finalResults);
  return success();
}
