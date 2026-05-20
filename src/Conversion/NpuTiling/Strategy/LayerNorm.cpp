#include "src/Conversion/NpuTiling/Strategy/LayerNorm.hpp"

#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

using namespace mlir;

namespace {

static SmallVector<int64_t> getFullOutputTile(linalg::GenericOp op) {
  if (op.getNumDpsInits() == 0)
    return {};
  auto outputType =
      dyn_cast<RankedTensorType>(op.getDpsInitOperand(0)->get().getType());
  if (!outputType || !outputType.hasStaticShape())
    return {};
  return SmallVector<int64_t>(
      outputType.getShape().begin(), outputType.getShape().end());
}

} // namespace

LogicalResult npux::tileLayerNormWithRoot(
    const FusionCursor &cursor, PatternRewriter &rewriter) {
  return tileFusedChainOp(
      cursor.seed, cursor.tail, cursor.tailTileSizes, rewriter,
      cursor.seedInputDmaAnalyses, cursor.rootOutputDmaAnalysis);
}

LogicalResult npux::tileLayerNormOp(
    linalg::GenericOp op, PatternRewriter &rewriter) {
  return tileStandaloneOp(op, getFullOutputTile(op), rewriter);
}
