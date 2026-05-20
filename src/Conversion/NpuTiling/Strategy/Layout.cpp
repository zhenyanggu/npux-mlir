#include "src/Conversion/NpuTiling/Strategy/Layout.hpp"

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

using namespace mlir;

LogicalResult npux::tileLayoutWithRoot(
    const FusionCursor &cursor, PatternRewriter &rewriter) {
  return tileFusedChainOp(
      cursor.seed, cursor.tail, cursor.tailTileSizes, rewriter,
      cursor.seedInputDmaAnalyses, cursor.rootOutputDmaAnalysis);
}

LogicalResult npux::tileLayoutOp(
    linalg::GenericOp op, PatternRewriter &rewriter) {
  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  return tileStandaloneOp(op, costModel.getOptimalTileSizes(op), rewriter);
}
