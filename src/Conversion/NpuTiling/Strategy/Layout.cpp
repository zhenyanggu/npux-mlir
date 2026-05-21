#include "src/Conversion/NpuTiling/Strategy/Layout.hpp"

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

using namespace mlir;

namespace {

static SmallVector<int64_t> getLayoutTileSizes(Operation *op) {
  StringRef libCall = op->getAttrOfType<StringAttr>("library_call").getValue();
  if (libCall == "npu_layout_nchw_to_nchwc32") {
    RankedTensorType inputType;
    if (auto packOp = dyn_cast<linalg::PackOp>(op))
      inputType = packOp.getSourceType();
    else
      inputType = dyn_cast<RankedTensorType>(
          cast<linalg::GenericOp>(op).getDpsInputOperand(0)->get().getType());
    if (!inputType || !inputType.hasStaticShape())
      return {};
    SmallVector<int64_t> inputShape(
        inputType.getShape().begin(), inputType.getShape().end());
    if (inputShape.size() == 5)
      inputShape.pop_back();
    return inputShape;
  }

  if (libCall == "npu_layout_nchwc32_to_nchw") {
    RankedTensorType outputType;
    if (auto unpackOp = dyn_cast<linalg::UnPackOp>(op))
      outputType = unpackOp.getDestType();
    else
      outputType = dyn_cast<RankedTensorType>(
          cast<linalg::GenericOp>(op).getDpsInitOperand(0)->get().getType());
    if (!outputType || !outputType.hasStaticShape())
      return {};
    return SmallVector<int64_t>(
        outputType.getShape().begin(), outputType.getShape().end());
  }

  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  return costModel.getOptimalTileSizes(cast<linalg::GenericOp>(op));
}

} // namespace

LogicalResult npux::tileLayoutWithRoot(
    const FusionCursor &cursor, PatternRewriter &rewriter) {
  return tileFusedChainOp(
      cursor.seed, cursor.tail, cursor.tailTileSizes, rewriter,
      cursor.seedInputDmaAnalyses, cursor.rootOutputDmaAnalysis);
}

LogicalResult npux::tileLayoutOp(
    Operation *op, PatternRewriter &rewriter) {
  return tileStandaloneOp(op, getLayoutTileSizes(op), rewriter);
}
