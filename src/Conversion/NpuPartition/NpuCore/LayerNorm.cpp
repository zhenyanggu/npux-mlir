//==============================================================//
// src/Conversion/NpuPartition/NpuCore/LayerNorm.cpp
// this file implements the conversion of LayerNorm to npucore for NPU
// partitioning.
//==============================================================//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/NpuPartition/NpuCoreConversionHelper.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"

using namespace mlir;

namespace {

/// Lower ONNX LayerNorm directly to the native npucore.layernorm op.
struct LayerNormToNpucore final
    : public OpConversionPattern<ONNXLayerNormalizationOp> {
  using OpConversionPattern<ONNXLayerNormalizationOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXLayerNormalizationOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!op.getMean().use_empty() || !op.getInvStdDev().use_empty())
      return rewriter.notifyMatchFailure(
          op, "NPU LayerNorm only supports inference (Y output only).");

    Value originInput = op.getX();
    auto dequantOp = originInput.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantOp)
      return failure();

    Value quantizedInput = dequantOp.getX();
    auto inputType = dyn_cast<RankedTensorType>(quantizedInput.getType());
    if (!inputType)
      return failure();

    if (!op.getY().hasOneUse())
      return failure();
    auto quantOp =
        dyn_cast<ONNXQuantizeLinearOp>(*op.getY().getUsers().begin());
    if (!quantOp)
      return failure();

    auto outputType = dyn_cast<RankedTensorType>(quantOp.getResult().getType());
    if (!outputType)
      return failure();

    auto inParams = npux::getScalarQuantParams(dequantOp);
    auto outParams = npux::getScalarQuantParams(quantOp);

    RankedTensorType inputType1 = npux::addEncoding1(inputType, rewriter);
    RankedTensorType outputType1 = npux::addEncoding1(outputType, rewriter);

    if (quantizedInput.getDefiningOp<ONNXConstantOp>()) {
      SmallVector<Value> copyDynamicSizes = npux::getDynamicSizes(
          rewriter, op.getLoc(), quantizedInput, inputType1.getShape());
      Value copyAlloc = rewriter.create<bufferization::AllocTensorOp>(
          op.getLoc(), inputType1, copyDynamicSizes);
      quantizedInput =
          rewriter
              .create<bufferization::MaterializeInDestinationOp>(
                  op.getLoc(), quantizedInput, copyAlloc)
              .getResult();
    } else {
      quantizedInput.setType(inputType1);
    }

    SmallVector<Value> dynamicSizes = npux::getDynamicSizes(
        rewriter, op.getLoc(), quantizedInput, inputType1.getShape());
    Value initTensor = rewriter.create<bufferization::AllocTensorOp>(
        op.getLoc(), outputType1, dynamicSizes);

    auto layerNormOp = rewriter.create<npucore::LayerNormOp>(op.getLoc(),
        TypeRange{outputType1}, ValueRange{quantizedInput},
        ValueRange{initTensor}, rewriter.getF32FloatAttr(inParams.scale),
        rewriter.getIntegerAttr(rewriter.getI32Type(), inParams.zeroPoint),
        rewriter.getF32FloatAttr(outParams.scale),
        rewriter.getIntegerAttr(rewriter.getI16Type(), outParams.zeroPoint),
        rewriter.getI64IntegerAttr(op.getAxis()),
        rewriter.getF32FloatAttr(op.getEpsilon().convertToFloat()));
    layerNormOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

    quantOp.getResult().setType(layerNormOp.getResultTensors()[0].getType());
    rewriter.replaceOp(quantOp, layerNormOp.getResultTensors()[0]);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void npux::populateNpucoreLayerNormPatterns(RewritePatternSet &patterns) {
  patterns.add<LayerNormToNpucore>(patterns.getContext());
}
