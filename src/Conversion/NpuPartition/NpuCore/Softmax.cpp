//==============================================================//
// src/Conversion/NpuPartition/NpuCore/Softmax.cpp
// this file implements the conversion of Softmax to npucore for NPU
// partitioning.
//==============================================================//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/NpuPartition/NpuCoreConversionHelper.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"

using namespace mlir;

namespace {

/// Lower ONNX Softmax directly to the native npucore.softmax op.
struct SoftmaxToNpucore final : public OpConversionPattern<ONNXSoftmaxOp> {
  using OpConversionPattern<ONNXSoftmaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXSoftmaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value originInput = op.getInput();
    auto dequantOp = originInput.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantOp)
      return failure();

    Value quantizedInput = dequantOp.getX();
    auto inputType = dyn_cast<RankedTensorType>(quantizedInput.getType());
    if (!inputType)
      return failure();

    if (!op.getResult().hasOneUse())
      return failure();
    auto quantOp =
        dyn_cast<ONNXQuantizeLinearOp>(*op.getResult().getUsers().begin());
    if (!quantOp)
      return failure();

    auto outputType = dyn_cast<RankedTensorType>(quantOp.getResult().getType());
    if (!outputType)
      return failure();

    auto inParams = npux::getScalarQuantParams(dequantOp);
    auto outParams = npux::getScalarQuantParams(quantOp);

    RankedTensorType inputType1 = npux::addEncoding1(inputType, rewriter);
    RankedTensorType outputType1 = npux::addEncoding1(outputType, rewriter);
    quantizedInput.setType(inputType1);

    SmallVector<Value> dynamicSizes = npux::getDynamicSizes(
        rewriter, op.getLoc(), quantizedInput, inputType1.getShape());
    Value initTensor = rewriter.create<bufferization::AllocTensorOp>(
        op.getLoc(), outputType1, dynamicSizes);

    auto softmaxOp = rewriter.create<npucore::SoftmaxOp>(op.getLoc(),
        TypeRange{outputType1}, ValueRange{quantizedInput},
        ValueRange{initTensor}, rewriter.getF32FloatAttr(inParams.scale),
        rewriter.getIntegerAttr(rewriter.getI32Type(), inParams.zeroPoint),
        rewriter.getF32FloatAttr(outParams.scale),
        rewriter.getIntegerAttr(rewriter.getI16Type(), outParams.zeroPoint),
        rewriter.getI64IntegerAttr(op.getAxis()));
    softmaxOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

    quantOp.getResult().setType(softmaxOp.getResultTensors()[0].getType());
    rewriter.replaceOp(quantOp, softmaxOp.getResultTensors()[0]);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void npux::populateNpucoreSoftmaxPatterns(RewritePatternSet &patterns) {
  patterns.add<SoftmaxToNpucore>(patterns.getContext());
}
