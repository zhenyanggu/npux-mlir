//==============================================================//
// src/Conversion/NpuPartition/NpuCore/MatAdd.cpp
// this file implements the conversion of Add to npucore matadd for NPU
// partitioning.
//==============================================================//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/NpuPartition/NpuCoreConversionHelper.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"

using namespace mlir;

namespace {

/// Lower ONNX Add directly to the native npucore.matadd op.
struct AddToNpucore final : public OpConversionPattern<ONNXAddOp> {
  using OpConversionPattern<ONNXAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value originInputA = op.getA();
    Value originInputB = op.getB();
    auto dequantOpA = originInputA.getDefiningOp<ONNXDequantizeLinearOp>();
    auto dequantOpB = originInputB.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantOpA || !dequantOpB)
      return failure();

    Value quantizedInputA = dequantOpA.getX();
    Value quantizedInputB = dequantOpB.getX();
    auto inputTypeA = dyn_cast<RankedTensorType>(quantizedInputA.getType());
    auto inputTypeB = dyn_cast<RankedTensorType>(quantizedInputB.getType());
    if (!inputTypeA || !inputTypeB)
      return failure();
    if (inputTypeA.getShape() != inputTypeB.getShape())
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

    auto inParamsA = npux::getScalarQuantParams(dequantOpA);
    auto inParamsB = npux::getScalarQuantParams(dequantOpB);
    auto outParams = npux::getScalarQuantParams(quantOp);

    RankedTensorType inputTypeA1 = npux::addEncoding1(inputTypeA, rewriter);
    RankedTensorType inputTypeB1 = npux::addEncoding1(inputTypeB, rewriter);
    RankedTensorType outputType1 = npux::addEncoding1(outputType, rewriter);

    auto processInput = [&](Value input, RankedTensorType targetType) -> Value {
      if (input.getDefiningOp<ONNXConstantOp>()) {
        SmallVector<Value> dynSizes = npux::getDynamicSizes(
            rewriter, op.getLoc(), input, targetType.getShape());
        Value alloc = rewriter.create<bufferization::AllocTensorOp>(
            op.getLoc(), targetType, dynSizes);
        return rewriter
            .create<bufferization::MaterializeInDestinationOp>(
                op.getLoc(), input, alloc)
            .getResult();
      }
      input.setType(targetType);
      return input;
    };

    quantizedInputA = processInput(quantizedInputA, inputTypeA1);
    quantizedInputB = processInput(quantizedInputB, inputTypeB1);

    SmallVector<Value> dynamicSizes = npux::getDynamicSizes(
        rewriter, op.getLoc(), quantizedInputA, inputTypeA1.getShape());
    Value initTensor = rewriter.create<bufferization::AllocTensorOp>(
        op.getLoc(), outputType1, dynamicSizes);

    auto mataddOp = rewriter.create<npucore::MatAddOp>(op.getLoc(),
        TypeRange{outputType1}, ValueRange{quantizedInputA, quantizedInputB},
        ValueRange{initTensor},
        rewriter.getF32FloatAttr(inParamsA.scale),
        rewriter.getIntegerAttr(rewriter.getI32Type(), inParamsA.zeroPoint),
        rewriter.getF32FloatAttr(inParamsB.scale),
        rewriter.getIntegerAttr(rewriter.getI32Type(), inParamsB.zeroPoint),
        rewriter.getF32FloatAttr(outParams.scale),
        rewriter.getIntegerAttr(rewriter.getI16Type(), outParams.zeroPoint));
    mataddOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

    quantOp.getResult().setType(mataddOp.getResultTensors()[0].getType());
    rewriter.replaceOp(quantOp, mataddOp.getResultTensors()[0]);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void npux::populateNpucoreMatAddPatterns(RewritePatternSet &patterns) {
  patterns.add<AddToNpucore>(patterns.getContext());
}
