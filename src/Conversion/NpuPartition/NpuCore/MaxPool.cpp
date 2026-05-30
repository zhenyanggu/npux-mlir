//==============================================================//
// src/Conversion/NpuPartition/NpuCore/MaxPool.cpp
// this file implements the conversion of MaxPool to npucore
// maxpool for NPU partitioning.
//==============================================================//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/NpuPartition/NpuCoreConversionHelper.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"

using namespace mlir;

namespace {

/// Read one scalar float constant used by QDQ.
static float getScalarFloat(Value v, float defaultVal = 1.0f) {
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue()))
      return static_cast<float>(floatAttr.getValueAsDouble());
  }
  if (auto constOp = v.getDefiningOp<ONNXConstantOp>()) {
    if (auto dense = dyn_cast<DenseElementsAttr>(constOp.getValueAttr()))
      return dense.getValues<float>()[0];
  }
  return defaultVal;
}

/// Read one scalar integer constant used by QDQ.
static int64_t getScalarInt(Value v, int64_t defaultVal = 0) {
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  }
  if (auto constOp = v.getDefiningOp<ONNXConstantOp>()) {
    if (auto dense = dyn_cast<DenseElementsAttr>(constOp.getValueAttr())) {
      if (dense.getElementType().isInteger(8))
        return dense.getValues<int8_t>()[0];
      if (dense.getElementType().isInteger(32))
        return dense.getValues<int32_t>()[0];
      if (dense.getElementType().isInteger(64))
        return dense.getValues<int64_t>()[0];
    }
  }
  return defaultVal;
}

/// Attach encoding=1 to a tensor value, materializing constants when needed.
static Value materializeToEncoding1(
    OpBuilder &builder, Location loc, Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return value;

  auto encodedType = npux::addEncoding1(type, builder);
  if (value.getDefiningOp<ONNXConstantOp>() ||
      value.getDefiningOp<arith::ConstantOp>()) {
    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < type.getRank(); ++i) {
      if (type.isDynamicDim(i))
        dynamicSizes.push_back(builder.create<tensor::DimOp>(loc, value, i));
    }
    Value alloc = builder.create<bufferization::AllocTensorOp>(
        loc, encodedType, dynamicSizes);
    return builder
        .create<bufferization::MaterializeInDestinationOp>(loc, value, alloc)
        .getResult();
  }

  value.setType(encodedType);
  return value;
}

/// Collect dynamic dimensions for the result tensor.
static SmallVector<Value> getDynamicSizes(
    OpBuilder &builder, Location loc, Value reference,
    ArrayRef<int64_t> outputShape) {
  SmallVector<Value> dynamicSizes;
  for (int64_t i = 0; i < static_cast<int64_t>(outputShape.size()); ++i) {
    if (ShapedType::isDynamic(outputShape[i]))
      dynamicSizes.push_back(builder.create<tensor::DimOp>(loc, reference, i));
  }
  return dynamicSizes;
}

/// Read a static int64 array attribute from ONNX with a fixed fallback length.
static SmallVector<int64_t> getIntArrayAttr(
    Operation *op, StringRef name, int64_t defaultVal, int size) {
  if (auto attr = op->getAttrOfType<ArrayAttr>(name)) {
    SmallVector<int64_t> values;
    for (Attribute value : attr)
      values.push_back(cast<IntegerAttr>(value).getInt());
    return values;
  }
  return SmallVector<int64_t>(size, defaultVal);
}

/// Lower ONNX MaxPool directly to npucore.maxpool without a dummy window.
struct MaxPoolToNpucore final
    : public OpConversionPattern<ONNXMaxPoolSingleOutOp> {
  using OpConversionPattern<ONNXMaxPoolSingleOutOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!npux::hasStrictQDQContext(op))
      return failure();

    auto dequantOp = op.getX().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantOp)
      return failure();

    if (!op.getResult().hasOneUse())
      return failure();
    auto quantOp =
        dyn_cast<ONNXQuantizeLinearOp>(*op.getResult().getUsers().begin());
    if (!quantOp)
      return failure();

    auto inputType = dyn_cast<RankedTensorType>(dequantOp.getX().getType());
    auto outputType = dyn_cast<RankedTensorType>(quantOp.getResult().getType());
    if (!inputType || !outputType)
      return failure();
    if (inputType.getRank() != 4 || outputType.getRank() != 4) {
      return rewriter.notifyMatchFailure(
          op, "npucore.maxpool currently requires rank-4 NCHW tensors");
    }

    SmallVector<int64_t> kernelShape =
        getIntArrayAttr(op, "kernel_shape", 1, 2);
    SmallVector<int64_t> strides = getIntArrayAttr(op, "strides", 1, 2);
    SmallVector<int64_t> dilations = getIntArrayAttr(op, "dilations", 1, 2);
    SmallVector<int64_t> pads = getIntArrayAttr(op, "pads", 0, 4);

    Value input = materializeToEncoding1(rewriter, op.getLoc(), dequantOp.getX());
    auto encodedOutputType = npux::addEncoding1(outputType, rewriter);
    SmallVector<Value> dynamicSizes =
        getDynamicSizes(rewriter, op.getLoc(), input, outputType.getShape());
    Value outputInit = rewriter.create<bufferization::AllocTensorOp>(
        op.getLoc(), encodedOutputType, dynamicSizes);

    float inScale = getScalarFloat(dequantOp.getXScale());
    int64_t inZp = getScalarInt(dequantOp.getXZeroPoint());
    float outScale = getScalarFloat(quantOp.getYScale());
    int64_t outZp = getScalarInt(quantOp.getYZeroPoint());

    auto maxpoolOp = rewriter.create<npucore::MaxPoolOp>(op.getLoc(),
        TypeRange{encodedOutputType}, ValueRange{input}, ValueRange{outputInit},
        rewriter.getF32FloatAttr(inScale), rewriter.getI32IntegerAttr(inZp),
        rewriter.getF32FloatAttr(outScale), rewriter.getI16IntegerAttr(outZp),
        rewriter.getI64ArrayAttr(kernelShape), rewriter.getI64ArrayAttr(strides),
        rewriter.getI64ArrayAttr(dilations), rewriter.getI64ArrayAttr(pads));
    maxpoolOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

    quantOp.getResult().setType(maxpoolOp.getResultTensors().front().getType());
    rewriter.replaceOp(quantOp, maxpoolOp.getResultTensors().front());
    rewriter.eraseOp(op);
    if (dequantOp->use_empty())
      rewriter.eraseOp(dequantOp);
    return success();
  }
};

} // namespace

void npux::populateNpucoreMaxPoolPatterns(RewritePatternSet &patterns) {
  patterns.add<MaxPoolToNpucore>(patterns.getContext());
}
