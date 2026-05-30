//==============================================================
// src/Conversion/NpuPartition/NpuCoreConversionHelper.cpp
// this file provides helper functions for npucore partitioning conversions.
//==============================================================

#include "src/Conversion/NpuPartition/NpuCoreConversionHelper.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace npux {

static float getScalarFloat(Value value, float defaultValue = 1.0f) {
  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto floatAttr = dyn_cast<FloatAttr>(constantOp.getValue()))
      return static_cast<float>(floatAttr.getValueAsDouble());
  }

  if (auto constantOp = value.getDefiningOp<ONNXConstantOp>()) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValueAttr()))
      return denseAttr.getValues<float>()[0];
  }

  return defaultValue;
}

static int64_t getScalarInt(Value value, int64_t defaultValue = 0) {
  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue()))
      return intAttr.getInt();
  }

  if (auto constantOp = value.getDefiningOp<ONNXConstantOp>()) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValueAttr())) {
      if (denseAttr.getElementType().isInteger(8))
        return denseAttr.getValues<int8_t>()[0];
      if (denseAttr.getElementType().isInteger(16))
        return denseAttr.getValues<int16_t>()[0];
      if (denseAttr.getElementType().isInteger(32))
        return denseAttr.getValues<int32_t>()[0];
      if (denseAttr.getElementType().isInteger(64))
        return denseAttr.getValues<int64_t>()[0];
    }
  }

  return defaultValue;
}

bool hasStrictQDQContext(Operation *op) {
  if (op->getNumOperands() == 0 || op->getNumResults() == 0)
    return false;

  Value input = op->getOperand(0);
  bool hasDq = (input.getDefiningOp<ONNXDequantizeLinearOp>() != nullptr);

  Value result = op->getResult(0);
  bool hasQ = false;
  if (result.hasOneUse() &&
      isa<ONNXQuantizeLinearOp>(*result.getUsers().begin())) {
    hasQ = true;
  }

  if (hasDq && hasQ)
    return true;

  StringRef warningAttrName = "npu_qdq_warning_emitted";
  if (!op->hasAttr(warningAttrName)) {
    op->emitWarning() << "Operation '" << op->getName()
                      << "' lacks strict Dequantize -> Op -> Quantize context. "
                      << "Skipping conversion to npucore.";
    op->setAttr(warningAttrName, UnitAttr::get(op->getContext()));
  }

  return false;
}

bool supportsNpucoreMatAdd(ONNXAddOp op) {
  if (!hasStrictQDQContext(op))
    return false;

  auto dequantOpA = op.getA().getDefiningOp<ONNXDequantizeLinearOp>();
  auto dequantOpB = op.getB().getDefiningOp<ONNXDequantizeLinearOp>();
  if (!dequantOpA || !dequantOpB)
    return false;

  auto quantizedInputA = dyn_cast<RankedTensorType>(dequantOpA.getX().getType());
  auto quantizedInputB = dyn_cast<RankedTensorType>(dequantOpB.getX().getType());
  if (!quantizedInputA || !quantizedInputB)
    return false;

  if (quantizedInputA.getShape() != quantizedInputB.getShape())
    return false;

  return true;
}

RankedTensorType addEncoding1(RankedTensorType type, OpBuilder &builder) {
  if (type.getEncoding()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(type.getEncoding())) {
      if (intAttr.getInt() == 1)
        return type;
    }
  }
  return RankedTensorType::get(
      type.getShape(), type.getElementType(), builder.getI64IntegerAttr(1));
}

ScalarQuantParams getScalarQuantParams(ONNXDequantizeLinearOp op) {
  return {getScalarFloat(op.getXScale()), getScalarInt(op.getXZeroPoint())};
}

ScalarQuantParams getScalarQuantParams(ONNXQuantizeLinearOp op) {
  return {getScalarFloat(op.getYScale()), getScalarInt(op.getYZeroPoint())};
}

SmallVector<Value> getDynamicSizes(OpBuilder &builder, Location loc,
    Value reference, ArrayRef<int64_t> outputShape) {
  SmallVector<Value> dynamicSizes;
  for (int64_t i = 0; i < static_cast<int64_t>(outputShape.size()); ++i) {
    if (ShapedType::isDynamic(outputShape[i]))
      dynamicSizes.push_back(builder.create<tensor::DimOp>(loc, reference, i));
  }
  return dynamicSizes;
}

void populateNpucoreConversionPatterns(RewritePatternSet &patterns) {
  populateNpucoreConvPatterns(patterns);
  populateNpucoreGeluPatterns(patterns);
  populateNpucoreSoftmaxPatterns(patterns);
  populateNpucoreLayerNormPatterns(patterns);
  populateNpucoreMatAddPatterns(patterns);
  populateNpucoreMatMulPatterns(patterns);
  populateNpucoreMaxPoolPatterns(patterns);
}

} // namespace npux
