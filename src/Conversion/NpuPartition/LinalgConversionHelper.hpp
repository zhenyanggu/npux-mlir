//==============================================================
//src/Conversion/NpuPartition/LinalgConversionHelper.hpp
// this file provides helper functions for NPU partitioning conversions.
//==============================================================


#pragma once

#include "mlir/Support/LLVM.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "mlir/IR/BuiltinAttributes.h"

namespace npux {

struct QuantizationParam {
  float scale;
  int64_t zeroPoint;
};

mlir::DenseElementsAttr getConstAttrFromOperand(mlir::Operation *op, int operandIndex);

template <typename OpT>
QuantizationParam getScalarQuantParams(OpT op) {
  static_assert(
      std::is_same<OpT, mlir::ONNXQuantizeLinearOp>::value ||
      std::is_same<OpT, mlir::ONNXDequantizeLinearOp>::value,
      "Only supports Q/DQ ops"
  );

  QuantizationParam params;
  
  auto scaleAttr = getConstAttrFromOperand(op, 1);
  assert(scaleAttr && scaleAttr.isSplat() && "Invalid Scale");
  params.scale = scaleAttr.template getSplatValue<float>();

  auto zpAttr = getConstAttrFromOperand(op, 2);
  params.zeroPoint = 0;

  if (zpAttr) {
    assert(zpAttr.isSplat() && "Invalid ZeroPoint");
    mlir::Type zpType = zpAttr.getElementType();
    if (zpType.isUnsignedInteger(8)) {
      params.zeroPoint = (int64_t)zpAttr.template getSplatValue<uint8_t>();
    } else if (zpType.isInteger(8)) {
      params.zeroPoint = (int64_t)zpAttr.template getSplatValue<int8_t>();
    } else if (zpType.isInteger(32)) {
      params.zeroPoint = (int64_t)zpAttr.template getSplatValue<int32_t>();
    }
  }

  return params;
}

mlir::SmallVector<mlir::Value> getDynamicSizes(
    mlir::OpBuilder &rewriter, mlir::Location loc, mlir::Value input, 
    mlir::ArrayRef<int64_t> shape);


void populateLinalgConversionPatterns(mlir::RewritePatternSet &patterns);

void populateLinalgLayerNormPattern(mlir::RewritePatternSet &patterns);

void populateLinalgConvPattern(mlir::RewritePatternSet &patterns);

void populateLinalgUnaryPatterns(mlir::RewritePatternSet &patterns);

void populateLinalgMataddPattern(mlir::RewritePatternSet &patterns);

void populateLinalgGemmPattern(mlir::RewritePatternSet &patterns);

void populateLinalgResamplePatterns(mlir::RewritePatternSet &patterns);

void populateLinalgTransposePattern(mlir::RewritePatternSet &patterns);

}// namespace npux

