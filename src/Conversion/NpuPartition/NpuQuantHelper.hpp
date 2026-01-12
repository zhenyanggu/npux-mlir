#ifndef NPU_QUANT_HELPER_HPP
#define NPU_QUANT_HELPER_HPP

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

}// namespace npux


#endif // NPU_QUANT_HELPER_HPP