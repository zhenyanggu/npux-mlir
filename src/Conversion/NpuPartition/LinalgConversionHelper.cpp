//==============================================================
//src/Conversion/NpuPartition/LinalgConversionHelper.cpp
// this file provides helper functions for NPU partitioning conversions.
//==============================================================

#include "src/Conversion/NpuPartition/LinalgConversionHelper.hpp" 
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace npux {

  
DenseElementsAttr getConstAttrFromOperand(Operation *op, int operandIndex) {
  if (operandIndex >= op->getNumOperands()) 
    return nullptr;

  Value val = op->getOperand(operandIndex);
  
  if (!val || llvm::isa<NoneType>(val.getType())) {
    return nullptr;
  }

  Operation *defOp = val.getDefiningOp();
  if (auto constOp = llvm::dyn_cast_or_null<ONNXConstantOp>(defOp)) {
    if (auto valueAttr = dyn_cast<DenseElementsAttr>(constOp.getValueAttr())) {
      return valueAttr;
    }
  }
  return nullptr;
}

SmallVector<Value> getDynamicSizes(
    OpBuilder &rewriter, Location loc, Value input, ArrayRef<int64_t> shape) {
  SmallVector<Value> dynSizes;
  for (int i = 0; i < shape.size(); ++i) {
    if (ShapedType::isDynamic(shape[i])) {
      dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
    }
  }
  return dynSizes;
}

void populateLinalgConversionPatterns(RewritePatternSet &patterns)
{
  populateLinalgLayerNormPattern(patterns);
  populateLinalgConvPattern(patterns);
  populateLinalgUnaryPatterns(patterns);
  populateLinalgGemmPattern(patterns);
  populateLinalgResamplePatterns(patterns);
  populateLinalgTransposePattern(patterns);
}

} // namespace npux