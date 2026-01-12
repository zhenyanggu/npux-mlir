#include "src/Conversion/NpuPartition/NpuQuantHelper.hpp" 
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Casting.h"

namespace npux {

  
mlir::DenseElementsAttr getConstAttrFromOperand(mlir::Operation *op, int operandIndex) {
  if (operandIndex >= op->getNumOperands()) 
    return nullptr;

  mlir::Value val = op->getOperand(operandIndex);
  
  if (!val || llvm::isa<mlir::NoneType>(val.getType())) {
    return nullptr;
  }

  mlir::Operation *defOp = val.getDefiningOp();
  if (auto constOp = llvm::dyn_cast_or_null<mlir::ONNXConstantOp>(defOp)) {
    if (auto valueAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValueAttr())) {
      return valueAttr;
    }
  }
  return nullptr;
}


} // namespace npux