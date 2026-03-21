//==============================================================
//src/Conversion/NpuPartition/LinalgConversionHelper.cpp
// this file provides helper functions for NPU partitioning conversions.
//==============================================================

#include "src/Conversion/NpuPartition/LinalgConversionHelper.hpp" 
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace npux {

namespace {

constexpr llvm::StringLiteral kLayerNameAttr = "npu.layer_name";
constexpr llvm::StringLiteral kLayerKindAttr = "npu.layer_kind";
constexpr llvm::StringLiteral kFusedOpsAttr = "npu.fused_ops";
constexpr llvm::StringLiteral kOriginOpTypeAttr = "npu.origin_op_type";
constexpr llvm::StringLiteral kNodeNameAttr = "onnx_node_name";

static std::string sanitizeProfileOpType(llvm::StringRef opName) {
  llvm::StringRef shortName = opName.rsplit('.').second;
  if (shortName.empty())
    shortName = opName;
  if (shortName.starts_with("ONNX"))
    shortName = shortName.drop_front(4);
  if (shortName.ends_with("Op"))
    shortName = shortName.drop_back(2);
  return shortName.str();
}

} // namespace

  
DenseElementsAttr getConstAttrFromOperand(Operation *op, int operandIndex) {
  if (operandIndex >= op->getNumOperands()) 
    return nullptr;

  Value val = op->getOperand(operandIndex);
  
  if (!val || llvm::isa<NoneType>(val.getType())) {
    return nullptr;
  }

  Operation *defOp = val.getDefiningOp();
  if (auto constOp = dyn_cast_or_null<ONNXConstantOp>(defOp)) {
    if (auto valueAttr = dyn_cast<DenseElementsAttr>(constOp.getValueAttr())) {
      return valueAttr;
    }
  }
  return nullptr;
}

std::string getNpuProfileOpTypeName(Operation *op) {
  if (!op)
    return "unknown";
  return sanitizeProfileOpType(op->getName().getStringRef());
}

std::string getNpuProfileLayerName(Operation *op) {
  if (!op)
    return "unknown";
  if (auto attr = op->getAttrOfType<StringAttr>(kLayerNameAttr))
    return attr.getValue().str();
  if (auto attr = op->getAttrOfType<StringAttr>(kNodeNameAttr))
    return attr.getValue().str();
  return getNpuProfileOpTypeName(op);
}

void setNpuProfileAttrs(Operation *target, Operation *source, Builder &builder,
    llvm::StringRef layerName, llvm::StringRef layerKind,
    llvm::ArrayRef<llvm::StringRef> fusedOps) {
  if (!target)
    return;

  std::string finalLayerName = layerName.str();
  if (finalLayerName.empty() && source) {
    if (auto attr = source->getAttrOfType<StringAttr>(kNodeNameAttr))
      finalLayerName = attr.getValue().str();
  }
  if (finalLayerName.empty() && source)
    finalLayerName = getNpuProfileLayerName(source);
  if (finalLayerName.empty())
    finalLayerName = "unknown";

  target->setAttr(kLayerNameAttr, builder.getStringAttr(finalLayerName));

  if (!layerKind.empty())
    target->setAttr(kLayerKindAttr, builder.getStringAttr(layerKind));

  SmallVector<Attribute> fusedAttrs;
  fusedAttrs.reserve(fusedOps.size());
  for (llvm::StringRef fusedOp : fusedOps)
    fusedAttrs.push_back(builder.getStringAttr(fusedOp));
  target->setAttr(kFusedOpsAttr, builder.getArrayAttr(fusedAttrs));

  std::string originOpType = source ? getNpuProfileOpTypeName(source) : "unknown";
  target->setAttr(kOriginOpTypeAttr, builder.getStringAttr(originOpType));

  if (source) {
    if (Attribute nodeNameAttr = source->getAttr(kNodeNameAttr))
      target->setAttr(kNodeNameAttr, nodeNameAttr);
    if (source->hasAttr("npu_qdq_warning_emitted")) {
      target->setAttr(
          "npu_qdq_warning_emitted", UnitAttr::get(builder.getContext()));
    }
  }
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
  populateLinalgBinaryPatterns(patterns);
  populateLinalgGemmPattern(patterns);
  populateLinalgResamplePatterns(patterns);
  populateLinalgTransposePattern(patterns);
}

} // namespace npux
