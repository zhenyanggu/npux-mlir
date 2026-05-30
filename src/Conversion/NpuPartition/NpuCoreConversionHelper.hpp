//==============================================================
// src/Conversion/NpuPartition/NpuCoreConversionHelper.hpp
// this file provides helper functions for npucore partitioning conversions.
//==============================================================

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace npux {

/// Scalar quantization parameters extracted from QDQ helper ops.
struct ScalarQuantParams {
  float scale;
  int64_t zeroPoint;
};

/// Return true when an op is enclosed in the strict DQ -> Op -> Q pattern.
bool hasStrictQDQContext(mlir::Operation *op);

/// Return true when ONNX Add matches the current npucore matadd subset.
bool supportsNpucoreMatAdd(mlir::ONNXAddOp op);

/// Add DRAM encoding=1 to a ranked tensor type for NPU tensor semantics.
mlir::RankedTensorType addEncoding1(
    mlir::RankedTensorType type, mlir::OpBuilder &builder);

/// Extract scalar scale / zero-point from ONNX DequantizeLinear.
ScalarQuantParams getScalarQuantParams(mlir::ONNXDequantizeLinearOp op);

/// Extract scalar scale / zero-point from ONNX QuantizeLinear.
ScalarQuantParams getScalarQuantParams(mlir::ONNXQuantizeLinearOp op);

/// Collect dynamic tensor dimensions in result-shape order.
mlir::SmallVector<mlir::Value> getDynamicSizes(mlir::OpBuilder &builder,
    mlir::Location loc, mlir::Value reference,
    mlir::ArrayRef<int64_t> outputShape);

/// Populate native npucore conversion patterns.
void populateNpucoreConversionPatterns(mlir::RewritePatternSet &patterns);

/// Populate ONNX Gelu -> npucore.gelu conversion pattern.
void populateNpucoreGeluPatterns(mlir::RewritePatternSet &patterns);

/// Populate ONNX Softmax -> npucore.softmax conversion pattern.
void populateNpucoreSoftmaxPatterns(mlir::RewritePatternSet &patterns);

/// Populate ONNX LayerNorm -> npucore.layernorm conversion pattern.
void populateNpucoreLayerNormPatterns(mlir::RewritePatternSet &patterns);

/// Populate ONNX Add -> npucore.matadd conversion pattern.
void populateNpucoreMatAddPatterns(mlir::RewritePatternSet &patterns);

/// Populate ONNX Gemm/MatMul -> npucore.matmul conversion pattern.
void populateNpucoreMatMulPatterns(mlir::RewritePatternSet &patterns);

/// Populate ONNX MaxPool -> npucore.maxpool conversion pattern.
void populateNpucoreMaxPoolPatterns(mlir::RewritePatternSet &patterns);

/// Populate ONNX Conv -> npucore.conv conversion pattern.
void populateNpucoreConvPatterns(mlir::RewritePatternSet &patterns);

} // namespace npux
