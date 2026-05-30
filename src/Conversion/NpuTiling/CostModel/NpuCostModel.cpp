
//======================================================
// src/Conversion/NpuTiling/CostModel/NpuCostModel.cpp
// This file implements the NPUCostModel class, which provides methods to
// calculate optimal tile sizes for different types of operations (Matmul,
// Conv2D, Elementwise based on the NPU's hardware constraints like SRAM size
// and number of MACs).
//======================================================
#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
namespace npux {

llvm::SmallVector<int64_t> NPUCostModel::getOptimalTileSizes(
    npucore::ConvOp op) {
  return getConv2DTileSizes(op);
}

llvm::SmallVector<int64_t> NPUCostModel::getOptimalTileSizes(
    npucore::MatMulOp op) {
  return getGemmTileSizes(op);
}

llvm::SmallVector<int64_t> NPUCostModel::getOptimalTileSizes(
    npucore::GeluOp op) {
  return getGeluTileSizes(op);
}

llvm::SmallVector<int64_t> NPUCostModel::getOptimalTileSizes(
    npucore::SoftmaxOp op) {
  return getSoftmaxTileSizes(op);
}

llvm::SmallVector<int64_t> NPUCostModel::getOptimalTileSizes(
    npucore::LayerNormOp op) {
  return getLayerNormTileSizes(op);
}

llvm::SmallVector<int64_t> NPUCostModel::getOptimalTileSizes(
    npucore::MatAddOp op) {
  return getMatAddTileSizes(op);
}

llvm::SmallVector<int64_t> NPUCostModel::getOptimalTileSizes(
    npucore::MaxPoolOp op) {
  return getMaxPoolTileSizes(op);
}

llvm::SmallVector<int64_t> NPUCostModel::getOptimalTileSizes(
    npucore::TransposeOp op) {
  return getTransposeTileSizes(op);
}

llvm::SmallVector<int64_t> NPUCostModel::getOptimalTileSizes(
    npucore::LayoutNchwToNchwc32Op op) {
  return getLayoutPackTileSizes(op);
}

llvm::SmallVector<int64_t> NPUCostModel::getOptimalTileSizes(
    npucore::LayoutNchwc32ToNchwOp op) {
  return getLayoutUnpackTileSizes(op);
}

} // namespace npux
