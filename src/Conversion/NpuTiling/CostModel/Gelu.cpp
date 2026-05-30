//======================================================
// src/Conversion/NpuTiling/CostModel/Gelu.cpp
// This file implements the cost model for Gelu/Elementwise operations,
// calculating tile sizes based on SRAM capacity constraints.
//======================================================

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include <algorithm>

using namespace mlir;

namespace npux {

template <typename OpTy>
static llvm::SmallVector<int64_t> getElementwiseTileSizes(
    OpTy op, int64_t spmSizeBytes) {
  auto outputType = cast<ShapedType>(op.getOutputs().front().getType());
  int64_t rank = outputType.getRank();
  SmallVector<int64_t> loopRanges(
      outputType.getShape().begin(), outputType.getShape().end());
  llvm::SmallVector<int64_t> tileSizes(rank, 1);
  if (rank == 0)
    return tileSizes;

  int64_t bitWidth = outputType.getElementType().getIntOrFloatBitWidth();
  int64_t bytesPerElem = std::max<int64_t>(1, bitWidth / 8);
  int64_t numOperands = op.getNumDpsInputs() + op.getNumDpsInits();
  int64_t maxElems = spmSizeBytes / (numOperands * bytesPerElem);

  if (maxElems > 0) {
    int64_t remainingElems = maxElems;
    for (int i = rank - 1; i >= 0; --i) {
      int64_t dimSize = loopRanges[i];

      if (dimSize <= 0) dimSize = 1;

      if (remainingElems >= dimSize) {
        tileSizes[i] = dimSize;
        remainingElems /= dimSize;
      } else {
        int64_t tile = (remainingElems / 16) * 16;
        if (tile == 0) tile = remainingElems; 

        tileSizes[i] = tile;
        remainingElems = 1; 
        break;              
      }
    }
  }
  return tileSizes;
}

llvm::SmallVector<int64_t> NPUCostModel::getGeluTileSizes(npucore::GeluOp op) {
  auto tileSizes = getElementwiseTileSizes(op, this->hw.spmSizeBytes);
  llvm::errs() << "[CostModel] Gelu(npucore): Tile=[";
  for (size_t i = 0; i < tileSizes.size(); ++i)
    llvm::errs() << tileSizes[i] << (i + 1 == tileSizes.size() ? "" : ", ");
  llvm::errs() << "]\n";
  return tileSizes;
}

llvm::SmallVector<int64_t> NPUCostModel::getSoftmaxTileSizes(
    npucore::SoftmaxOp op) {
  auto tileSizes = getElementwiseTileSizes(op, this->hw.spmSizeBytes);
  llvm::errs() << "[CostModel] Softmax(npucore): Tile=[";
  for (size_t i = 0; i < tileSizes.size(); ++i)
    llvm::errs() << tileSizes[i] << (i + 1 == tileSizes.size() ? "" : ", ");
  llvm::errs() << "]\n";
  return tileSizes;
}

llvm::SmallVector<int64_t> NPUCostModel::getLayerNormTileSizes(
    npucore::LayerNormOp op) {
  auto tileSizes = getElementwiseTileSizes(op, this->hw.spmSizeBytes);
  llvm::errs() << "[CostModel] LayerNorm(npucore): Tile=[";
  for (size_t i = 0; i < tileSizes.size(); ++i)
    llvm::errs() << tileSizes[i] << (i + 1 == tileSizes.size() ? "" : ", ");
  llvm::errs() << "]\n";
  return tileSizes;
}

} // namespace npux
