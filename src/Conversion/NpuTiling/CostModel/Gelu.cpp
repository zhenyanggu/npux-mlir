//======================================================
// src/Conversion/NpuTiling/CostModel/Gelu.cpp
// This file implements the cost model for Gelu/Elementwise operations,
// calculating tile sizes based on SRAM capacity constraints.
//======================================================

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <algorithm>

using namespace mlir;

namespace npux {

llvm::SmallVector<int64_t> NPUCostModel::getGeluTileSizes(mlir::linalg::GenericOp op) {
  auto loopRanges = op.getStaticLoopRanges();
  int64_t rank = loopRanges.size();
  
  llvm::SmallVector<int64_t> tileSizes(rank, 1);
  if (rank == 0) return tileSizes; 

  auto outputType = cast<RankedTensorType>(op.getOutputs()[0].getType());
  int64_t bitWidth = outputType.getElementType().getIntOrFloatBitWidth();
  int64_t bytesPerElem = std::max<int64_t>(1, bitWidth / 8);

  int64_t numOperands = op.getNumDpsInputs() + op.getNumDpsInits();
  
  // 使用 NPUCostModel 内部的硬件配置
  int64_t maxElems = this->hw.spmSizeBytes / (numOperands * bytesPerElem);

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

  // 打印日志
  llvm::errs() << "[CostModel] Gelu: SPM=" << this->hw.spmSizeBytes 
               << " Problem=[";
  for (size_t i = 0; i < rank; ++i) {
    llvm::errs() << loopRanges[i] << (i == rank - 1 ? "" : ", ");
  }
  llvm::errs() << "] -> Tile=[";
  for (size_t i = 0; i < rank; ++i) {
    llvm::errs() << tileSizes[i] << (i == rank - 1 ? "" : ", ");
  }
  llvm::errs() << "]\n";

  return tileSizes;
}

} // namespace npux