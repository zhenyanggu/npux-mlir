//======================================================
// src/Conversion/NpuTiling/CostModel/MatAdd.cpp
// This file implements the cost model for MatAdd operations,
// calculating tile sizes based on hardware 2D limits.
//======================================================

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace npux {

llvm::SmallVector<int64_t> NPUCostModel::getMatAddTileSizes(mlir::linalg::GenericOp op) {
  auto loopRanges = op.getStaticLoopRanges();
  int64_t rank = loopRanges.size();
  
  // 默认全部切分为 1 (最保守情况)
  llvm::SmallVector<int64_t> tileSizes(rank, 1);
  if (rank == 0) return tileSizes; // 处理 Scalar

  int64_t hwMaxColDim = 256;
  int64_t hwMaxRowDim = 128;

  int state = 0;
  int64_t current_acc = 1;

  for (int i = rank - 1; i >= 0; --i) {
    int64_t dim = loopRanges[i] > 0 ? loopRanges[i] : 1;

    if (state == 2) {
      tileSizes[i] = 1;
      continue;
    }

    if (state == 0) { // 状态 0：累乘 Col
      if (current_acc * dim <= hwMaxColDim) {
        tileSizes[i] = dim;
        current_acc *= dim;
      } else {
        tileSizes[i] = hwMaxColDim / current_acc;
        state = 1;
        current_acc = 1;
      }
    } else if (state == 1) { // 状态 1：累乘 Row
      if (current_acc * dim <= hwMaxRowDim) {
        tileSizes[i] = dim;
        current_acc *= dim;
      } else {
        tileSizes[i] = hwMaxRowDim / current_acc;
        state = 2;
      }
    }
  }

  // 打印日志
  llvm::errs() << "[CostModel] MatAdd: Problem=[";
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