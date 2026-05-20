//======================================================
// src/Conversion/NpuTiling/CostModel/Softmax.cpp
// This file implements the current cost model for Softmax operations.
// For now softmax is only modeled as full-shape execution.
//======================================================

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace npux {

llvm::SmallVector<int64_t> NPUCostModel::getSoftmaxTileSizes(
    mlir::linalg::GenericOp op) {
  auto loopRanges = op.getStaticLoopRanges();
  llvm::SmallVector<int64_t> tileSizes(
      loopRanges.begin(), loopRanges.end());

  llvm::errs() << "[CostModel] Softmax: Full-shape Tile=[";
  for (size_t i = 0; i < tileSizes.size(); ++i)
    llvm::errs() << tileSizes[i] << (i + 1 == tileSizes.size() ? "" : ", ");
  llvm::errs() << "]\n";

  return tileSizes;
}

} // namespace npux
