//======================================================
// src/Conversion/NpuTiling/CostModel/LayerNorm.cpp
// This file implements the current cost model for LayerNorm operations.
// For now layernorm is only modeled as full-shape execution.
//======================================================

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace npux {

llvm::SmallVector<int64_t> NPUCostModel::getLayerNormTileSizes(
    mlir::linalg::GenericOp op) {
  auto loopRanges = op.getStaticLoopRanges();
  llvm::SmallVector<int64_t> tileSizes(
      loopRanges.begin(), loopRanges.end());

  llvm::errs() << "[CostModel] LayerNorm: Full-shape Tile=[";
  for (size_t i = 0; i < tileSizes.size(); ++i)
    llvm::errs() << tileSizes[i] << (i + 1 == tileSizes.size() ? "" : ", ");
  llvm::errs() << "]\n";

  return tileSizes;
}

} // namespace npux
