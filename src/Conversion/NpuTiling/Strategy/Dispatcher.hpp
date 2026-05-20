#pragma once

#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

namespace npux {

struct FusionCursor {
  mlir::linalg::GenericOp seed;
  mlir::linalg::GenericOp tail;
  llvm::SmallVector<mlir::linalg::GenericOp> chainOps;
  llvm::SmallVector<int64_t> seedTileSizes;
  llvm::SmallVector<int64_t> tailTileSizes;
  llvm::SmallVector<DmaTileAnalysis> seedInputDmaAnalyses;
  std::optional<DmaTileAnalysis> rootOutputDmaAnalysis;
};

mlir::LogicalResult tileSeedOp(
    const FusionCursor &cursor,
    mlir::PatternRewriter &rewriter);

} // namespace npux
