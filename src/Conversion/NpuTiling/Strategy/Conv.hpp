#pragma once

#include "src/Conversion/NpuTiling/Strategy/Dispatcher.hpp"

namespace npux {

llvm::SmallVector<int64_t> inferConvActivationTileShape(
    mlir::linalg::GenericOp convOp, llvm::ArrayRef<int64_t> seedTileSizes);

llvm::SmallVector<int64_t> inferConvWeightTileShape(
    mlir::linalg::GenericOp convOp, llvm::ArrayRef<int64_t> seedTileSizes);

mlir::LogicalResult tileConvWithRoot(
    const FusionCursor &cursor,
    mlir::PatternRewriter &rewriter);

} // namespace npux
