#pragma once

#include "src/Conversion/NpuTiling/Strategy/Dispatcher.hpp"

namespace npux {

mlir::LogicalResult tileSoftmaxWithRoot(
    const FusionCursor &cursor,
    mlir::PatternRewriter &rewriter);

mlir::LogicalResult tileSoftmaxOp(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter &rewriter);

} // namespace npux
