#pragma once

#include "src/Conversion/NpuTiling/Strategy/Dispatcher.hpp"

namespace npux {

mlir::LogicalResult tileLayerNormWithRoot(
    const FusionCursor &cursor,
    mlir::PatternRewriter &rewriter);

mlir::LogicalResult tileLayerNormOp(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter &rewriter);

} // namespace npux
