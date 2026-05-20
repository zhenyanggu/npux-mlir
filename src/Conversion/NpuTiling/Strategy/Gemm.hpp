#pragma once

#include "src/Conversion/NpuTiling/Strategy/Dispatcher.hpp"

namespace npux {

mlir::LogicalResult tileGemmWithRoot(
    const FusionCursor &cursor,
    mlir::PatternRewriter &rewriter);

} // namespace npux
