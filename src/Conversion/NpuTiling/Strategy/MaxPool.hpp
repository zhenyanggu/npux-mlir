#pragma once

#include "src/Conversion/NpuTiling/Strategy/Dispatcher.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

namespace npux {

mlir::LogicalResult tileMaxPoolWithRoot(
    const FusionCursor &cursor, mlir::PatternRewriter &rewriter);

mlir::LogicalResult tileMaxPoolOp(
    mlir::linalg::GenericOp op, mlir::PatternRewriter &rewriter);

} // namespace npux
