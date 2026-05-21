//=============================================================================
// Shared planning utilities for the two-stage NPU fusion tiling pipeline.
//=============================================================================

#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "src/Conversion/NpuTiling/FusionAnalysis/FusionEvaluator.hpp"
#include "src/Conversion/NpuTiling/Strategy/Dispatcher.hpp"
#include "src/Dialect/Npux/NpuxOps.hpp"

namespace npux {

bool isCandidateSeedOp(mlir::Operation *op);

mlir::FailureOr<FusionCursor> buildFusionCursorFromSeed(
    mlir::Operation *seed, mlir::FusionCostEvaluator &evaluator);

void serializeFusionCursorToGroup(const FusionCursor &cursor,
    npux::FusionGroupOp group, mlir::Builder &builder);

mlir::FailureOr<FusionCursor> deserializeFusionCursorFromGroup(
    npux::FusionGroupOp group);

} // namespace npux
