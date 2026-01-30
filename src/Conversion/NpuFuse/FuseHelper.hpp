//========================================
// src/Conversion/NpuFuse/FuseHelper.hpp
// this file contains helper functions for
// NPU operation fusion patterns
//========================================

#pragma once

#include "mlir/IR/PatternMatch.h"

namespace npux {
void populateNpuFusePatterns(mlir::RewritePatternSet &patterns);
void populateReluAccFusionPatterns(mlir::RewritePatternSet &patterns);
} // namespace npux