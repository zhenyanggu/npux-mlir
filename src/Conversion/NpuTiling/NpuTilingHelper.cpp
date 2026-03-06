//=======================================================
// src/Conversion/NpuTiling/NpuTilingHelper.cpp
// this file contains helper functions for npu tiling patterns
//=======================================================

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

#include <algorithm>
#include <cmath>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;

namespace npux {
void populateNpuTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  populateElemWiseTilingPatterns(patterns, context);
  populateConvTilingPatterns(patterns, context);
  populateGemmTilingPatterns(patterns, context);
  populateLayoutTilingPatterns(patterns, context);
  populateMaxPoolTilingPatterns(patterns, context);
};
} // namespace npux