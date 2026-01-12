#pragma once
#include "mlir/IR/PatternMatch.h"

namespace npux {
void populateElemWiseTilingPatterns(mlir::RewritePatternSet &patterns,
                                    mlir::MLIRContext *context) ;
}// namespace npux