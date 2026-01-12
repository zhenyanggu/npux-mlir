//====================================================
//src/Conversion/NpuToLLVM/NpuxConversionHelper.hpp
//this file declares helper functions for npux conversion 
//====================================================

#pragma once


#include "mlir/IR/PatternMatch.h"

namespace npux {



    void populateLinalgToNpuxPatterns(mlir::RewritePatternSet &patterns);

    void populateLinalgSfuToNpuxPattern(mlir::RewritePatternSet &patterns);

    void populateSramDataMovementPatterns(mlir::RewritePatternSet &patterns);

    void populateHostAllocToNpuxPatterns(mlir::RewritePatternSet &patterns);

    void populateNpuLifecyclePatterns(mlir::RewritePatternSet &patterns);
}
