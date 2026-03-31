//=======================================================
//src/Conversion/NpuTiling/NpuTilingHelper.hpp
//this file contains helper functions for npu tiling patterns
//=======================================================


#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "src/Compiler/NpuConfig.hpp"

mlir::LogicalResult peelForLoopLastIteration(
    mlir::RewriterBase &b, mlir::scf::ForOp forOp, mlir::scf::ForOp &lastIteration);

namespace npux{
void populateNpuTilingPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context);
void populateElemWiseTilingPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context);
void populateConvTilingPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context);
void populateConvInnerTilingPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context
);
void populateElemWiseInnerTilingPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context
);
void populateGemmTilingPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context 
);
void populateLayoutTilingPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context
);

void populateMaxPoolTilingPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context
);

}