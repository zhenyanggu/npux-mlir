//=======================================================
//src/Conversion/NpuTiling/NpuTilingHelper.hpp
//this file contains helper functions for npu tiling patterns
//=======================================================


#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"


namespace npux{
void applyTileConfigNCHWc32(llvm::SmallVectorImpl<int64_t> &sizes, const std::vector<int64_t> &configSizes);
llvm::SmallVector<int64_t> getNpuTileSizes(mlir::linalg::GenericOp op);
llvm::SmallVector<int64_t> getGemmTileSizes(mlir::linalg::GenericOp op);
bool isTilingNecessary(llvm::ArrayRef<int64_t> tileSizes, llvm::ArrayRef<int64_t> loopRanges);
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

}