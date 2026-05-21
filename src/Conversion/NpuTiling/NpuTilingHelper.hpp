//=======================================================
//src/Conversion/NpuTiling/NpuTilingHelper.hpp
//this file contains helper functions for npu tiling patterns
//=======================================================

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "src/Compiler/NpuConfig.hpp"
#include <optional>

mlir::LogicalResult peelForLoopLastIteration(
    mlir::RewriterBase &b, mlir::scf::ForOp forOp, mlir::scf::ForOp &lastIteration);

namespace npux{

struct DmaTileAnalysis {
  llvm::SmallVector<int64_t> splitSizes;
  int colDimIdx = -1;
  int64_t callCount = 0;
};

std::optional<DmaTileAnalysis> analyzeDmaTileFromKnownTile(
    llvm::ArrayRef<int64_t> fullShape, llvm::ArrayRef<int64_t> tileShape);

std::optional<DmaTileAnalysis> buildFullTensorDmaAnalysis(mlir::Value value);

mlir::FailureOr<mlir::linalg::GenericOp> cloneGenericOpToMemorySpace(
    mlir::linalg::GenericOp op, int64_t memorySpace,
    mlir::PatternRewriter &rewriter);

mlir::FailureOr<mlir::Operation *> cloneLinalgOpToMemorySpace(
    mlir::Operation *op, int64_t memorySpace,
    mlir::PatternRewriter &rewriter);

mlir::linalg::CopyOp createDmaGenericOp(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value input,
    llvm::StringRef dmaName, int64_t encoding, mlir::Value dest = nullptr);

mlir::FailureOr<mlir::Value> maybeSplitDmaOp(
    mlir::linalg::CopyOp dmaOp, const DmaTileAnalysis &dmaAnalysis,
    mlir::PatternRewriter &rewriter);

// Tiles one standalone op with the provided tile sizes and peels the tail loops.
mlir::LogicalResult tileStandaloneOp(
    mlir::Operation *op, llvm::ArrayRef<int64_t> tileSizes,
    mlir::PatternRewriter &rewriter);

// Tiles the root consumer and fuses producers until the seed, then peels tails.
mlir::LogicalResult tileFusedChainOp(
    mlir::Operation *seed, mlir::Operation *root,
    llvm::ArrayRef<int64_t> tileSizes, mlir::PatternRewriter &rewriter,
    llvm::ArrayRef<DmaTileAnalysis> seedInputDmaAnalyses = {},
    const std::optional<DmaTileAnalysis> &rootOutputDmaAnalysis = std::nullopt,
    llvm::function_ref<void(llvm::ArrayRef<mlir::Operation *>,
        mlir::PatternRewriter &)> postprocessTiledOps = {});

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
