/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- Passes.hpp - ONNX-MLIR Passes Definition ------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file exposes the entry points to create compiler passes for ONNX-MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PASSES_H
#define ONNX_MLIR_PASSES_H

#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class MLIRContext;
class Pass;
} // namespace mlir

namespace onnx_mlir {

/// Pass for removing DisposableElementsAttr attributes.
std::unique_ptr<mlir::Pass> createScrubDisposablePass(bool closeAfter = true);

/// Pass for ONNX graph level optimization
std::unique_ptr<mlir::Pass> createONNXOpTransformPass();
std::unique_ptr<mlir::Pass> createONNXOpTransformPass(int threshold,
    bool report, bool targetCPU, bool enableSimdDataLayoutOpt,
    bool enableConvOptPass, bool enableRecomposeOptPass);

/// Pass for rewriting inside frontend dialect.
std::unique_ptr<mlir::Pass> createDecomposeONNXToONNXPass(
    const std::string &target = "");
std::unique_ptr<mlir::Pass> createRecomposeONNXToONNXPass(
    const std::string &target = "");

std::unique_ptr<mlir::Pass> createConvOptONNXToONNXPass(
    bool enableSimdDataLayoutOpt = false);

std::unique_ptr<mlir::Pass> createReplaceOpWithItsOperandPass(
    std::vector<std::string> nodeNameRegexList);

std::unique_ptr<mlir::Pass> createShapeInferencePass();

// To configure ConstPropONNXToONNXPass at program start.
void configureConstPropONNXToONNXPass(bool roundFPToInt, int expansionBound,
    llvm::ArrayRef<std::string> disabledPatterns, bool constantPropIsDisabled);

std::unique_ptr<mlir::Pass> createConstPropONNXToONNXPass();

/// Pass for instrument the ops in specific stage.
std::unique_ptr<mlir::Pass> createInstrumentPass();
std::unique_ptr<mlir::Pass> createInstrumentPass(
    const std::string &ops, unsigned actions);
/// Pass for instrument cleanup.
std::unique_ptr<mlir::Pass> createInstrumentCleanupPass();

/// Passes for instrumenting the ONNX ops to print their operand type
/// signatures at runtime.
std::unique_ptr<mlir::Pass> createInstrumentONNXSignaturePass(
    const std::string opPattern, const std::string nodePattern);

/// Pass for simplifying shape-related ONNX operations.
std::unique_ptr<mlir::Pass> createSimplifyShapeRelatedOpsPass();

/// Pass for replacing ONNXReturnOp with func::ReturnOp.
std::unique_ptr<mlir::Pass> createStandardFuncReturnPass();

/// Pass that combines multiple ONNX dialect transformations,
/// including shape inference.
std::unique_ptr<mlir::Pass> createONNXHybridTransformPass(
    bool enableRecomposition);

/// Pass for analyzing unknown dimension in ONNX operations.
std::unique_ptr<mlir::Pass> createONNXDimAnalysisPass();

/// Pass for setting onnx_node_name attribute if absent.
std::unique_ptr<mlir::Pass> createSetONNXNodeNamePass();

/// Pass for verifying Onnx ops before lowering to Krnl
std::unique_ptr<mlir::Pass> createONNXPreKrnlVerifyPass();

/// Add pass for lowering to Krnl IR.
std::unique_ptr<mlir::Pass> createLowerToKrnlPass();
std::unique_ptr<mlir::Pass> createLowerToKrnlPass(bool enableTiling,
    bool enableSIMD, bool enableParallel, bool enableFastMath,
    std::string opsForCall);
void configureOnnxToKrnlLoweringPass(bool reportOnParallel,
    bool parallelIsEnabled, std::string specificParallelOps, bool reportOnSimd,
    bool simdIsEnabled);
std::unique_ptr<mlir::Pass> createProcessScfParallelPrivatePass();
std::unique_ptr<mlir::Pass> createProcessKrnlParallelClausePass();

#ifdef ONNX_MLIR_ENABLE_STABLEHLO
/// Add pass for lowering to Stablehlo IR.
std::unique_ptr<mlir::Pass> createLowerToStablehloPass();
std::unique_ptr<mlir::Pass> createLowerToStablehloPass(bool enableUnroll);
#endif

/// Pass for eliding the values of global Krnl operations.
std::unique_ptr<mlir::Pass> createElideConstGlobalValuePass();

namespace krnl {
/// Pass for lowering frontend dialects to Krnl IR dialect.
std::unique_ptr<mlir::Pass> createConvertKrnlToAffinePass();
std::unique_ptr<mlir::Pass> createConvertKrnlToAffinePass(bool parallelEnabled);

/// Pass for lowering Seq in Krnl dialect.
std::unique_ptr<mlir::Pass> createConvertSeqToMemrefPass();

/// Pass for lowering krnl.region operation.
std::unique_ptr<mlir::Pass> createLowerKrnlRegionPass();

/// Pass for lowering Krnl dialect to LLVM dialect.
std::unique_ptr<mlir::Pass> createConvertKrnlToLLVMPass();
std::unique_ptr<mlir::Pass> createConvertKrnlToLLVMPass(bool verifyInputTensors,
    bool useLRODATA, bool storeConstantsToFile,
    float constantsToFileSingleThreshold, float constantsToFileTotalThreshold,
    std::string outputNameNoExt, bool enableParallel);

} // namespace krnl

/// Pass for lowering Onnx ops to TOSA dialect
std::unique_ptr<mlir::Pass> createConvertONNXToTOSAPass();

} // namespace onnx_mlir


namespace npux {
std::unique_ptr<mlir::Pass> createONNXToLinalgNpuPass();

std::unique_ptr<mlir::Pass> createNpuOutlinePass();

std::unique_ptr<mlir::Pass> createNpuSramPromotionPass();


std::unique_ptr<mlir::Pass> createNpuInlinePass();

std::unique_ptr<mlir::Pass> createNpuDPSConversionPass();

std::unique_ptr<mlir::Pass> createNpuMergePass();

std::unique_ptr<mlir::Pass> createNpuTilingPass();

std::unique_ptr<mlir::Pass> createNpuInnerTilingPass();

std::unique_ptr<mlir::Pass> createConvertLinalgToNpuPass();

std::unique_ptr<mlir::Pass> createNpuMemPlanPass();

std::unique_ptr<mlir::Pass> createNpuCleanPackPass();

std::unique_ptr<mlir::Pass> createNpuLowerPackPass();

std::unique_ptr<mlir::Pass> createNpuRegionExtensionPass();

std::unique_ptr<mlir::Pass> createNpuFusePass();

std::unique_ptr<mlir::Pass> createWeightHoistingPass();

std::unique_ptr<mlir::Pass> createNpuICLoopSplitPass();

std::unique_ptr<mlir::Pass> createSplitConvIcPass();

std::unique_ptr<mlir::Pass> createLowerNpuSubviewPass();

std::unique_ptr<mlir::Pass> createNpuxComputeFusionPass();

std::unique_ptr<mlir::Pass> createGemmPipelinePass();

std::unique_ptr<mlir::Pass> createCustomBufferLoopHoistingPass();

std::unique_ptr<mlir::Pass> createNpuxSfu5DShapePatchPass();

std::unique_ptr<mlir::Pass> createNpuInsertDmaPass();

std::unique_ptr<mlir::Pass> createNpuOpSplittingPass();

std::unique_ptr<mlir::Pass> createNpuSpatialPeelingPass();

std::unique_ptr<mlir::Pass> createRestoreAllocSpacePass();

std::unique_ptr<mlir::Pass> createNpuRemoveRedundantDmaPass();

std::unique_ptr<mlir::Pass> createRemoveDuplicateMvinBiasPass();

std::unique_ptr<mlir::Pass> createEraseNpuMemorySpacePass();

std::unique_ptr<mlir::Pass> createNpuFusionPlanningPass();

std::unique_ptr<mlir::Pass> createNpuFusionMaterializePass();

void registerBufferDeallocTest();
}// namespace npux

#endif
