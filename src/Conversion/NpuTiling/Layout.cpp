//=============================================================
// src/Conversion/NpuTiling/Layout.cpp
// This file implements tiling patterns for npucore layout ops.
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include <algorithm>
#include <cmath>

using namespace mlir;
using namespace npux;

namespace {

// ============================================================================
// Helper: preserve loop target labels on peeled tails.
// ============================================================================
static void inheritNpuAttributes(scf::ForOp source, scf::ForOp target) {
  if (!source || !target)
    return;
  if (auto attr = source->getAttr("npu.target"))
    target->setAttr("npu.target", attr);
}

// ============================================================================
// Helper: tile the last two dimensions of transpose under SRAM constraints.
// ============================================================================
static SmallVector<int64_t> getTransposeTileSizes(npucore::TransposeOp op) {
  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  return costModel.getOptimalTileSizes(op);
}

// ============================================================================
// Helper: tile NCHW -> NCHWc32 using the old H/W/Cblk cost rule.
// ============================================================================
static SmallVector<int64_t> getLayoutPackTileSizes(
    npucore::LayoutNchwToNchwc32Op op) {
  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  return costModel.getOptimalTileSizes(op);
}

// ============================================================================
// Helper: tile NCHWc32 -> NCHW by reusing the old packed H/W/Cblk rule.
// ============================================================================
static SmallVector<int64_t> getLayoutUnpackTileSizes(
    npucore::LayoutNchwc32ToNchwOp op) {
  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  return costModel.getOptimalTileSizes(op);
}

// ============================================================================
// Helper: common layout tiling and tail peeling implementation.
// ============================================================================
template <typename OpTy>
static LogicalResult tileLayoutOp(
    OpTy op, PatternRewriter &rewriter, ArrayRef<int64_t> tileShape) {
  if (tileShape.empty())
    return failure();

  auto tilingInterfaceOp = cast<TilingInterface>(op.getOperation());
  scf::SCFTilingOptions options;
  options.setTileSizes(
      getAsOpFoldResult(rewriter.getI64ArrayAttr(tileShape)));

  FailureOr<scf::SCFTilingResult> tilingResult =
      scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);
  if (failed(tilingResult))
    return failure();

  for (LoopLikeOpInterface loop : tilingResult->loops)
    loop->setAttr("npu.target", rewriter.getStringAttr("npu"));

  for (Operation *tiledOp : tilingResult->tiledOps)
    tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());

  SmallVector<Value> finalResults = tilingResult->replacements;
  auto loops = tilingResult->loops;
  for (int i = (int)loops.size() - 1; i >= 0; --i) {
    auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
    if (!loopOp)
      continue;

    scf::ForOp partialIteration;
    if (succeeded(scf::peelForLoopAndSimplifyBounds(
            rewriter, loopOp, partialIteration))) {
      inheritNpuAttributes(loopOp, partialIteration);
      partialIteration->setAttr("npu.peeled_tail", rewriter.getUnitAttr());
      if (i == 0)
        finalResults = partialIteration->getResults();
    }
  }

  rewriter.replaceOp(op, finalResults);
  return success();
}

// ============================================================================
// Pattern: tile npucore.transpose.
// ============================================================================
struct NpuTransposeTilingPattern : public OpRewritePattern<npucore::TransposeOp> {
  using OpRewritePattern<npucore::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::TransposeOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.tiled"))
      return failure();
    SmallVector<int64_t> tileSizes = getTransposeTileSizes(op);
    return tileLayoutOp(op, rewriter, tileSizes);
  }
};

// ============================================================================
// Pattern: tile npucore.layout_nchw_to_nchwc32.
// ============================================================================
struct NpuLayoutNchwToNchwc32TilingPattern
    : public OpRewritePattern<npucore::LayoutNchwToNchwc32Op> {
  using OpRewritePattern<npucore::LayoutNchwToNchwc32Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(npucore::LayoutNchwToNchwc32Op op,
      PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.tiled"))
      return failure();
    SmallVector<int64_t> tileSizes = getLayoutPackTileSizes(op);
    return tileLayoutOp(op, rewriter, tileSizes);
  }
};

// ============================================================================
// Pattern: tile npucore.layout_nchwc32_to_nchw.
// ============================================================================
struct NpuLayoutNchwc32ToNchwTilingPattern
    : public OpRewritePattern<npucore::LayoutNchwc32ToNchwOp> {
  using OpRewritePattern<npucore::LayoutNchwc32ToNchwOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(npucore::LayoutNchwc32ToNchwOp op,
      PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.tiled"))
      return failure();
    SmallVector<int64_t> tileSizes = getLayoutUnpackTileSizes(op);
    return tileLayoutOp(op, rewriter, tileSizes);
  }
};

} // namespace

void npux::populateLayoutTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuTransposeTilingPattern>(context);
  patterns.add<NpuLayoutNchwToNchwc32TilingPattern>(context);
  patterns.add<NpuLayoutNchwc32ToNchwTilingPattern>(context);
}
