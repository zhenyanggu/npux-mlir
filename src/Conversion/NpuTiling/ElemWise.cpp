//=============================================================
// src/Conversion/NpuTiling/ElemWise.cpp
// this file is for elemwise op tiling pattern
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h" // 核心 Tiling 工具
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "npu-tiling"
using namespace mlir;
using namespace npux;

namespace {
// ============================================================================
// Helper: calculate tile sizes for npucore.gelu directly.
// ============================================================================
static SmallVector<int64_t> getGeluTileSizes(npucore::GeluOp op) {
  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  return costModel.getOptimalTileSizes(op);
}

// ============================================================================
// Helper: calculate tile sizes for npucore.softmax directly.
// ============================================================================
static SmallVector<int64_t> getSoftmaxTileSizes(npucore::SoftmaxOp op) {
  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  return costModel.getOptimalTileSizes(op);
}

// ============================================================================
// Helper: calculate tile sizes for npucore.layernorm directly.
// ============================================================================
static SmallVector<int64_t> getLayerNormTileSizes(npucore::LayerNormOp op) {
  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  return costModel.getOptimalTileSizes(op);
}

// ============================================================================
// Helper: calculate tile sizes for npucore.matadd directly.
// ============================================================================
static SmallVector<int64_t> getMatAddTileSizes(npucore::MatAddOp op) {
  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  return costModel.getOptimalTileSizes(op);
}

// ============================================================================
// Tiling Pattern: native npucore.gelu.
// ============================================================================
struct NpuElemWiseTilingPattern : public OpRewritePattern<npucore::GeluOp> {
  using OpRewritePattern<npucore::GeluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::GeluOp op, PatternRewriter &rewriter) const override {

    if (op->hasAttr("npu.tiled"))
      return failure();

    SmallVector<int64_t> rawTileSizes = getGeluTileSizes(op);

    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizes =
        getAsOpFoldResult(rewriter.getI64ArrayAttr(rawTileSizes));

    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult))
      return failure();

    for (auto loop : tilingResult->loops) {
      loop->setAttr("npu.target", rewriter.getStringAttr("npu"));
    }

    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
    }

    // 【修复 2】: 稳健的 Peeling 逻辑
    // 我们必须确保从内向外 Peel，并且正确处理 Loop 结构的更新
    auto loops = tilingResult->loops;
    SmallVector<Value> finalResults = tilingResult->replacements;

    // 倒序遍历处理 Peeling (从内向外)
    for (int i = loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp)
        continue;

      scf::ForOp partialIteration;
      LogicalResult status =
          scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
        if (i == 0) {
          finalResults = partialIteration->getResults();
        }
      }
    }

    // 使用更新后的结果进行替换
    rewriter.replaceOp(op, finalResults);
    return success();
  }
};

// ============================================================================
// Tiling Pattern: native npucore.softmax.
// ============================================================================
struct NpuSoftmaxTilingPattern : public OpRewritePattern<npucore::SoftmaxOp> {
  using OpRewritePattern<npucore::SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::SoftmaxOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.tiled"))
      return failure();

    SmallVector<int64_t> rawTileSizes = getSoftmaxTileSizes(op);

    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizes =
        getAsOpFoldResult(rewriter.getI64ArrayAttr(rawTileSizes));

    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult))
      return failure();

    for (auto loop : tilingResult->loops) {
      loop->setAttr("npu.target", rewriter.getStringAttr("npu"));
    }

    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
    }

    auto loops = tilingResult->loops;
    SmallVector<Value> finalResults = tilingResult->replacements;

    for (int i = loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp)
        continue;

      scf::ForOp partialIteration;
      LogicalResult status =
          scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
        if (i == 0) {
          finalResults = partialIteration->getResults();
        }
      }
    }

    rewriter.replaceOp(op, finalResults);
    return success();
  }
};

// ============================================================================
// Tiling Pattern: native npucore.layernorm.
// ============================================================================
struct NpuLayerNormTilingPattern
    : public OpRewritePattern<npucore::LayerNormOp> {
  using OpRewritePattern<npucore::LayerNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::LayerNormOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.tiled"))
      return failure();

    SmallVector<int64_t> rawTileSizes = getLayerNormTileSizes(op);

    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizes =
        getAsOpFoldResult(rewriter.getI64ArrayAttr(rawTileSizes));

    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult))
      return failure();

    for (auto loop : tilingResult->loops) {
      loop->setAttr("npu.target", rewriter.getStringAttr("npu"));
    }

    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
    }

    auto loops = tilingResult->loops;
    SmallVector<Value> finalResults = tilingResult->replacements;

    for (int i = loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp)
        continue;

      scf::ForOp partialIteration;
      LogicalResult status =
          scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
        if (i == 0) {
          finalResults = partialIteration->getResults();
        }
      }
    }

    rewriter.replaceOp(op, finalResults);
    return success();
  }
};

// ============================================================================
// Tiling Pattern: native npucore.matadd.
// ============================================================================
struct NpuMatAddTilingPattern : public OpRewritePattern<npucore::MatAddOp> {
  using OpRewritePattern<npucore::MatAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::MatAddOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.tiled"))
      return failure();

    SmallVector<int64_t> rawTileSizes = getMatAddTileSizes(op);

    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizes =
        getAsOpFoldResult(rewriter.getI64ArrayAttr(rawTileSizes));

    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult))
      return failure();

    for (auto loop : tilingResult->loops) {
      loop->setAttr("npu.target", rewriter.getStringAttr("npu"));
    }

    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
    }

    auto loops = tilingResult->loops;
    SmallVector<Value> finalResults = tilingResult->replacements;

    for (int i = loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp)
        continue;

      scf::ForOp partialIteration;
      LogicalResult status =
          scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
        if (i == 0) {
          finalResults = partialIteration->getResults();
        }
      }
    }

    rewriter.replaceOp(op, finalResults);
    return success();
  }
};

} // namespace

void npux::populateElemWiseTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuElemWiseTilingPattern>(context);
  patterns.add<NpuSoftmaxTilingPattern>(context);
  patterns.add<NpuLayerNormTilingPattern>(context);
  patterns.add<NpuMatAddTilingPattern>(context);
}
