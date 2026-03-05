//=============================================================
// 建议新建 src/Conversion/NpuTiling/MaxPool.cpp
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Pass/Passes.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

#define DEBUG_TYPE "npu-tiling"
using namespace mlir;
using namespace npux; 

namespace {

// === MaxPool Tiling Pattern ===
struct NpuMaxPoolTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    
    // 避免无限循环 Tiling
    if (op->hasAttr("npu.tiled")) return failure();

    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (!libCall || libCall.getValue() != "npu_maxpool") {
        return failure();
    }

    // 1. 调用 Helper 获取 TileSize
    SmallVector<int64_t> rawTileSizes = getMaxPoolTileSizes(op, libCall.getValue());
    if (rawTileSizes.empty()) return failure();

    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizes = getAsOpFoldResult(rewriter.getI64ArrayAttr(rawTileSizes));
    
    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    // 2. 执行 Tiling
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult)) return failure();

    // 赋予下游所需的 NPU Attributes
    for (auto loop : tilingResult->loops) {
      loop->setAttr("npu.target", rewriter.getStringAttr("npu"));
    }

    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
    }

    // 3. Peeling 处理 Tail 边界情况（处理特征图无法被完美整除的情况）
    auto loops = tilingResult->loops;
    SmallVector<Value> finalResults = tilingResult->replacements;

    for (int i = loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp) continue;

      scf::ForOp partialIteration;
      LogicalResult status = scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
        // 标记尾部，方便下游降级处理不规则的尺寸
        partialIteration->setAttr("npu.peeled_tail", rewriter.getUnitAttr());
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

void npux::populateMaxPoolTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuMaxPoolTilingPattern>(context);
}