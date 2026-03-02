//=============================================================
// src/Conversion/NpuTiling/Layout.cpp
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Pass/Passes.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp" // 引入 Helper

#define DEBUG_TYPE "npu-tiling"
using namespace mlir;
using namespace npux; 

namespace {

// === Layout Tiling Pattern ===
struct NpuLayoutTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    
    if (op->hasAttr("npu.tiled")) return failure();

    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (!libCall) return failure();

    StringRef opName = libCall.getValue();
    if (opName != "npu_layout_nchw_to_nchwc32" &&
        opName != "npu_layout_nchwc32_to_nchw" &&
        opName != "npu_transpose") {
        return failure();
    }

    // 1. 调用 NpuTilingHelper.cpp 中的接口获取硬件感知的 TileSize
    SmallVector<int64_t> rawTileSizes = getLayoutTileSizes(op, opName);
    auto loopRanges = op.getStaticLoopRanges();

    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizes = getAsOpFoldResult(rewriter.getI64ArrayAttr(rawTileSizes));
    
    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    // 2. Tiling
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult)) return failure();

    for (auto loop : tilingResult->loops) {
      loop->setAttr("npu.target", rewriter.getStringAttr("npu"));
    }

    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
    }

    // 3. Peeling 处理 Tail 边界情况
    auto loops = tilingResult->loops;
    SmallVector<Value> finalResults = tilingResult->replacements;

    for (int i = loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp) continue;

      scf::ForOp partialIteration;
      LogicalResult status = scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
        // 标记尾部，当下游 Pass 看到此属性时，可生成 NPU 片上 Padding/Memset 指令
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

void npux::populateLayoutTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuLayoutTilingPattern>(context);
}