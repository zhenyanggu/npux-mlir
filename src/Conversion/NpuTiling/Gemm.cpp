//==============================================
// src/Conversion/NpuTiling/Gemm.cpp
// This file implements the tiling strategy of 
// gemm op (converted linalg.generic)
//==============================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

using namespace mlir;
using namespace npux;

namespace {

struct NpuGemmTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    
    // 1. 防止重复 Tiling
    if (op->hasAttr("npu.tiled")) return failure();

    // 2. 检查是否为 Gemm/MatMul
    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (!libCall) return failure();
    StringRef opName = libCall.getValue();
    if (opName != "npu_gemm" && opName != "npu_matmul") {
        return failure();
    }

    // 3. 获取分块大小
    SmallVector<int64_t> tileSizes = getGemmTileSizes(op);
    if (tileSizes.empty()) return failure();

    // 4. 检查是否必要
    auto loopRanges = op.getStaticLoopRanges();
    if (!isTilingNecessary(tileSizes, loopRanges)) {
        op->setAttr("npu.tiled", rewriter.getUnitAttr());
        op->setAttr("npu.trivial_tiling", rewriter.getUnitAttr());
        return success();
    }

    // 5. 执行 Tiling
    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizesOpFold = getAsOpFoldResult(rewriter.getI64ArrayAttr(tileSizes));
    
    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizesOpFold);

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult)) return failure();

    // 标记内部计算 Op
    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
      tiledOp->setAttr("library_call", libCall); 
      if (auto target = op->getAttr("npu.target")) {
         tiledOp->setAttr("npu.target", target);
      }
    }

    // =================================================================
    // Step 6.5: Loop Labeling (打标)
    // 逻辑：Gemm 的 Loop Order 总是 [Batch..., M, N, K]
    // =================================================================
    auto generatedLoops = tilingResult->loops;
    int currentLoopIdx = 0;
    int64_t rank = tileSizes.size(); // e.g. 3 for [M,N,K], 4 for [B,M,N,K]

    for (size_t dimIdx = 0; dimIdx < tileSizes.size(); ++dimIdx) {
        // 如果该维度没有切分 (size == 0)，说明没有生成 loop，跳过
        if (tileSizes[dimIdx] == 0) continue;

        if (currentLoopIdx >= generatedLoops.size()) break;

        auto loopOp = generatedLoops[currentLoopIdx].getOperation();
        
        // 动态生成 Label
        StringRef label = "Batch"; // 默认是 Batch
        
        // 倒数第1个是 K (Reduction)
        if (dimIdx == rank - 1) label = "K";
        // 倒数第2个是 N
        else if (dimIdx == rank - 2) label = "N";
        // 倒数第3个是 M
        else if (dimIdx == rank - 3) label = "M";

        // 设置属性
        // 1. npu.loop_dim: 告诉 LoopSplitPass 哪个是 K 维
        loopOp->setAttr("npu.loop_dim", rewriter.getStringAttr(label));
        
        // 2. npu.computeop: 统一标记为 "gemm"，这样 GemmPipelinePass 可以识别
        loopOp->setAttr("npu.computeop", rewriter.getStringAttr("gemm"));

        currentLoopIdx++;
    }
    // =================================================================

    // 6. 处理 Peeling (倒序处理)
    auto loops = tilingResult->loops;
    SmallVector<Value> finalResults = tilingResult->replacements;

    for (int i = loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp) continue;

      scf::ForOp partialIteration;
      LogicalResult status = scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
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

void npux::populateGemmTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuGemmTilingPattern>(context);
}