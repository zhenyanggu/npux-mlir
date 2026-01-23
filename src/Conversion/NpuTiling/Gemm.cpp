//==============================================
//src/Conversion/NpuTiling/Gemm.cpp
//this file implemnent the tiling strategy of 
//gemm op(including gemm&matmul)
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

// 使用 Pattern 而不是直接在 Pass 里写逻辑
struct NpuGemmTilingPattern : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::MatmulOp op, PatternRewriter &rewriter) const override {
    
    // 1. 防止重复 Tiling
    if (op->hasAttr("npu.tiled")) return failure();

    // 2. 获取分块大小 (调用 Helper，Helper 会去查 Config)
    SmallVector<int64_t> tileSizes = getGemmTileSizes(op);
    if (tileSizes.empty()) return failure();

    // 3. 检查是否必要 (如果 Tile Size > Problem Size，就没必要切)
    auto loopRanges = op.getStaticLoopRanges();
    if (!isTilingNecessary(tileSizes, loopRanges)) {
        op->setAttr("npu.tiled", rewriter.getUnitAttr());
        op->setAttr("npu.trivial_tiling", rewriter.getUnitAttr());
        return success();
    }

    // 4. 执行 Tiling (统一使用 TilingInterface + SCF)
    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizesOpFold = getAsOpFoldResult(rewriter.getI64ArrayAttr(tileSizes));
    
    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizesOpFold);

    // Gemm 的循环顺序是 M, N, K。
    // 如果你想做 Loop Interchange (例如调整为 N, M, K)，可以在这里设置 options.setInterchange(...)
    // 但通常 M, N, K 顺序配合 Output Stationary (K 在最内层) 是符合 NPU 逻辑的。

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult)) return failure();

    // 标记新生成的 Op 为已处理
    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
    }

    // 5. 处理 Peeling (完全复用 ElemWise 中的逻辑，确保尾块正确)
    auto loops = tilingResult->loops;
    SmallVector<Value> finalResults = tilingResult->replacements;

    // 倒序遍历处理 Peeling (从 K -> N -> M)
    for (int i = loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp) continue;

      scf::ForOp partialIteration;
      LogicalResult status = scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
        partialIteration->setAttr("npu.peeled_tail", rewriter.getUnitAttr());
        
        // 关键：如果剥离的是最外层循环，需要更新替换结果
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

// 注册函数
void npux::populateGemmTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuGemmTilingPattern>(context);
}