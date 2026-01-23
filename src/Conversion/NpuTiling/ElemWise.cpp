//=============================================================
// src/Conversion/NpuTiling/ElemWise.cpp
// this file is for elemwise op tiling pattern
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h" // 核心 Tiling 工具
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Pass/Passes.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

#define DEBUG_TYPE "npu-tiling"
using namespace mlir;
using namespace npux; 

namespace {


// === 2. Tiling Pattern ===
struct NpuElemWiseTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    
    if (op->hasAttr("npu.tiled")) return failure();

    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (libCall && libCall.getValue() != "npu_gelu") {
        return failure(); // 把机会留给 NpuConvTilingPattern
    }

    SmallVector<int64_t> rawTileSizes = getNpuTileSizes(op);
    auto loopRanges = op.getStaticLoopRanges();


    if (!isTilingNecessary(rawTileSizes, loopRanges)) {
        op->setAttr("npu.tiled", rewriter.getUnitAttr());
        
        op->setAttr("npu.trivial_tiling", rewriter.getUnitAttr());

        return success();
    }

    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizes = getAsOpFoldResult(rewriter.getI64ArrayAttr(rawTileSizes));
    
    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult)) return failure();

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
      if (!loopOp) continue;

      scf::ForOp partialIteration;
      LogicalResult status = scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
        // 1. 标记 Tail (可选)
        partialIteration->setAttr("npu.peeled_tail", rewriter.getUnitAttr());
        
        // 2. 【关键修复】更新替换值
        // 如果当前处理的是最外层循环 (index 0)，或者该循环的结果直接对应 Op 的结果
        // 我们必须把 finalResults 更新为 Tail Loop 的结果
        // 因为 Tail Loop 串在 Main Loop 后面，它才持有最终完整的数据
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


} // namespace

void npux::populateElemWiseTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuElemWiseTilingPattern>(context);
}

