//=============================================================
// src/Conversion/NpuTiling/Conv.cpp
// this file is for Conv op tiling pattern
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h" // 核心 Tiling 工具
#include "mlir/IR/PatternMatch.h"
#include "src/Pass/Passes.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"




using namespace mlir;
using namespace npux;


namespace {

struct NpuConvTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    
    // 1. 检查是否是 NPU 卷积
    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (!libCall || libCall.getValue() != "npu_conv") {
      return failure();
    }

    // 2. 防止重复 Tiling
    if (op->hasAttr("npu.tiled")) {
      return failure();
    }


    SmallVector<int64_t> tileSizes = getNpuTileSizes(op);


    auto loopRanges = op.getStaticLoopRanges();

    // 2. 检查是否真的需要切分
    if (!isTilingNecessary(tileSizes, loopRanges)) {
        // 情况 A：不需要切分 (例如 DSE 这里的 size 刚好等于或大于 feature map size)
        
        op->setAttr("npu.tiled", rewriter.getUnitAttr());
        
        // 打上特殊标记
        op->setAttr("npu.trivial_tiling", rewriter.getUnitAttr());
      
        
        op->setAttr("npu.accumulate_mode", rewriter.getUnitAttr());

        return success();
    }

    // 5. 执行 Tiling
    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizesOpFold =
        getAsOpFoldResult(rewriter.getI64ArrayAttr(tileSizes));

    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizesOpFold);

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult)) {
      return failure();
    }

    // 6. 标记新生成的 Op
    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
      // 传递 Layer Name 以便 Debug
      if (op->hasAttr("npu.layer_name")) {
         tiledOp->setAttr("npu.layer_name", op->getAttr("npu.layer_name"));
      }
      // 传递 Accumulate Mode 标记 (给 Lowering 用)
      // 只有 Reduction Loop (IC) 内的 Op 需要累加
      // 这里简单起见全部打上，具体由 Lowering 阶段根据 Buffer 初始化状态决定
      tiledOp->setAttr("npu.accumulate_mode", rewriter.getUnitAttr()); 
    }

    // 7. 处理 Peeling (解决不能整除的问题)
    auto loops = tilingResult->loops;
    SmallVector<Value> finalResults = tilingResult->replacements;

    // 倒序遍历处理 Peeling (从内向外：IC -> OW -> OH -> OC)
    for (int i = loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp) continue;

      scf::ForOp partialIteration;
      LogicalResult status = scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
        // 标记 Tail Op，防止被后续 Pass 误伤
        // 注意：Peeling 会复制 loop body 里的 Op，所以新 Op 也会带有 npu.tiled
        partialIteration->setAttr("npu.peeled_tail", rewriter.getUnitAttr());

        // 如果 Peeling 发生，scf.for 的结果可能会变，需要更新 replacements
        // 特别是对于 Parallel 维度，Peeling 可能会影响 Tensor 的 InsertSlice 链
        if (i == 0) {
           finalResults = partialIteration->getResults();
        }
      }
    }

    // 8. 替换原 Op
    rewriter.replaceOp(op, finalResults);

    return success();
  }
};

} // namespace




void npux::populateConvTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuConvTilingPattern>(context);
};