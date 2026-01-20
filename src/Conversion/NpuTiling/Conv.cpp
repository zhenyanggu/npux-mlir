//=============================================================
// src/Conversion/NpuTiling/ElemWise.cpp
// this file is for elemwise op tiling pattern
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

    // 3. 读取 DSE 属性 [t_oh, t_ow, t_ic, t_oc]
    // 这个属性是你刚刚在 ConvToLinalg 里 setAttr 的
    auto dseAttr = op->getAttrOfType<ArrayAttr>("npu.dse_tiling");
    if (!dseAttr || dseAttr.size() != 4) {
        // 如果没有 DSE 数据，这里可以 fallback 到默认值或者 fail
        // 为了稳健性，建议 fail 或者使用默认值 1
        return failure(); 
    }

    // 提取物理维度的切分参数
    int64_t dse_oh = cast<IntegerAttr>(dseAttr[0]).getInt();
    int64_t dse_ow = cast<IntegerAttr>(dseAttr[1]).getInt();
    int64_t dse_ic = cast<IntegerAttr>(dseAttr[2]).getInt();
    int64_t dse_oc = cast<IntegerAttr>(dseAttr[3]).getInt();

    // 4. 映射到 NCHWc32 的 9 个维度
    // Generic Loop Order: 
    // d0: N
    // d1: OC_chunk (Parallel)  <-- 对应 dse_oc
    // d2: OH       (Parallel)  <-- 对应 dse_oh
    // d3: OW       (Parallel)  <-- 对应 dse_ow
    // d4: IC_chunk (Reduction) <-- 对应 dse_ic
    // d5: KH       (Reduction)
    // d6: KW       (Reduction)
    // d7: IC_block (Reduction) <-- 固定 32，不分块 (0)
    // d8: OC_block (Parallel)  <-- 固定 32，不分块 (0)

    SmallVector<int64_t> tileSizes(9, 0); // 初始化为 0 (不切分)

    // 【关键】通道除以 32
    // 如果 DSE 给出的 ic 是 64，意味着我们要切 2 个 block
    // 确保至少为 1，避免除以 32 变成 0 导致变成 "不切分" (除非本来就是 0)
    tileSizes[1] = (dse_oc > 32) ? (dse_oc / 32) : 1; 
    tileSizes[2] = dse_oh;
    tileSizes[3] = dse_ow;
    tileSizes[4] = (dse_ic > 32) ? (dse_ic / 32) : 1;

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