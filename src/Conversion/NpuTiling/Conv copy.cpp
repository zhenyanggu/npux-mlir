//=============================================================
// src/Conversion/NpuTiling/Conv.cpp
// this file is for Conv op tiling pattern
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Pass/Passes.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"

using namespace mlir;
using namespace npux;

namespace {

// 定义维度名称映射，对应 helper 中的 NCHWc32 逻辑
// d0:N, d1:OC, d2:OH, d3:OW, d4:IC
static const SmallVector<StringRef> kDimensionLabels = {"OC", "OH", "OW", "IC"};

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

    // 3. 获取切分配置
    SmallVector<int64_t> tileSizes = getNpuTileSizes(op);
    auto loopRanges = op.getStaticLoopRanges();

    // 4. 检查是否真的需要切分
    if (!isTilingNecessary(tileSizes, loopRanges)) {
        op->setAttr("npu.tiled", rewriter.getUnitAttr());
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

    // 6. 标记新生成的 Op (Inner Compute Op)
    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
    }

    // =================================================================
    // [加强点] Step 6.5: 给 Loop 打上维度标签 (Labeling)
    // scf::tileUsingSCF 返回的 loops 仅包含被切分的维度 (tileSize > 0)
    // 我们需要遍历 tileSizes 来对齐维度索引
    // =================================================================
    auto generatedLoops = tilingResult->loops;
    int currentLoopIdx = 0;

    for (size_t dimIdx = 0; dimIdx < tileSizes.size(); ++dimIdx) {
        // 如果这个维度没有被切分 (size == 0)，则跳过，因为没有生成对应的 Loop
        if (tileSizes[dimIdx] == 0) continue;

        // 安全检查
        if (currentLoopIdx >= generatedLoops.size()) break;

        // 获取对应的 Loop Operation
        auto loopOp = generatedLoops[currentLoopIdx].getOperation();
        
        // 确定 Label 名称
        StringRef label = "UNKNOWN";
        if (dimIdx < kDimensionLabels.size()) {
            label = kDimensionLabels[dimIdx];
        }

        // 打标：例如 npu.loop_dim = "OC"
        loopOp->setAttr("npu.loop_dim", rewriter.getStringAttr(label));
        loopOp->setAttr("npu.computeop",rewriter.getStringAttr("conv"));

        // 移动到下一个生成的 Loop
        currentLoopIdx++;
    }
    // =================================================================


    // 7. 处理 Peeling (解决不能整除的问题)
    auto loops = tilingResult->loops; // 使用 tilingResult 中的 loops
    SmallVector<Value> finalResults = tilingResult->replacements;

    // 倒序遍历处理 Peeling (从内向外：IC -> OW -> OH -> OC)
    for (int i = loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp) continue;

      scf::ForOp partialIteration;
      LogicalResult status = scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
        // 标记 Tail Op
        partialIteration->setAttr("npu.peeled_tail", rewriter.getUnitAttr());
        
        // 如果 Peeling 发生，scf.for 的结果可能会变，需要更新 replacements
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