//=============================================================
// src/Conversion/NpuTiling/ElemWiseInnerTiling.cpp
// Second pass: Inner tiling for ElemWise C chunks [1]
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

struct NpuElemWiseInnerTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    
    // 1. 检查是否是 NPU Gelu (与其他 Elementwise)
    // 根据上一层逻辑，这里主要针对 npu_gelu，或者你可以放宽条件
    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (!libCall || libCall.getValue() != "npu_gelu") {
      return failure();
    }

    // 2. [关键] 必须是已经被外层 Pass 切分过的 Op
    if (!op->hasAttr("npu.tiled")) {
      return failure();
    }

    if (op->hasAttr("npu.trivial_tiling")) {
      return failure();
    }

    // 3. [关键] 防止死循环：检查是否已经进行过内层切分
    if (op->hasAttr("npu.inner_tiled")) {
      return failure();
    }

    // 4. 设置切分参数
    // ElementWise on NCHWc32 通常是 5 维:
    // d0: N
    // d1: C_chunk (Parallel) <-- Target: 1
    // d2: H       (Parallel)
    // d3: W       (Parallel)
    // d4: C_block (Parallel/Reduction depending on impl, usually 32)
    
    // 获取 Rank 确保安全 (通常为 5)
    auto rank = op.getNumLoops();
    SmallVector<int64_t> tileSizes(rank, 0); 

    // 只要 Rank 足够，我们强制切分 d1 (C Chunk)
    if (rank > 1) {
        tileSizes[1] = 1; 
    } else {
        return failure(); // 维度不足，无法切分 C Chunk
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

    // ==========================================================
    // 给 Loop 打 Label
    // 因为我们只切分了 d1，所以 loops 数组里应该只有一个 scf.for
    // ==========================================================
    auto loops = tilingResult->loops;
    if (!loops.empty()) {
        if (auto loop = dyn_cast<scf::ForOp>(loops[0].getOperation())) {
            loop->setAttr("npu.loop_type", rewriter.getStringAttr("inner_c"));
            // 可选：Debug 用
            loop->setAttr("npu.loop_name", rewriter.getStringAttr("Loop_Inner_Channel")); 
        }
    }

    // 6. 标记新生成的 Op (最内层)
    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.inner_tiled", rewriter.getUnitAttr());
      
      // 继承 Layer Name
      if (op->hasAttr("npu.layer_name")) {
         tiledOp->setAttr("npu.layer_name", op->getAttr("npu.layer_name"));
      }
    }

    // 7. 替换原 Op
    rewriter.replaceOp(op, tilingResult->replacements);

    return success();
  }
};

} // namespace

void npux::populateElemWiseInnerTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuElemWiseInnerTilingPattern>(context);
};