//=============================================================
// src/Conversion/NpuTiling/ConvInnerTiling.cpp
// Second pass: Inner tiling for Cout/Cin chunks [1, 1]
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

struct NpuConvInnerTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    
    // 1. 检查是否是 NPU 卷积
    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (!libCall || libCall.getValue() != "npu_conv") {
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

    // 4. 设置切分参数：只切分 OC_chunk (d1) 和 IC_chunk (d4) 为 1
    // d1: OC_chunk (Parallel)
    // d4: IC_chunk (Reduction)
    SmallVector<int64_t> tileSizes(9, 0); 
    tileSizes[1] = 1; // Inner OC Loop
    tileSizes[4] = 1; // Inner IC Loop

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
    // 新增：给生成的 Loop 打 Label
    // scf::tileUsingSCF 生成的 loops 顺序对应 tileSizes 中非零维度的顺序 (从小到大)
    // tileSizes[1] 是 OC (Chunk), tileSizes[4] 是 IC (Chunk)
    // 所以 loops[0] 是 OC Loop, loops[1] 是 IC Loop
    // ==========================================================
    auto loops = tilingResult->loops;

    // 标记 OC Inner Loop
    if (loops.size() > 0) {
        if (auto loop = dyn_cast<scf::ForOp>(loops[0].getOperation())) {
            loop->setAttr("npu.loop_type", rewriter.getStringAttr("inner_oc"));
            // 可选：方便 Debug 查看
            loop->setAttr("npu.loop_name", rewriter.getStringAttr("Loop_Inner_Cout")); 
        }
    }

    // 标记 IC Inner Loop
    if (loops.size() > 1) {
        if (auto loop = dyn_cast<scf::ForOp>(loops[1].getOperation())) {
            loop->setAttr("npu.loop_type", rewriter.getStringAttr("inner_ic"));
            // 可选：方便 Debug 查看
            loop->setAttr("npu.loop_name", rewriter.getStringAttr("Loop_Inner_Cin"));
        }
    }

    // 6. 标记新生成的 Op (最内层的 GenericOp)
    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.inner_tiled", rewriter.getUnitAttr());
      
      // 继承属性
      if (op->hasAttr("npu.layer_name")) {
         tiledOp->setAttr("npu.layer_name", op->getAttr("npu.layer_name"));
      }
      if (op->hasAttr("npu.accumulate_mode")) {
         tiledOp->setAttr("npu.accumulate_mode", op->getAttr("npu.accumulate_mode"));
      }
    }

    // 7. 替换原 Op
    rewriter.replaceOp(op, tilingResult->replacements);

    return success();
  }
};

} // namespace

void npux::populateConvInnerTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuConvInnerTilingPattern>(context);
};