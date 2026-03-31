//=============================================================
// src/Conversion/NpuTiling/ElemWise.cpp
// this file is for elemwise op tiling pattern
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h" // 核心 Tiling 工具
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "npu-tiling"
using namespace mlir;
using namespace npux;

namespace {
SmallVector<int64_t> getElemWiseTileSizes(linalg::GenericOp op, StringRef opName) {
  
  // 1. 如果后续需要拦截 CLI 手动配置，可以在此处添加读取逻辑并直接 return

  // 2. 调用内置 Cost-Model 动态计算
  npux::HardwareConfig hwConfig;
  
  // 如果需要从全局变量覆写默认 hwConfig 参数，可以在此处操作
  // auto &config = npux::NPUConfig::getInstance();
  // hwConfig.spmSizeBytes = config.getSpmSize();
  // hwConfig.accSizeBytes = config.getAccSize();

  npux::NPUCostModel costModel(hwConfig);
  
  // NPUCostModel::getOptimalTileSizes 会根据 op 的 library_call 属性
  // 自动派发给 getGeluTileSizes 或 getMatAddTileSizes
  return costModel.getOptimalTileSizes(op);
}

// === 2. Tiling Pattern ===
struct NpuElemWiseTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {

    if (op->hasAttr("npu.tiled"))
      return failure();

    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (!libCall)
      return failure();

    StringRef opName = libCall.getValue();

    // 【修改点】：允许 npu_gelu 和 npu_matadd 通过
    if (opName != "npu_gelu" && opName != "npu_matadd") {
      return failure(); // 把机会留给其他 Tiling Pattern (如 Conv)
    }

    SmallVector<int64_t> rawTileSizes = getElemWiseTileSizes(op, opName);
    auto loopRanges = op.getStaticLoopRanges();

    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizes =
        getAsOpFoldResult(rewriter.getI64ArrayAttr(rawTileSizes));

    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult))
      return failure();

    for (auto loop : tilingResult->loops) {
      loop->setAttr("npu.target", rewriter.getStringAttr("npu"));
    }

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
      if (!loopOp)
        continue;

      scf::ForOp partialIteration;
      LogicalResult status =
          scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
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