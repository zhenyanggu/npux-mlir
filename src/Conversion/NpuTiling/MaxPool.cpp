//=============================================================
// src/Conversion/NpuTiling/MaxPool.cpp
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Pass/Passes.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

#define DEBUG_TYPE "npu-tiling"
using namespace mlir;
using namespace npux; 

namespace {

SmallVector<int64_t> calculateAutoMaxPoolTileNCHW(
    linalg::GenericOp op, int64_t spmSize) {
  
  auto loopRanges = op.getStaticLoopRanges();
  // 【修改1】现在我们的迭代空间是 6D [N, C, H_out, W_out, K_h, K_w]
  if (loopRanges.size() != 6) return {1, 1, 1, 1, 0, 0}; 

  int64_t N = loopRanges[0];
  int64_t C = loopRanges[1];
  int64_t H_out = loopRanges[2];
  int64_t W_out = loopRanges[3];

  auto outputType = cast<RankedTensorType>(op.getOutputs()[0].getType());
  int64_t bitWidth = outputType.getElementType().getIntOrFloatBitWidth();
  int64_t bytesPerElem = std::max<int64_t>(1, bitWidth / 8);

  // 核心内存计算保持不变
  int64_t bytesPerOutPixel = 5 * bytesPerElem;
  int64_t maxPixels = spmSize / bytesPerOutPixel;

  // 【修改2】失败时返回 6 个维度的默认值
  if (maxPixels <= 0) return {1, 1, 1, 1, 0, 0}; 

  // 初始化 Tiling Sizes
  int64_t t_c = 1, t_h = 1, t_w = 1;
  int64_t remaining_pixels = maxPixels;

  // 1. 优先填满 W 维度
  t_w = std::min<int64_t>(W_out, remaining_pixels);
  remaining_pixels /= t_w;

  // 2. 尝试填满 H 维度
  if (remaining_pixels > 0) {
      t_h = std::min<int64_t>(H_out, remaining_pixels);
      remaining_pixels /= t_h;
  }

  // 3. 最后利用剩余空间切分 C 维度
  if (remaining_pixels > 0) {
      t_c = std::min<int64_t>(C, remaining_pixels);
  }

  // 【修改3】返回对应 [N, C, H, W, KH, KW] 的分块配置
  // 后两个 0 代表不切分 Kernel 维度（将完整的 2x2 窗口保留在内层）
  return {1, t_c, t_h, t_w, 0, 0};
}

SmallVector<int64_t> getMaxPoolTileSizes(linalg::GenericOp op, StringRef opName) {
  auto &config = npux::NPUConfig::getInstance();
  int64_t spmSize = config.getSpmSize();

  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  SmallVector<int64_t> tileSizes(loopRanges.size(), 0);

  std::string msg;
  llvm::raw_string_ostream os(msg);

  // 【修改4】这里要匹配 6D 的 loopRanges
  if (opName == "npu_maxpool" && loopRanges.size() == 6) {
    tileSizes = calculateAutoMaxPoolTileNCHW(op, spmSize);

    os << "Tiling [MaxPool 2x2 NCHW] (Auto): SPM=" << spmSize 
       << " Problem=[" << loopRanges[0] << ", " << loopRanges[1] << ", " 
       << loopRanges[2] << ", " << loopRanges[3] << "] "
       << "-> Tile=[N:" << tileSizes[0] << ", C:" << tileSizes[1]
       << ", H_out:" << tileSizes[2] << ", W_out:" << tileSizes[3] << "]\n";
    llvm::errs() << os.str();
  }

  return tileSizes;
}

// === MaxPool Tiling Pattern ===
struct NpuMaxPoolTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    
    // 避免无限循环 Tiling
    if (op->hasAttr("npu.tiled")) return failure();

    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (!libCall || libCall.getValue() != "npu_maxpool") {
        return failure();
    }

    // 1. 调用 Helper 获取 TileSize
    SmallVector<int64_t> rawTileSizes = getMaxPoolTileSizes(op, libCall.getValue());
    if (rawTileSizes.empty()) return failure();

    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizes = getAsOpFoldResult(rewriter.getI64ArrayAttr(rawTileSizes));
    
    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    // 2. 执行 Tiling
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult)) return failure();

    // =========================================================================
    // 【核心新增】：过河拆桥！将切分后的 6D MaxPool 降维回 4D，断开 Dummy Window 依赖
    // =========================================================================
    for (Operation *tiledOp : tilingResult->tiledOps) {
      auto genericOp = dyn_cast<linalg::GenericOp>(tiledOp);
      if (!genericOp || genericOp.getNumDpsInputs() != 2) continue;

      rewriter.setInsertionPoint(genericOp);
      Value realInput = genericOp.getInputs()[0];   // 切好的真实特征图切片
      Value outInit = genericOp.getOutputs()[0];    // 切好的输出切片

      // 构建 4D 迭代器 (全部 parallel)
      SmallVector<utils::IteratorType> iteratorTypes(4, utils::IteratorType::parallel);
      
      // 构建 4D AffineMap: 重新使用 stride=2 的映射，保证 4D 也能通过 MLIR 校验
      auto n = rewriter.getAffineDimExpr(0);
      auto c = rewriter.getAffineDimExpr(1);
      auto oh = rewriter.getAffineDimExpr(2);
      auto ow = rewriter.getAffineDimExpr(3);
      auto inputMap = AffineMap::get(4, 0, {n, c, oh * 2, ow * 2}, rewriter.getContext());
      auto outputMap = rewriter.getMultiDimIdentityMap(4);

      // 创建崭新的单输入 4D linalg.generic
      auto new4DOp = rewriter.create<linalg::GenericOp>(
          genericOp.getLoc(),
          outInit.getType(),
          ValueRange{realInput}, // 只有真实输入，抛弃 Dummy Window
          ValueRange{outInit},
          ArrayRef<AffineMap>{inputMap, outputMap},
          iteratorTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
              // 填一个合法的纯输出，因为 NPU Backend 只认 library_call，里面写啥无所谓
              b.create<linalg::YieldOp>(loc, args[0]); 
          });

      // 继承 NPU Attributes
      new4DOp->setAttr("library_call", rewriter.getStringAttr("npu_maxpool"));
      new4DOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
      new4DOp->setAttr("npu.tiled", rewriter.getUnitAttr());

      // 替换掉带 dummy 的 6D op
      rewriter.replaceOp(genericOp, new4DOp.getResults());
    }
    // =========================================================================

    // 赋予下游所需的 NPU Attributes 给 loops
    for (auto loop : tilingResult->loops) {
      loop->setAttr("npu.target", rewriter.getStringAttr("npu"));
    }

    // 注意：这里删除了对 tiledOps 遍历 setAttr 的原逻辑，因为我们在上面 new4DOp 时已经设好了

    // 3. Peeling 处理 Tail 边界情况（保持不变）
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

void npux::populateMaxPoolTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuMaxPoolTilingPattern>(context);
}