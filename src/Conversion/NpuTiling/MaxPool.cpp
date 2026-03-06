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
  // 安全检查：NCHW 迭代空间为 Rank 4 [N, C, H_out, W_out]
  if (loopRanges.size() != 4) return {1, 1, 1, 1}; 

  int64_t N = loopRanges[0];
  int64_t C = loopRanges[1];
  int64_t H_out = loopRanges[2];
  int64_t W_out = loopRanges[3];

  auto outputType = cast<RankedTensorType>(op.getOutputs()[0].getType());
  int64_t bitWidth = outputType.getElementType().getIntOrFloatBitWidth();
  int64_t bytesPerElem = std::max<int64_t>(1, bitWidth / 8);

  // 核心内存计算 (针对 2x2 MaxPool, stride=2):
  // 1 个 Output 元素对应 2x2 = 4 个 Input 元素
  // 暂存这 1 个输出像素的计算，SRAM 需要的空间为 1(Out) + 4(In) = 5 个元素大小
  int64_t bytesPerOutPixel = 5 * bytesPerElem;
  int64_t maxPixels = spmSize / bytesPerOutPixel;

  if (maxPixels <= 0) return {1, 1, 1, 1}; // SPM极度受限的保护

  // 初始化 Tiling Sizes
  int64_t t_c = 1, t_h = 1, t_w = 1;
  int64_t remaining_pixels = maxPixels;

  // 1. 优先填满 W 维度 (NCHW 下 W 是最内侧维度，连续 DMA 效率最高)
  t_w = std::min<int64_t>(W_out, remaining_pixels);
  remaining_pixels /= t_w;

  // 2. 尝试填满 H 维度 (获取完整的特征图平面)
  if (remaining_pixels > 0) {
      t_h = std::min<int64_t>(H_out, remaining_pixels);
      remaining_pixels /= t_h;
  }

  // 3. 最后利用剩余空间切分 C 维度 (Channel)
  if (remaining_pixels > 0) {
      t_c = std::min<int64_t>(C, remaining_pixels);
  }

  // 返回对应 [N, C, H, W] 的分块配置
  return {1, t_c, t_h, t_w};
}

SmallVector<int64_t> getMaxPoolTileSizes(linalg::GenericOp op, StringRef opName) {
  auto &config = npux::NPUConfig::getInstance();
  int64_t spmSize = config.getSpmSize();

  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  SmallVector<int64_t> tileSizes(loopRanges.size(), 0);

  std::string msg;
  llvm::raw_string_ostream os(msg);

  if (opName == "npu_maxpool" && loopRanges.size() == 4) {
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

    // MaxPool 当前是通过 output->input 的缩放 map 来表达 2x2/stride=2，
    // 通用 tileUsingSCF 在“整块不切分”场景下仍会物化 input slice，
    // 但它无法为 pooling window 自动补 halo，最终会把 56x56 错切成 55x55。
    // 因此当 TileSize 已经覆盖完整迭代空间时，直接打标记跳过实际 tiling。
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

    // 2. 执行 Tiling
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult)) return failure();

    // 赋予下游所需的 NPU Attributes
    for (auto loop : tilingResult->loops) {
      loop->setAttr("npu.target", rewriter.getStringAttr("npu"));
    }

    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
    }

    // 3. Peeling 处理 Tail 边界情况（处理特征图无法被完美整除的情况）
    auto loops = tilingResult->loops;
    SmallVector<Value> finalResults = tilingResult->replacements;

    for (int i = loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp) continue;

      scf::ForOp partialIteration;
      LogicalResult status = scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
        // 标记尾部，方便下游降级处理不规则的尺寸
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