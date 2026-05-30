//======================================================
// src/Conversion/NpuTiling/CostModel/Layout.cpp
// This file implements the cost model for layout transformation 
// operations (Transpose, NCHW<->NCHWc32), calculating tile sizes 
// based on SRAM capacity constraints.
//======================================================

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include <cmath>
#include <algorithm>

using namespace mlir;

namespace npux {

static std::pair<int64_t, int64_t> calculateAutoTransposeTile(
    RankedTensorType outputType, int64_t spmSize) {
  int64_t bitWidth = outputType.getElementType().getIntOrFloatBitWidth();
  int64_t bytesPerElem = std::max<int64_t>(1, bitWidth / 8);

  ArrayRef<int64_t> shape = outputType.getShape();
  if (shape.size() < 2)
    return {32, 32};
  int64_t M = shape[shape.size() - 2];
  int64_t N = shape[shape.size() - 1];

  // 3. 计算 SPM 能放下的最大元素对 (Input + Output 都要放进 SPM)
  // 占用内存 = (Tm * Tn * bytesPerElem) * 2
  int64_t maxElems = spmSize / (bytesPerElem * 2);

  if (maxElems <= 0) return {1, 1};

  // 4. 计算方形分块的目标边长 (平衡读写的 DMA stride)
  int64_t targetDim = std::floor(std::sqrt(maxElems));

  // 硬件对齐优化：向 16 或 32 舍入
  targetDim = (targetDim / 16) * 16;
  if (targetDim == 0) targetDim = 16; // 保证最小分块

  int64_t t_m = std::min<int64_t>(M, targetDim);
  int64_t t_n = std::min<int64_t>(N, targetDim);

  // 如果某个维度比较小没达到 targetDim，可以将剩余空间补偿给另一个维度
  if (t_m < targetDim) {
      t_n = std::min<int64_t>(N, (maxElems / t_m) / 16 * 16);
      if (t_n == 0) t_n = 16;
  } else if (t_n < targetDim) {
      t_m = std::min<int64_t>(M, (maxElems / t_n) / 16 * 16);
      if (t_m == 0) t_m = 16;
  }

  return {t_m, t_n};
}

static SmallVector<int64_t> calculateAutoLayoutTileNCHWc32(
    RankedTensorType outputType, int64_t spmSize) {
  ArrayRef<int64_t> loopRanges = outputType.getShape();
  int64_t C_blk = loopRanges[1];
  int64_t H = loopRanges[2];
  int64_t W = loopRanges[3];
  int64_t inner_c = loopRanges[4];
  int64_t bytesPerElem = std::max<int64_t>(1, outputType.getElementType().getIntOrFloatBitWidth() / 8);

  // 每个空间像素的内存占用 (Input + Output 各占 inner_c)
  int64_t bytesPerPixel = (inner_c * 2) * bytesPerElem;
  int64_t maxPixels = spmSize / bytesPerPixel;

  if (maxPixels <= 0) return {1, 1, 1, 1, 0}; 

  // 初始化各维度分块大小，t_n 永远为 1
  int64_t t_c = 1, t_h = 1, t_w = 1;
  int64_t remaining_pixels = maxPixels;

  // 1. 优先填满 W 维度 (行连续)
  t_w = std::min<int64_t>(W, remaining_pixels);
  remaining_pixels /= t_w;

  // 2. 尝试填满 H 维度
  if (remaining_pixels > 0) {
      t_h = std::min<int64_t>(H, remaining_pixels);
      remaining_pixels /= t_h;
  }

  // 3. H 和 W 都填满后，如果还有空间，继续放大 C_blk
  if (remaining_pixels > 0) {
      t_c = std::min<int64_t>(C_blk, remaining_pixels);
      // N 不再分块，所以不需要再更新 remaining_pixels 了
  }

  // 返回 Rank 5 的 Tile Size 数组: N 永远是 1，最内层 c 永远是 0
  return {1, t_c, t_h, t_w, 0};
}

llvm::SmallVector<int64_t> NPUCostModel::getTransposeTileSizes(
    npucore::TransposeOp op) {
  auto outputType =
      dyn_cast<RankedTensorType>(op.getOutputs().front().getType());
  if (!outputType) {
    return {};
  }
  int64_t spmSize = this->hw.spmSizeBytes;
  SmallVector<int64_t> loopRanges(
      outputType.getShape().begin(), outputType.getShape().end());
  int64_t rank = loopRanges.size();
  SmallVector<int64_t> tileSizes(rank, 0);
  auto tileHW = calculateAutoTransposeTile(outputType, spmSize);
  tileSizes[rank - 2] = tileHW.first;
  tileSizes[rank - 1] = tileHW.second;
  return tileSizes;
}

llvm::SmallVector<int64_t> NPUCostModel::getLayoutPackTileSizes(
    npucore::LayoutNchwToNchwc32Op op) {
  auto outputType =
      dyn_cast<RankedTensorType>(op.getOutputs().front().getType());
  if (!outputType || outputType.getRank() != 5)
    return {};
  return calculateAutoLayoutTileNCHWc32(outputType, this->hw.spmSizeBytes);
}

llvm::SmallVector<int64_t> NPUCostModel::getLayoutUnpackTileSizes(
    npucore::LayoutNchwc32ToNchwOp op) {
  auto inputType = dyn_cast<RankedTensorType>(op.getInputs().front().getType());
  if (!inputType || inputType.getRank() != 5)
    return {};

  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t cBlk = inputShape[1];
  int64_t h = inputShape[2];
  int64_t w = inputShape[3];
  int64_t inner = inputShape[4];
  int64_t bytesPerElem =
      std::max<int64_t>(1, inputType.getElementType().getIntOrFloatBitWidth() / 8);
  int64_t maxPixels = this->hw.spmSizeBytes / ((inner * 2) * bytesPerElem);
  if (maxPixels <= 0)
    return {};

  int64_t tileCBlk = 1;
  int64_t tileH = 1;
  int64_t tileW = std::min<int64_t>(w, maxPixels);
  int64_t remainingPixels = maxPixels / std::max<int64_t>(tileW, 1);
  if (remainingPixels > 0) {
    tileH = std::min<int64_t>(h, remainingPixels);
    remainingPixels /= std::max<int64_t>(tileH, 1);
  }
  if (remainingPixels > 0)
    tileCBlk = std::min<int64_t>(cBlk, remainingPixels);

  auto outputType =
      dyn_cast<RankedTensorType>(op.getOutputs().front().getType());
  if (!outputType || outputType.getRank() != 4)
    return {};
  return {1, std::min<int64_t>(outputType.getShape()[1], tileCBlk * inner),
      tileH, tileW};
}

} // namespace npux
