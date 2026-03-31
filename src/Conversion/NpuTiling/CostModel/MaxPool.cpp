//======================================================
// src/Conversion/NpuTiling/CostModel/MaxPool.cpp
// This file implements the cost model for MaxPool operations,
// calculating tile sizes based on SRAM capacity constraints.
//======================================================

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <algorithm>

using namespace mlir;

namespace npux {

llvm::SmallVector<int64_t> NPUCostModel::getMaxPoolTileSizes(mlir::linalg::GenericOp op) {
  auto loopRanges = op.getStaticLoopRanges();
  
  // 我们的迭代空间是 6D [N, C, H_out, W_out, K_h, K_w]
  if (loopRanges.size() != 6) {
    return {1, 1, 1, 1, 0, 0}; // 安全回退
  }

  int64_t N = loopRanges[0];
  int64_t C = loopRanges[1];
  int64_t H_out = loopRanges[2];
  int64_t W_out = loopRanges[3];

  auto outputType = cast<RankedTensorType>(op.getOutputs()[0].getType());
  int64_t bitWidth = outputType.getElementType().getIntOrFloatBitWidth();
  int64_t bytesPerElem = std::max<int64_t>(1, bitWidth / 8);

  // 使用 NPUCostModel 内部的硬件配置
  int64_t spmSize = this->hw.spmSizeBytes;

  // 核心内存计算
  int64_t bytesPerOutPixel = 5 * bytesPerElem;
  int64_t maxPixels = spmSize / bytesPerOutPixel;

  // 失败时返回 6 个维度的默认值
  if (maxPixels <= 0) {
    return {1, 1, 1, 1, 0, 0}; 
  }

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

  llvm::SmallVector<int64_t> tileSizes = {1, t_c, t_h, t_w, 0, 0};

  // 打印日志
  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "[CostModel] MaxPool 2x2 NCHW: SPM=" << spmSize 
     << " Problem=[" << loopRanges[0] << ", " << loopRanges[1] << ", " 
     << loopRanges[2] << ", " << loopRanges[3] << "] "
     << "-> Tile=[N:" << tileSizes[0] << ", C:" << tileSizes[1]
     << ", H_out:" << tileSizes[2] << ", W_out:" << tileSizes[3] << "]\n";
  llvm::errs() << os.str();

  // 返回对应 [N, C, H, W, KH, KW] 的分块配置
  // 后两个 0 代表不切分 Kernel 维度（将完整的 2x2 窗口保留在内层）
  return tileSizes;
}

} // namespace npux