//=======================================================
// src/Conversion/NpuTiling/NpuTilingHelper.cpp
// this file contains helper functions for npu tiling patterns
//=======================================================


#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h" // 核心 Tiling 工具
#include "mlir/Dialect/SCF/Transforms/Transforms.h"

#include "src/Compiler/NpuConfig.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

#include <algorithm> // 必须添加
#include <cmath>


using namespace mlir;

static const std::vector<int64_t> HARDCODED_GELU_DEFAULT = {16, 16};
static const std::vector<int64_t> HARDCODED_CONV_DEFAULT = {16, 16};

namespace npux {
void applyTileConfigNCHWc32(
    SmallVectorImpl<int64_t> &sizes, const std::vector<int64_t> &configSizes) {
  if (configSizes.empty())
    return;

  int opRank = sizes.size(); 
  int configRank = configSizes.size();

  int opIdx = opRank - 2; // 跳过 c32
  int cfgIdx = configRank - 1;

  // 从 W 维度向前遍历
  while (opIdx >= 0 && cfgIdx >= 0) {
    sizes[opIdx] = configSizes[cfgIdx];
    opIdx--;
    cfgIdx--;
  }
}

std::pair<int64_t, int64_t> calculateAutoSpatialTile(
    linalg::GenericOp op, int64_t spmSize) {
  
  // 1. 获取 Output Tensor 类型信息
  auto outputType = cast<RankedTensorType>(op.getOutputs()[0].getType());
  auto shape = outputType.getShape();
  int64_t rank = outputType.getRank();

  // 安全检查：必须是 NCHWc32 (Rank 5)
  if (rank != 5) {
    return {16, 16}; // 非标准格式，回退到保守值
  }

  // 2. 获取维度信息
  // NCHWc32: [N, C_out, H, W, c32]
  int64_t N = shape[0];
  int64_t C_out = shape[1];
  int64_t H = shape[2];
  int64_t W = shape[3];
  int64_t C_in = shape[4]; // 应该是 32

  // 处理动态维度 (ShapedType::kDynamic)，如果遇到动态维度，保守处理
  if (N < 0) N = 1; 
  if (C_out < 0) C_out = 1; 

  // 3. 计算元素大小 (Bytes)
  int64_t bitWidth = outputType.getElementType().getIntOrFloatBitWidth();
  int64_t bytesPerElem = bitWidth / 8;
  if (bytesPerElem == 0) bytesPerElem = 1; // 避免 i1 等情况

  // 4. 计算固定维度的开销 (N * C_out * c32)
  // 也就是说，每一个空间像素 (1x1) 在内存中占用的体积
  int64_t pixelVol = N * C_out * C_in; 
  int64_t bytesPerPixel = pixelVol * bytesPerElem;

  // 5. 计算 SRAM 可容纳的最大空间像素数
  int64_t maxTotalPixels = spmSize / (bytesPerPixel);

  if (maxTotalPixels <= 0) {
    // SRAM 太小连一个像素的完整 Channel 都放不下，这是异常情况
    // 强制最小分块 1x1
    return {1, 1};
  }

  // 6. 贪心策略：优先填满 W (行连续)，再增加 H
  int64_t tileW = W;
  int64_t tileH = 1;

  if (maxTotalPixels >= W) {
    // SRAM 足够放下一整行 (甚至多行)
    tileW = W;
    // 计算能放多少行
    tileH = maxTotalPixels / W;
    // 不超过实际高度
    tileH = std::min(tileH, H);
  } else {
    // SRAM 放不下一整行，只能切分 W，H 固定为 1
    tileH = 1;
    tileW = maxTotalPixels;
  }

  // 7. 对齐建议 (可选)：为了 SIMD 效率，H 最好是偶数或特定倍数
  // 这里简单处理：如果 tileH > 1，尽量保持不做太细碎的切分
  // (用户现在的需求是"尽量放满"，暂不强制对齐)

  return {tileH, tileW};
}


SmallVector<int64_t> getNpuTileSizes(linalg::GenericOp op) {
  auto libCall = op->getAttrOfType<StringAttr>("library_call");
  if (!libCall) return {};

  StringRef opName = libCall.getValue();

  unsigned rank = 0;
  if (op.getOutputs().size() > 0) {
    if (auto type = dyn_cast<RankedTensorType>(op.getOutputs()[0].getType())) {
      rank = type.getRank();
    }
  }
  if (rank == 0) return {};

  SmallVector<int64_t> sizes(rank, 0);
  auto &config = npux::NPUConfig::getInstance();
  std::vector<int64_t> configSizes;

  // ================= MODIFIED START =================
  if (opName == "npu_gelu") {
    // 1. 尝试从 Config (JSON/CLI) 读取
    configSizes = config.getGeluTileSize();

    // 2. 如果没有配置，则根据 SRAM 自动计算
    if (configSizes.empty()) {
      // 获取 SRAM 大小 (e.g., 128KB -> 131072)
      int64_t spmSize = config.getSpmSize();
      
      // 调用计算逻辑
      std::pair<int64_t, int64_t> autoHW = calculateAutoSpatialTile(op, spmSize);
      
      // 注意：calculateAutoSpatialTile 返回的是 {H, W}
      // 我们需要放入 configSizes 向量，随后由 applyTileConfigNCHWc32 映射
      configSizes = {autoHW.first, autoHW.second};
      

      llvm::errs() << "Auto Tiling Gelu: SRAM=" << spmSize 
                   << ", Shape=[" << sizes.size() << "]"
                   << " -> Tile=[" << autoHW.first << "x" << autoHW.second << "]\n";
    }

    applyTileConfigNCHWc32(sizes, configSizes);

  } else if (opName == "npu_conv") {
    // Conv 逻辑保持不变，或者你也想应用类似的逻辑？
    configSizes = config.getConvTileSize();
    if (configSizes.empty()) {
      configSizes = HARDCODED_CONV_DEFAULT;
    }
    applyTileConfigNCHWc32(sizes, configSizes);
  }
  // ================= MODIFIED END =================

  return sizes;
}
} // namespace npux
