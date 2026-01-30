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

static std::vector<int64_t> getDseAttrValues(linalg::GenericOp op) {
    auto dseAttr = op->getAttrOfType<ArrayAttr>("npu.dse_tiling");
    if (!dseAttr || dseAttr.size() != 4) {
        return {};
    }
    std::vector<int64_t> values;
    for (auto val : dseAttr) {
        values.push_back(cast<IntegerAttr>(val).getInt());
    }
    return values;
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

  if (opName == "npu_gelu") {
    // 1. 尝试从 Config (JSON/CLI) 读取
    configSizes = config.getGeluTileSize();

    // 2. 如果没有配置，则根据 SRAM 自动计算
    bool isAuto = false;
    if (configSizes.empty()) {
      isAuto = true;
      int64_t spmSize = config.getSpmSize();
      std::pair<int64_t, int64_t> autoHW = calculateAutoSpatialTile(op, spmSize);
      configSizes = {autoHW.first, autoHW.second};
    }

    applyTileConfigNCHWc32(sizes, configSizes);
    
    // [LOGGING] Gelu
    if (!configSizes.empty()) {
        llvm::errs() << "Tiling [Gelu] (" << (isAuto ? "Auto" : "Manual") << "): "
                     << "Shape Rank=[" << sizes.size() << "] "
                     << "-> Tile=[H:" << configSizes[0] << ", W:" << configSizes[1] << "]\n";
    }

  } else if (opName == "npu_conv") {
    // 1. 尝试获取 Config (CLI > JSON)
    configSizes = config.getConvTileSize();

    // 2. Fallback: 尝试获取 Attribute (DSE Tiling)
    if (configSizes.empty()) {
        configSizes = getDseAttrValues(op);
    }

    // 3. 如果找到了参数，进行维度映射
    if (configSizes.size() >= 4) {
        int64_t t_oh = configSizes[0];
        int64_t t_ow = configSizes[1];
        int64_t t_ic = configSizes[2];
        int64_t t_oc = configSizes[3];

        // [LOGGING] Conv (新增)
        llvm::errs() << "Tiling [Conv]: "
                     << "Shape Rank=[" << sizes.size() << "] "
                     << "-> Tile=[OH:" << t_oh << ", OW:" << t_ow 
                     << ", IC_blk:" << t_ic << ", OC_blk:" << t_oc << "]\n";

        // Generic Loop Order for Conv: 
        // d0:N, d1:OC_chunk, d2:OH, d3:OW, d4:IC_chunk, ...
        if (sizes.size() >= 5) { 
            sizes[1] = (t_oc > 32) ? (t_oc / 32) : 1; 
            sizes[2] = t_oh;
            sizes[3] = t_ow;
            sizes[4] = (t_ic > 32) ? (t_ic / 32) : 1;
        }
    }
  }

  return sizes;
}

static SmallVector<int64_t, 3> calculateAutoGemmTile(
    int64_t M, int64_t N, int64_t K, 
    int64_t spmSize, int64_t accSize) {

    // 硬件对齐参数
    const int64_t arraySizeH = 32;
    const int64_t arraySizeW = 32;
    const int64_t inputDtypeBytes = 1; // int8
    const int64_t accDtypeBytes = 4;   // int32

    // ---------------------------------------------------------
    // Step 1: 确定 Tm 和 Tn (基于 ACC 容量 - Output Stationary)
    // ---------------------------------------------------------
    int64_t maxAccElem = accSize / accDtypeBytes;
    
    int64_t targetDim = std::floor(std::sqrt(maxAccElem));
    
    // M 维度分块
    int64_t t_m = std::min(M, targetDim);
    if (t_m >= arraySizeH) t_m = (t_m / arraySizeH) * arraySizeH;
    else t_m = arraySizeH;

    // N 维度分块
    int64_t t_n = std::min(N, maxAccElem / t_m);
    if (t_n >= arraySizeW) t_n = (t_n / arraySizeW) * arraySizeW;
    else t_n = arraySizeW;

    // ACC 溢出修正
    while ((t_m * t_n * accDtypeBytes) > accSize) {
        t_n -= arraySizeW;
        if (t_n < arraySizeW) { t_n = arraySizeW; t_m -= arraySizeH; }
        if (t_m < arraySizeH) break; 
    }

    // ---------------------------------------------------------
    // Step 2: 确定 Tk (基于 SPM 容量)
    // ---------------------------------------------------------
    int64_t bytesPerK = (t_m + t_n) * inputDtypeBytes;
    int64_t t_k = K;

    if (bytesPerK > 0) {
        int64_t maxTk = spmSize / bytesPerK;
        t_k = std::min(K, maxTk);
    }

    // Tk 对齐
    if (t_k >= arraySizeH) t_k = (t_k / arraySizeH) * arraySizeH;
    else t_k = arraySizeH; 

    return {t_m, t_n, t_k};
}

SmallVector<int64_t> getGemmTileSizes(linalg::GenericOp op) {
    auto &config = npux::NPUConfig::getInstance();
    
    // 1. 获取手动配置
    std::vector<int64_t> manualSizes = config.getMatMulTileSize();
    
    // 2. 获取 Loop Ranges
    SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
    int64_t rank = loopRanges.size();
    
    if (rank < 3) {
        return {}; 
    }

    // 提取 M, N, K
    int64_t K = loopRanges[rank - 1];
    int64_t N = loopRanges[rank - 2];
    int64_t M = loopRanges[rank - 3];

    SmallVector<int64_t, 3> computedSizes;
    bool isManual = false;

    if (!manualSizes.empty() && manualSizes.size() >= 3) {
        computedSizes = {manualSizes[0], manualSizes[1], manualSizes[2]};
        isManual = true;
    } else {
        int64_t spmSize = config.getSpmSize();
        int64_t accSize = config.getAccSize();
        computedSizes = calculateAutoGemmTile(M, N, K, spmSize, accSize);
    }

    // [LOGGING] Gemm (新增)
    llvm::errs() << "Tiling [Gemm] (" << (isManual ? "Manual" : "Auto") << "): "
                 << "Problem=[M:" << M << ", N:" << N << ", K:" << K << "] "
                 << "-> Tile=[Tm:" << computedSizes[0] 
                 << ", Tn:" << computedSizes[1] 
                 << ", Tk:" << computedSizes[2] << "]\n";

    // 3. 构建最终的 Tile Sizes 数组
    SmallVector<int64_t> finalTileSizes(rank, 0);

    // [..., tm, tn, tk]
    finalTileSizes[rank - 3] = computedSizes[0]; // M -> tm
    finalTileSizes[rank - 2] = computedSizes[1]; // N -> tn
    finalTileSizes[rank - 1] = computedSizes[2]; // K -> tk

    return finalTileSizes;
}


bool isTilingNecessary(ArrayRef<int64_t> tileSizes, ArrayRef<int64_t> loopRanges) {
    if (tileSizes.empty()) return false;
    
    // 遍历每一个维度
    for (size_t i = 0; i < std::min(tileSizes.size(), loopRanges.size()); ++i) {
        int64_t tile = tileSizes[i];
        int64_t range = loopRanges[i];

        // 只有当 TileSize 有效(>0) 且确实小于 ProblemSize 时，才需要切分
        if (tile > 0 && tile < range) {
            return true; 
        }
    }
    return false;
}

} // namespace npux