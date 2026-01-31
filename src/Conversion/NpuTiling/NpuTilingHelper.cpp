//=======================================================
// src/Conversion/NpuTiling/NpuTilingHelper.cpp
// this file contains helper functions for npu tiling patterns
//=======================================================

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h" 
#include "mlir/Dialect/SCF/Transforms/Transforms.h"

#include "src/Compiler/NpuConfig.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

#include <algorithm> 
#include <cmath>
#include <llvm/Support/raw_ostream.h>

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
        std::string msg;
        llvm::raw_string_ostream os(msg);
        os << "Tiling [Gelu] (" << (isAuto ? "Auto" : "Manual") << "): "
           << "Shape Rank=[" << sizes.size() << "] "
           << "-> Tile=[H:" << configSizes[0] << ", W:" << configSizes[1] << "]\n";
        llvm::errs() << os.str(); 
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
        std::string msg;
        llvm::raw_string_ostream os(msg);
        os << "Tiling [Conv]: "
           << "Shape Rank=[" << sizes.size() << "] "
           << "-> Tile=[OH:" << t_oh << ", OW:" << t_ow 
           << ", IC_blk:" << t_ic << ", OC_blk:" << t_oc << "]\n";
        llvm::errs() << os.str();

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
    const int64_t inputDtypeBytes = 1;  // int8 (输入)
    const int64_t outputDtypeBytes = 1; // int8 (输出到SPM也是int8)
    const int64_t accDtypeBytes = 4;    // int32 (ACC累加)

    // ---------------------------------------------------------
    // Step 1: 初始估计 Tm 和 Tn (基于 ACC 容量)
    // ---------------------------------------------------------
    int64_t maxAccElem = accSize / accDtypeBytes;
    
    int64_t targetDim = std::floor(std::sqrt(maxAccElem));
    
    // M 维度初步分块
    int64_t t_m = std::min(M, targetDim);
    if (t_m >= arraySizeH) t_m = (t_m / arraySizeH) * arraySizeH;
    else t_m = arraySizeH;

    // N 维度初步分块
    int64_t t_n = std::min(N, maxAccElem / t_m);
    if (t_n >= arraySizeW) t_n = (t_n / arraySizeW) * arraySizeW;
    else t_n = arraySizeW;

    // ---------------------------------------------------------
    // Step 2: 联合调整 Tm, Tn, Tk (基于 ACC 和 SPM 容量)
    // ---------------------------------------------------------
    // 这里需要循环，因为如果 SPM 放不下 (Input + Output)，
    // 我们需要缩小 Tm/Tn 来腾出空间。
    int64_t t_k = arraySizeH; // 初始设为最小对齐单位

    while (true) {
        // 1. 检查 ACC 限制 (Accumulator overflow check)
        // ---------------------------------------------
        bool accFits = (t_m * t_n * accDtypeBytes) <= accSize;

        // 2. 检查 SPM 限制 (SPM overflow check)
        // ---------------------------------------------
        // Output 占用: Tm * Tn * 1 byte
        int64_t outputSpmBytes = t_m * t_n * outputDtypeBytes;
        
        // Input 单位 K 占用: (Tm + Tn) * 1 byte
        int64_t inputBytesPerK = (t_m + t_n) * inputDtypeBytes;
        
        // 计算 SPM 中剩余给 Input 的空间
        int64_t remainingSpmForInput = spmSize - outputSpmBytes;

        // 至少要能放下一个最小单位的 Tk (arraySizeH)
        bool spmFits = (remainingSpmForInput >= (inputBytesPerK * arraySizeH));

        // 3. 如果 ACC 或 SPM 爆了，缩小 Tm/Tn
        // ---------------------------------------------
        if (!accFits || !spmFits) {
            t_n -= arraySizeW; // 优先缩减 N
            if (t_n < arraySizeW) {
                t_n = arraySizeW;
                t_m -= arraySizeH; // N 缩无可缩，缩 M
            }
            
            // 保护机制：如果连最小块都放不下（极少见），强制退出
            if (t_m < arraySizeH) {
                t_m = arraySizeH;
                t_n = arraySizeW;
                break; 
            }
            continue; // 重新检查新的 Tm/Tn
        }

        // 4. 计算最终的 Tk
        // ---------------------------------------------
        // 到这里说明 Tm, Tn 既符合 ACC，也给 SPM 留出了至少 32*K 的空间
        int64_t maxTk = remainingSpmForInput / inputBytesPerK;
        t_k = std::min(K, maxTk);
        
        // Tk 对齐
        if (t_k >= arraySizeH) t_k = (t_k / arraySizeH) * arraySizeH;
        else t_k = arraySizeH;
        
        // 成功找到合适的分块
        break;
    }

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

    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "Tiling [Gemm] (" << (isManual ? "Manual" : "Auto") << "): "
       << "Problem=[M:" << M << ", N:" << N << ", K:" << K << "] "
       << "-> Tile=[Tm:" << computedSizes[0] 
       << ", Tn:" << computedSizes[1] 
       << ", Tk:" << computedSizes[2] << "]\n";
    llvm::errs() << os.str();

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