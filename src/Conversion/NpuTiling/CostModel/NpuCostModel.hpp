// ========================================================================
// src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp
// This file defines the NPUCostModel class, which provides methods to calculate
// optimal tile sizes for different types of operations (Matmul, Conv2D,
// Elementwise based on the NPU's hardware constraints like SRAM size and number
// of MACs).
// ========================================================================
#pragma once
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include "llvm/ADT/SmallVector.h"
#include <cmath>
#include <vector>

namespace npux {

// 对应 Python 中的 HardwareConfig
struct HardwareConfig {
    int64_t spmSizeBytes=512 * 1024;
    int64_t accSizeBytes=256 * 1024;
    
    int64_t dtypeInput = 1; // int8
    int64_t dtypeAcc = 4;   // int32
    
    int64_t sysArraySize = 32;
    double measuredBandwidthMbps = 1300.0;
    
    double measuredConvUs = 5.0;
    double instrOverheadUs = 1.54;
};

// 内部用于承载从 MLIR 提取出的卷积参数 (对应 LayerParams)
struct ConvParams {
    int64_t H, W, IC, OC;
    int64_t K_h, K_w;
    int64_t S_h, S_w;
    int64_t D_h, D_w;
    // Linalg 通常 Padding 已经在张量维度或显式 PadOp 中处理，
    // 若模型中存在 explicitly padding，需在此扩展。
    
    int64_t getK_eff_h() const { return (K_h - 1) * D_h + 1; }
    int64_t getK_eff_w() const { return (K_w - 1) * D_w + 1; }
    // 假设输入 H, W 已经是 padding 后的完整维度
    int64_t getOH() const { return (H - getK_eff_h()) / S_h + 1; }
    int64_t getOW() const { return (W - getK_eff_w()) / S_w + 1; }
};

// 对应 Python 中的 TileResult
struct TileResult {
    int64_t t_oh = 0;
    int64_t t_ow = 0;
    int64_t t_ic = 0;
    int64_t t_oc = 0;
    double latencyMs = std::numeric_limits<double>::max();
    double spmUtil = 0.0;
    double accUtil = 0.0;
    bool valid = false;
};

class NPUCostModel {
private:
    HardwareConfig hw;

    llvm::SmallVector<int64_t> getConv2DTileSizes(npucore::ConvOp op);
    llvm::SmallVector<int64_t> getGemmTileSizes(npucore::MatMulOp op);
    llvm::SmallVector<int64_t> getGeluTileSizes(npucore::GeluOp op);
    llvm::SmallVector<int64_t> getSoftmaxTileSizes(npucore::SoftmaxOp op);
    llvm::SmallVector<int64_t> getLayerNormTileSizes(npucore::LayerNormOp op);
    llvm::SmallVector<int64_t> getMatAddTileSizes(npucore::MatAddOp op);
    llvm::SmallVector<int64_t> getTransposeTileSizes(npucore::TransposeOp op);
    llvm::SmallVector<int64_t> getLayoutPackTileSizes(
        npucore::LayoutNchwToNchwc32Op op);
    llvm::SmallVector<int64_t> getLayoutUnpackTileSizes(
        npucore::LayoutNchwc32ToNchwOp op);
    TileResult evaluateConvTile(const ConvParams& layer, int64_t t_oh, int64_t t_ow, int64_t t_oc, int64_t t_ic);
    std::vector<int64_t> getSafeRange(int64_t limit, int64_t step, int64_t maxVal = 64);
    llvm::SmallVector<int64_t> getMaxPoolTileSizes(npucore::MaxPoolOp op);


public:
    NPUCostModel(const HardwareConfig& config) : hw(config) {}

    llvm::SmallVector<int64_t> getOptimalTileSizes(npucore::ConvOp op);
    llvm::SmallVector<int64_t> getOptimalTileSizes(npucore::MatMulOp op);
    llvm::SmallVector<int64_t> getOptimalTileSizes(npucore::GeluOp op);
    llvm::SmallVector<int64_t> getOptimalTileSizes(npucore::SoftmaxOp op);
    llvm::SmallVector<int64_t> getOptimalTileSizes(npucore::LayerNormOp op);
    llvm::SmallVector<int64_t> getOptimalTileSizes(npucore::MatAddOp op);
    llvm::SmallVector<int64_t> getOptimalTileSizes(npucore::MaxPoolOp op);
    llvm::SmallVector<int64_t> getOptimalTileSizes(npucore::TransposeOp op);
    llvm::SmallVector<int64_t> getOptimalTileSizes(
        npucore::LayoutNchwToNchwc32Op op);
    llvm::SmallVector<int64_t> getOptimalTileSizes(
        npucore::LayoutNchwc32ToNchwOp op);
};

} // namespace npux
