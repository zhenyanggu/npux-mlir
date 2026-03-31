//======================================================
// src/Conversion/NpuTiling/CostModel/Conv.cpp
// This file implements the cost model for Conv2D operations,
// providing a method to calculate optimal tile sizes based on
// the NPU's hardware constraints (like SRAM size and number of MACs).
//======================================================

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
using namespace mlir;

namespace npux {

static void extractIntArrayAttr(
    Operation *op, llvm::StringRef attrName, int64_t &h, int64_t &w) {
  if (auto attr = op->getAttrOfType<ArrayAttr>(attrName)) {
    if (attr.size() >= 2) {
      h = cast<IntegerAttr>(attr[0]).getInt();
      w = cast<IntegerAttr>(attr[1]).getInt();
      return;
    }
  } else if (auto denseAttr = op->getAttrOfType<DenseI64ArrayAttr>(attrName)) {
    if (denseAttr.size() >= 2) {
      h = denseAttr[0];
      w = denseAttr[1];
      return;
    }
  }
  h = 1;
  w = 1; // 默认值
}

std::vector<int64_t> NPUCostModel::getSafeRange(
    int64_t limit, int64_t step, int64_t maxVal) {
  int64_t end = std::min(limit, maxVal);
  if (end <= 0)
    return {};
  if (end < step)
    return {end};

  std::vector<int64_t> candidates;
  for (int64_t i = step; i <= end; i += step) {
    candidates.push_back(i);
  }
  if (std::find(candidates.begin(), candidates.end(), end) ==
      candidates.end()) {
    candidates.push_back(end);
  }
  return candidates;
}

// 评估单组配置 (对应 Python CostModel::evaluate)
TileResult NPUCostModel::evaluateConvTile(const ConvParams &layer, int64_t t_oh,
    int64_t t_ow, int64_t t_oc, int64_t t_ic) {
  TileResult res;
  res.t_oh = t_oh;
  res.t_ow = t_ow;
  res.t_oc = t_oc;
  res.t_ic = t_ic;

  // 1. 空间需求计算
  int64_t raw_t_ih = (t_oh - 1) * layer.S_h + layer.getK_eff_h();
  int64_t raw_t_iw = (t_ow - 1) * layer.S_w + layer.getK_eff_w();

  int64_t sizeIfmTile = raw_t_ih * raw_t_iw * t_ic * hw.dtypeInput;
  int64_t sizeWgtTile = layer.K_h * layer.K_w * t_ic * t_oc * hw.dtypeInput;
  int64_t sizeOfmTile = t_oh * t_ow * t_oc * hw.dtypeInput;

  int64_t totalSpmNeeded = sizeIfmTile + sizeWgtTile + sizeOfmTile;
  int64_t sizeAccNeeded = t_oh * t_ow * t_oc * hw.dtypeAcc;

  // 内存溢出检查
  if (totalSpmNeeded > hw.spmSizeBytes || sizeAccNeeded > hw.accSizeBytes) {
    return res; // valid 默认为 false
  }

  // 2. 性能计算 (使用整数除法向上取整: (a + b - 1) / b )
  int64_t n_cout = (layer.OC + t_oc - 1) / t_oc;
  int64_t n_h = (layer.getOH() + t_oh - 1) / t_oh;
  int64_t n_w = (layer.getOW() + t_ow - 1) / t_ow;
  int64_t n_cin = (layer.IC + t_ic - 1) / t_ic;

  int64_t totalGlobalTiles = n_cout * n_h * n_w * n_cin;

  // DMA 数据量计算
  int64_t totalDmaIfmBytes = totalGlobalTiles * sizeIfmTile;
  int64_t totalDmaWgtBytes = totalGlobalTiles * sizeWgtTile;
  int64_t totalDmaBiasBytes = totalGlobalTiles * (t_oc * hw.dtypeAcc);
  int64_t totalDmaOfmBytes = (n_cout * n_h * n_w) * sizeOfmTile;

  // 计算延迟
  int64_t numSpatialMicroOps =
      (t_oh * t_ow + hw.sysArraySize - 1) / hw.sysArraySize;
  int64_t numCoutMicroOps = (t_oc + hw.sysArraySize - 1) / hw.sysArraySize;
  int64_t numCinMicroOps = (t_ic + hw.sysArraySize - 1) / hw.sysArraySize;

  int64_t opsPerConvTile =
      numCoutMicroOps * numSpatialMicroOps * numCinMicroOps;
  int64_t totalAtomicOps = totalGlobalTiles * opsPerConvTile;

  int64_t totalTrafficBytes = totalDmaIfmBytes + totalDmaWgtBytes +
                              totalDmaBiasBytes + totalDmaOfmBytes;
  int64_t numDmaInstructions = (totalGlobalTiles * 3) + (n_cout * n_h * n_w);

  double latDmaUs =
      (numDmaInstructions * hw.instrOverheadUs) +
      (static_cast<double>(totalTrafficBytes) / hw.measuredBandwidthMbps);
  double latComputeUs = totalAtomicOps * hw.measuredConvUs;

  res.latencyMs = (latDmaUs + latComputeUs) / 1000.0;
  res.spmUtil = static_cast<double>(totalSpmNeeded) / hw.spmSizeBytes;
  res.accUtil = static_cast<double>(sizeAccNeeded) / hw.accSizeBytes;
  res.valid = true;

  return res;
}

llvm::SmallVector<int64_t> NPUCostModel::getConv2DTileSizes(
    linalg::GenericOp op) {
  ConvParams params;

  // 1. 提取 Blocked 格式的张量形状
  auto inType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
  auto filterType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
  auto outType = dyn_cast<RankedTensorType>(op.getOutputs()[0].getType());

  if (!inType || !filterType || !outType || outType.getRank() < 5) {
    return {}; // 格式不符合预期
  }

  auto inShape = inType.getShape();
  auto filterShape = filterType.getShape();

  // 根据 tensor<1x16x30x30x32xi8> 提取 [N, IC_outer, H, W, IC_inner]
  params.H = inShape[2];
  params.W = inShape[3];
  params.IC = inShape[1] * inShape[4]; // IC = IC_outer * IC_inner

  // 根据 tensor<16x16x3x3x32x32xi8> 提取 [OC_outer, IC_outer, KH, KW, IC_inner,
  // OC_inner]
  params.K_h = filterShape[2];
  params.K_w = filterShape[3];
  params.OC = filterShape[0] * filterShape[5]; // OC = OC_outer * OC_inner

  // 提取步长和空洞 (strides, dilations)
  extractIntArrayAttr(op, "strides", params.S_h, params.S_w);
  extractIntArrayAttr(op, "dilations", params.D_h, params.D_w);

  // 2. 运行 Cost-Model 搜索逻辑 (复用之前写好的 getSafeRange 和
  // evaluateConvTile)
  auto r_oh = getSafeRange(params.getOH(), 4, 64);
  auto r_ow = getSafeRange(params.getOW(), 4, 64);

  int64_t start_ic = (params.IC >= 32) ? 32 : params.IC;
  int64_t start_oc = (params.OC >= 32) ? 32 : params.OC;

  TileResult bestRes;

  for (int64_t t_ic = start_ic; t_ic <= std::min<int64_t>(params.IC, 256);
      t_ic += 32) {
    for (int64_t t_oc = start_oc; t_oc <= std::min<int64_t>(params.OC, 256);
        t_oc += 32) {
      for (int64_t t_oh : r_oh) {
        for (int64_t t_ow : r_ow) {
          TileResult res = evaluateConvTile(params, t_oh, t_ow, t_oc, t_ic);
          if (res.valid && res.latencyMs < bestRes.latencyMs) {
            bestRes = res;
          }
        }
      }
    }
  }

  // 3. 按照你原有的业务逻辑组装输出 (映射到 NCHWc32)
  unsigned rank = outType.getRank();
  llvm::SmallVector<int64_t> sizes(rank, 0);

  if (bestRes.valid) {
    int64_t t_oh = bestRes.t_oh;
    int64_t t_ow = bestRes.t_ow;
    int64_t t_ic = bestRes.t_ic;
    int64_t t_oc = bestRes.t_oc;

    // 完美还原你提供的映射规则：[N, OC_outer, OH, OW, IC_outer]
    sizes[0] = 1;                             // N 维度不分块
    sizes[1] = (t_oc > 32) ? (t_oc / 32) : 1; // OC (对齐 32)
    sizes[2] = t_oh;                          // OH
    sizes[3] = t_ow;                          // OW
    sizes[4] = (t_ic > 32) ? (t_ic / 32) : 1; // IC (对齐 32)

    llvm::errs() << "[CostModel] Conv: Best Tile=[OH:" << t_oh << ", OW:" << t_ow
               << ", IC_blk:" << t_ic << ", OC_blk:" << t_oc
               << "], Latency: " << bestRes.latencyMs << "ms\n";
  }

  

  return sizes;
}

} // namespace npux