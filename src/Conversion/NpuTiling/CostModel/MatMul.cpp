//======================================================
// src/Conversion/NpuTiling/CostModel/Gemm.cpp
// This file implements the cost model for GEMM/MatMul operations,
// providing a method to calculate optimal tile sizes based on
// the NPU's hardware constraints.
//======================================================

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "src/Compiler/NpuConfig.hpp"
#include <algorithm>

using namespace mlir;

namespace npux {

static SmallVector<int64_t, 3> calculateAutoGemmTile(
    int64_t M, int64_t N, int64_t K, int64_t spmSize, int64_t accSize) {

  const int64_t arraySizeH = 32;
  const int64_t arraySizeW = 32;
  const int64_t inputDtypeBytes = 1; 
  const int64_t outputDtypeBytes = 1;
  const int64_t accDtypeBytes = 4;   

  auto get_valid_tile_size = [&](int64_t val) -> int64_t {
    if (val < 32) return val;
    return (val / 32) * 32;
  };

  int64_t m_aligned = get_valid_tile_size(M);
  int64_t n_aligned = get_valid_tile_size(N);

  // ==========================================
  // Step 1: 最大化 Tk
  // ==========================================
  int64_t min_tm = std::min<int64_t>(M, arraySizeH);
  int64_t min_tn = std::min<int64_t>(N, arraySizeW);
  int64_t base_out_spm = min_tm * min_tn * outputDtypeBytes;
  
  int64_t max_tk_spm = 1;
  if (spmSize > base_out_spm) {
    max_tk_spm = (spmSize - base_out_spm) / ((min_tm + min_tn) * inputDtypeBytes);
  }
  int64_t t_k = std::max<int64_t>(1, std::min(K, max_tk_spm));

  // ==========================================
  // Step 2: 在固定 Tk 的前提下，最大化 Tm
  // ==========================================
  int64_t max_tm_acc = accSize / (min_tn * accDtypeBytes);
  
  int64_t max_tm_spm = m_aligned; 
  int64_t spm_rem_for_m = spmSize - min_tn * t_k * inputDtypeBytes; 
  if (spm_rem_for_m > 0) {
    max_tm_spm = spm_rem_for_m / (min_tn * outputDtypeBytes + t_k * inputDtypeBytes);
  }
  
  int64_t t_m = std::min({m_aligned, max_tm_acc, max_tm_spm});
  t_m = get_valid_tile_size(t_m);

  // ==========================================
  // Step 3: 在固定 Tk 和 Tm 的前提下，计算剩余的 Tn
  // ==========================================
  int64_t max_tn_acc = accSize / (t_m * accDtypeBytes);
  
  int64_t max_tn_spm = n_aligned;
  int64_t spm_rem_for_n = spmSize - t_k * t_m * inputDtypeBytes;
  if (spm_rem_for_n > 0) {
    max_tn_spm = spm_rem_for_n / (t_m * outputDtypeBytes + t_k * inputDtypeBytes);
  }

  int64_t t_n = std::min({n_aligned, max_tn_acc, max_tn_spm});
  t_n = get_valid_tile_size(t_n);

  return {t_m, t_n, t_k};
}

llvm::SmallVector<int64_t> NPUCostModel::getGemmTileSizes(
    npucore::MatMulOp op) {
  auto outputType =
      dyn_cast<RankedTensorType>(op.getOutputs().front().getType());
  if (!outputType)
    return {};
  int64_t outRank = outputType.getRank();
  if (outRank != 2 && outRank != 3) {
    return {};
  }

  auto lhsType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
  if (!lhsType)
    return {};

  int64_t rank = outRank == 2 ? 3 : 4;
  auto &config = npux::NPUConfig::getInstance();
  std::vector<int64_t> manualSizes = config.getMatMulTileSize();
  if (!manualSizes.empty() && manualSizes.size() >= 3) {
    SmallVector<int64_t> finalTileSizes(rank, 0);
    int64_t tm = manualSizes[0];
    int64_t tn = manualSizes[1];
    int64_t tk = manualSizes[2];

    if (rank == 4) {
      finalTileSizes[0] = 1;
      finalTileSizes[1] = tn;
      finalTileSizes[2] = tm;
      finalTileSizes[3] = tk;
    } else {
      finalTileSizes[0] = tn;
      finalTileSizes[1] = tm;
      finalTileSizes[2] = tk;
    }
    return finalTileSizes;
  }

  int64_t M = outputType.getShape()[outRank - 2];
  int64_t N = outputType.getShape()[outRank - 1];
  int64_t K = lhsType.getShape()[lhsType.getRank() - 1];
  if (ShapedType::isDynamic(M) || ShapedType::isDynamic(N) ||
      ShapedType::isDynamic(K))
    return {};

  int64_t spmSize = this->hw.spmSizeBytes;
  int64_t accSize = this->hw.accSizeBytes;

  SmallVector<int64_t, 3> computedSizes = calculateAutoGemmTile(M, N, K, spmSize, accSize);

  llvm::errs() << "[CostModel] Gemm: Best Tile=[Tm:" << computedSizes[0] 
               << ", Tn:" << computedSizes[1]
               << ", Tk:" << computedSizes[2] << "]\n";

  SmallVector<int64_t> finalTileSizes(rank, 0);

  // 按照 Linalg Generic Iterator 的顺序填充分块大小
  if (rank == 4) {
    finalTileSizes[0] = 1;                // Batch 永远按 1 分块
    finalTileSizes[1] = computedSizes[1]; // N -> tn
    finalTileSizes[2] = computedSizes[0]; // M -> tm
    finalTileSizes[3] = computedSizes[2]; // K -> tk
  } else {
    finalTileSizes[0] = computedSizes[1]; // N -> tn
    finalTileSizes[1] = computedSizes[0]; // M -> tm
    finalTileSizes[2] = computedSizes[2]; // K -> tk
  }

  return finalTileSizes;
}

} // namespace npux
