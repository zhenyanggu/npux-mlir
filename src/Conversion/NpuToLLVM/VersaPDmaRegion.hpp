// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "src/Dialect/Npux/NpuxOps.hpp"
#include "llvm/Support/Error.h"

namespace npux::versap {

// The G2 extraction boundary is a complete, ordered DMA--GEMM--DMA region.
// Host pointers remain runtime values and must be patched by the eventual
// runtime submission ABI; they are intentionally not converted to uint32_t
// descriptor bases here.
struct GemmDmaRegion {
  DmaMvinOp loadA;
  DmaMvinOp loadW;
  MvinBiasOp loadBias;
  MvinScaleOp loadScale;
  MvinMetadataOp loadMetadata;
  ResaddLoadOp loadResidual;
  ComputeRunOp compute;
  DmaMvoutOp storeO;
};

struct GemvDmaRegion {
  DmaMvinOp loadA;
  DmaMvinOp loadW;
  DmaMvinOp loadMetadata;
  GemvRunOp compute;
  DmaMvoutOp storeO;
};

struct VpuDmaRegion {
  VpuRunOp compute;
  DmaMvoutOp storeO;
};

llvm::Expected<GemmDmaRegion> extractGemmDmaRegion(ComputeRunOp compute,
    bool requireOutputDma = true);
llvm::Expected<GemvDmaRegion> extractGemvDmaRegion(GemvRunOp compute);
llvm::Expected<VpuDmaRegion> extractVpuDmaRegion(VpuRunOp compute);

} // namespace npux::versap
