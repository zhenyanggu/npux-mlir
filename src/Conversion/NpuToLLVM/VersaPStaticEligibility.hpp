// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "src/Conversion/NpuToLLVM/VersaPBankScheduler.hpp"
#include "src/Conversion/NpuToLLVM/VersaPDmaRegion.hpp"

namespace npux::versap {

enum class VersaPFallbackRule {
  None,
  AbiDisabled,
  UnsupportedOp,
  DynamicShapeOrOffset,
  IncompleteSubgraph,
  DmaAlignment,
  LocalBankCapacity,
};

struct StaticGemmCandidate {
  bool descriptorAbiEnabled = false;
  bool isGemm = false;
  bool hasStaticShapeAndOffset = false;
  bool isCompleteSubgraph = false;
  TileRequest tile;
};

struct VersaPEligibilityDecision {
  bool eligible = false;
  VersaPFallbackRule fallback = VersaPFallbackRule::None;
};

VersaPEligibilityDecision evaluateStaticGemmEligibility(
    const StaticGemmCandidate &candidate);
const char *fallbackRuleName(VersaPFallbackRule rule);

// Builds the first lowering slice: a static, raw-INT32 DMA(A)/DMA(W)/GEMM/
// DMA(O) region. Host addresses stay dynamic and are patched by runtime.
StaticGemmCandidate buildStaticRawInt32GemmCandidate(
    GemmDmaRegion region, bool descriptorAbiEnabled);

} // namespace npux::versap