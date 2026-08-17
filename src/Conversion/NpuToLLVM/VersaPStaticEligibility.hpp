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

struct StaticGemvCandidate {
  bool descriptorAbiEnabled = false;
  bool hasStaticShapeAndOffset = false;
  bool isCompleteSubgraph = false;
  GemvRequest request;
};

struct StaticVpuCandidate {
  bool descriptorAbiEnabled = false;
  bool hasStaticShapeAndOffset = false;
  bool isCompleteSubgraph = false;
  VpuRequest request;
};

VersaPEligibilityDecision evaluateStaticGemmEligibility(
    const StaticGemmCandidate &candidate);
const char *fallbackRuleName(VersaPFallbackRule rule);

// Builds the first lowering slice: a static, raw-INT32 DMA(A)/DMA(W)/GEMM/
// DMA(O) region. Host addresses stay dynamic and are patched by runtime.
StaticGemmCandidate buildStaticRawInt32GemmCandidate(
    GemmDmaRegion region, bool descriptorAbiEnabled);

StaticGemmCandidate buildStaticConvCandidate(
    GemmDmaRegion region, bool descriptorAbiEnabled);

// Builds a statically shaped QK or PV DMA--SA--DMA slice. The caller must
// pair a QK and PV candidate belonging to one attention head before emitting
// their descriptor DAG.
StaticGemmCandidate buildStaticAttentionCandidate(
    GemmDmaRegion region, SaOperation operation, bool descriptorAbiEnabled);
StaticGemvCandidate buildStaticGemvCandidate(
    GemvDmaRegion region, bool descriptorAbiEnabled);
VersaPEligibilityDecision evaluateStaticGemvEligibility(
    const StaticGemvCandidate &candidate);
StaticVpuCandidate buildStaticVpuCandidate(
    VpuDmaRegion region, bool descriptorAbiEnabled);

} // namespace npux::versap
