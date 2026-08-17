// SPDX-License-Identifier: Apache-2.0

#include "src/Conversion/NpuToLLVM/VersaPStaticEligibility.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace npux::versap {
namespace {

bool isAligned(uint32_t value) { return value % kLocalWordBytes == 0; }

bool hasAlignedDma(const TileRequest &tile) {
  if (!tile.aDma || !tile.wDma || !tile.outputDma)
    return false;
  return isAligned(tile.aDma->dramBase) &&
         isAligned(tile.aDma->dramRowStrideBytes) &&
         isAligned(tile.wDma->dramBase) &&
         (tile.wDma->dramSpanBytes == 0 || isAligned(tile.wDma->dramSpanBytes)) &&
         isAligned(tile.outputDma->dramBase) &&
         (tile.outputDma->dramRowStrideBytes == 0 ||
             isAligned(tile.outputDma->dramRowStrideBytes));
}

} // namespace

VersaPEligibilityDecision evaluateStaticGemmEligibility(
    const StaticGemmCandidate &candidate) {
  if (!candidate.descriptorAbiEnabled)
    return {false, VersaPFallbackRule::AbiDisabled};
  if (!candidate.isGemm)
    return {false, VersaPFallbackRule::UnsupportedOp};
  if (!candidate.hasStaticShapeAndOffset)
    return {false, VersaPFallbackRule::DynamicShapeOrOffset};
  if (!candidate.isCompleteSubgraph)
    return {false, VersaPFallbackRule::IncompleteSubgraph};
  if (!hasAlignedDma(candidate.tile))
    return {false, VersaPFallbackRule::DmaAlignment};
  VersaPBankScheduler scheduler;
  if (!scheduler.scheduleGemmTile(candidate.tile))
    return {false, VersaPFallbackRule::LocalBankCapacity};
  return {true, VersaPFallbackRule::None};
}


namespace {

std::optional<int64_t> constantInt(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return integer.getInt();
  return std::nullopt;
}

std::optional<uint16_t> positiveU16(mlir::Value value) {
  auto integer = constantInt(value);
  if (!integer || *integer <= 0 || *integer > UINT16_MAX)
    return std::nullopt;
  return static_cast<uint16_t>(*integer);
}

std::optional<uint32_t> alignedU32(mlir::Value value) {
  auto integer = constantInt(value);
  if (!integer || *integer < 0 || *integer > UINT32_MAX ||
      static_cast<uint32_t>(*integer) % kLocalWordBytes != 0)
    return std::nullopt;
  return static_cast<uint32_t>(*integer);
}

std::optional<uint16_t> localWordOffset(mlir::Value value) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  auto offset = definition->getAttrOfType<mlir::IntegerAttr>("npu.offset");
  if (!offset || offset.getInt() < 0 ||
      static_cast<uint64_t>(offset.getInt()) % kLocalWordBytes != 0)
    return std::nullopt;
  uint64_t words = static_cast<uint64_t>(offset.getInt()) / kLocalWordBytes;
  if (words >= kLocalBankWords)
    return std::nullopt;
  return static_cast<uint16_t>(words);
}

} // namespace

StaticGemmCandidate buildStaticRawInt32GemmCandidate(
    GemmDmaRegion region, bool descriptorAbiEnabled) {
  StaticGemmCandidate candidate;
  candidate.descriptorAbiEnabled = descriptorAbiEnabled;
  auto gemm = region.compute;
  candidate.isGemm = gemm.getOpType() == ComputeOpType::gemm;
  candidate.isCompleteSubgraph = candidate.isGemm;
  if (!candidate.isGemm || gemm.getPsumMemref())
    return candidate;

  auto m = positiveU16(region.loadA.getRowNum());
  auto k = positiveU16(region.loadA.getColNum());
  auto n = positiveU16(region.loadW.getRowNum());
  auto wK = positiveU16(region.loadW.getColNum());
  auto outRows = positiveU16(region.storeO.getRowNum());
  auto outBytes = positiveU16(region.storeO.getColNum());
  auto aStride = alignedU32(region.loadA.getDramStride());
  auto outStride = alignedU32(region.storeO.getDramStride());
  auto aBase = localWordOffset(gemm.getInputA());
  auto wBase = localWordOffset(gemm.getInputB());
  auto oBase = localWordOffset(gemm.getOutput());
  auto accumulate = constantInt(gemm.getIsAccumulate());
  auto bias = constantInt(gemm.getAccBias());
  auto relu = constantInt(gemm.getReluEnable());
  if (!m || !n || !k || !wK || !outRows || !outBytes || !aStride ||
      !outStride || !aBase || !wBase || !oBase || !accumulate || !bias ||
      !relu || *wK != *k || *outRows != *m ||
      *outBytes != static_cast<uint32_t>(*n) * 4 || *accumulate != 0 ||
      *bias != 0 || (*relu != 0 && *relu != 1))
    return candidate;

  candidate.hasStaticShapeAndOffset = true;
  candidate.tile = TileRequest{*m, *n, *k};
  candidate.tile.relu = *relu != 0;
  candidate.tile.quantization.requireRawInt32 = true;
  candidate.tile.aDma = MvinAConfig{0, *aStride, *m, *k, false, 0};
  const uint64_t wSpan =
      static_cast<uint64_t>(*n) * ((*k + kLocalWordBytes - 1) / kLocalWordBytes) *
      kLocalWordBytes;
  if (wSpan > UINT32_MAX)
    return StaticGemmCandidate{};
  candidate.tile.wDma =
      MvinWConfig{0, static_cast<uint32_t>(wSpan), *n, *k, 0};
  candidate.tile.outputDma =
      MvoutConfig{0, *outStride, *m, *outBytes, 0};
  candidate.tile.localAddresses = GemmLocalAddresses{*aBase, *wBase, *oBase};
  return candidate;
}
const char *fallbackRuleName(VersaPFallbackRule rule) {
  switch (rule) {
  case VersaPFallbackRule::None: return "none";
  case VersaPFallbackRule::AbiDisabled: return "abi-disabled";
  case VersaPFallbackRule::UnsupportedOp: return "unsupported-op";
  case VersaPFallbackRule::DynamicShapeOrOffset: return "dynamic-shape-or-offset";
  case VersaPFallbackRule::IncompleteSubgraph: return "incomplete-subgraph";
  case VersaPFallbackRule::DmaAlignment: return "dma-alignment";
  case VersaPFallbackRule::LocalBankCapacity: return "local-bank-capacity";
  }
  return "unknown";
}

} // namespace npux::versap