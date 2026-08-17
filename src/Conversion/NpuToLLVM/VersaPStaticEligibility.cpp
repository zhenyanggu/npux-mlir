// SPDX-License-Identifier: Apache-2.0

#include "src/Conversion/NpuToLLVM/VersaPStaticEligibility.hpp"

#include <algorithm>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

namespace npux::versap {
namespace {

bool isAligned(uint32_t value) { return value % kLocalWordBytes == 0; }

bool hasAlignedDma(const TileRequest &tile) {
  if (!tile.aDma || !tile.wDma ||
      (tile.storeToDram && !tile.outputDma))
    return false;
  const bool outputIsAligned = !tile.outputDma ||
      (isAligned(tile.outputDma->dramBase) &&
          (tile.outputDma->dramRowStrideBytes == 0 ||
              isAligned(tile.outputDma->dramRowStrideBytes)));
  return isAligned(tile.aDma->dramBase) &&
         isAligned(tile.aDma->dramRowStrideBytes) &&
         isAligned(tile.wDma->dramBase) &&
         (tile.wDma->dramSpanBytes == 0 || isAligned(tile.wDma->dramSpanBytes)) &&
         outputIsAligned;
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
  if (candidate.tile.operation == SaOperation::Conv
          ? !scheduler.scheduleConvTile(candidate.tile)
          : !scheduler.scheduleGemmTile(candidate.tile))
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

VersaPEligibilityDecision evaluateStaticGemvEligibility(
    const StaticGemvCandidate &candidate) {
  if (!candidate.descriptorAbiEnabled)
    return {false, VersaPFallbackRule::AbiDisabled};
  if (!candidate.isCompleteSubgraph)
    return {false, VersaPFallbackRule::IncompleteSubgraph};
  if (!candidate.hasStaticShapeAndOffset)
    return {false, VersaPFallbackRule::DynamicShapeOrOffset};
  VersaPBankScheduler scheduler;
  if (!scheduler.scheduleGemv(candidate.request))
    return {false, VersaPFallbackRule::LocalBankCapacity};
  return {true, VersaPFallbackRule::None};
}

std::optional<uint16_t> nonnegativeU16(mlir::Value value) {
  auto integer = constantInt(value);
  if (!integer || *integer < 0 || *integer >= UINT16_MAX)
    return std::nullopt;
  return static_cast<uint16_t>(*integer);
}

std::optional<uint16_t> tiledChannels(mlir::Value value, unsigned outer,
    unsigned inner) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || type.getRank() <= std::max(outer, inner) ||
      type.isDynamicDim(outer) || type.isDynamicDim(inner))
    return std::nullopt;
  const int64_t channels = type.getDimSize(outer) * type.getDimSize(inner);
  if (channels <= 0 || channels > UINT16_MAX)
    return std::nullopt;
  return static_cast<uint16_t>(channels);
}

std::optional<uint8_t> u8Value(mlir::Value value, uint8_t maximum) {
  auto integer = constantInt(value);
  if (!integer || *integer < 0 || *integer > maximum)
    return std::nullopt;
  return static_cast<uint8_t>(*integer);
}

std::optional<uint32_t> staticByteCount(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || !type.hasStaticShape())
    return std::nullopt;
  uint64_t elements = 1;
  for (int64_t dimension : type.getShape()) {
    if (dimension <= 0)
      return std::nullopt;
    elements *= static_cast<uint64_t>(dimension);
  }
  const uint64_t bytes = elements *
      static_cast<uint64_t>((type.getElementTypeBitWidth() + 7) / 8);
  if (bytes == 0 || bytes > UINT32_MAX)
    return std::nullopt;
  return static_cast<uint32_t>(bytes);
}

} // namespace

StaticGemmCandidate buildStaticRawInt32GemmCandidate(
    GemmDmaRegion region, bool descriptorAbiEnabled) {
  StaticGemmCandidate candidate;
  candidate.descriptorAbiEnabled = descriptorAbiEnabled;
  auto gemm = region.compute;
  candidate.isGemm = gemm.getOpType() == ComputeOpType::gemm;
  candidate.isCompleteSubgraph = candidate.isGemm;
  if (!candidate.isGemm)
    return candidate;

  llvm::StringRef accStage = "single";
  if (auto stage = gemm->getAttrOfType<mlir::StringAttr>(
          "npux.versa_p_acc_stage"))
    accStage = stage.getValue();
  const bool firstAccSlice = accStage == "first";
  const bool bodyAccSlice = accStage == "body";
  const bool finalAccSlice = accStage == "final";
  if (accStage != "single" && !firstAccSlice && !bodyAccSlice &&
      !finalAccSlice)
    return candidate;
  const bool writePartial = firstAccSlice || bodyAccSlice;
  const bool needsAccumulate = bodyAccSlice || finalAccSlice;
  if ((needsAccumulate && !gemm.getPsumMemref()) ||
      (!needsAccumulate && gemm.getPsumMemref()) ||
      (writePartial && region.storeO) ||
      (finalAccSlice && !region.storeO))
    return candidate;

  auto m = positiveU16(region.loadA.getRowNum());
  auto k = positiveU16(region.loadA.getColNum());
  auto n = positiveU16(region.loadW.getRowNum());
  auto wK = positiveU16(region.loadW.getColNum());
  const bool hasOutputDma = static_cast<bool>(region.storeO);
  std::optional<uint16_t> outRows;
  std::optional<uint16_t> outBytes;
  if (hasOutputDma) {
    outRows = positiveU16(region.storeO.getRowNum());
    outBytes = positiveU16(region.storeO.getColNum());
  }
  auto aStride = alignedU32(region.loadA.getDramStride());
  std::optional<uint32_t> outStride;
  if (hasOutputDma)
    outStride = alignedU32(region.storeO.getDramStride());
  auto aBase = localWordOffset(gemm.getInputA());
  auto wBase = localWordOffset(gemm.getInputB());
  auto oBase = localWordOffset(gemm.getOutput());
  auto accumulatorBase = gemm.getPsumMemref()
      ? localWordOffset(gemm.getPsumMemref()) : oBase;
  auto outputType = mlir::dyn_cast<mlir::MemRefType>(gemm.getOutput().getType());
  auto inputAType = mlir::dyn_cast<mlir::MemRefType>(gemm.getInputA().getType());
  auto inputWType = mlir::dyn_cast<mlir::MemRefType>(gemm.getInputB().getType());
  const bool outputIsBf16 = outputType &&
      mlir::isa<mlir::BFloat16Type>(outputType.getElementType());
  const bool outputIsInt8 = outputType &&
      outputType.getElementType().isInteger(8);
  const bool inputsAreInt8 = inputAType && inputWType &&
      inputAType.getElementType().isInteger(8) &&
      inputWType.getElementType().isInteger(8);
  std::optional<uint32_t> tensorScaleQ8_24;
  if (auto scale = gemm->getAttrOfType<mlir::IntegerAttr>(
          "npux.versa_p_tensor_scale_q8_24")) {
    const int64_t value = scale.getInt();
    if (value >= 0 && value <= UINT32_MAX)
      tensorScaleQ8_24 = static_cast<uint32_t>(value);
  }
  auto accumulate = constantInt(gemm.getIsAccumulate());
  auto bias = constantInt(gemm.getAccBias());
  auto relu = constantInt(gemm.getReluEnable());
  std::optional<uint32_t> biasBytes;
  if (region.loadBias)
    biasBytes = staticByteCount(region.loadBias.getSource());
  std::optional<uint32_t> scaleBytes;
  if (region.loadScale)
    scaleBytes = staticByteCount(region.loadScale.getSource());
  std::optional<uint32_t> packedMetadataBytes;
  if (region.loadMetadata)
    packedMetadataBytes = staticByteCount(region.loadMetadata.getSource());
  std::optional<uint32_t> residualBytes;
  if (region.loadResidual)
    residualBytes = staticByteCount(region.loadResidual.getSource());
  const bool needsResadd = gemm->hasAttr("npux.versa_p_resadd");
  const bool hasPerChannelScale =
      gemm->hasAttr("npux.versa_p_per_channel_scale");
  const bool hasPackedMetadata =
      gemm->hasAttr("npux.versa_p_packed_metadata");
  const uint32_t perChannelWords = n ? (*n + 7) / 8 : 0;
  const uint64_t scaleMetadataBytes =
      static_cast<uint64_t>(perChannelWords) * kLocalWordBytes;
  const uint64_t packedMetadataSize = 2 * scaleMetadataBytes;
  if (!m || !n || !k || !wK || !aStride || !aBase || !wBase || !oBase ||
      !accumulatorBase ||
      !accumulate || !bias || !relu || *wK != *k ||
      ((outputIsBf16 || outputIsInt8) && !inputsAreInt8) ||
      (hasOutputDma && (!outRows || !outBytes || !outStride ||
          *outRows != *m || *outBytes !=
              static_cast<uint32_t>(*n) *
                  (outputIsInt8 ? 1 : outputIsBf16 ? 2 : 4))) ||
      (!writePartial && outputIsInt8 && !hasPerChannelScale &&
          (!tensorScaleQ8_24 || *tensorScaleQ8_24 > 511 * 16 ||
              (*tensorScaleQ8_24 % 16) != 0)) ||
      (hasPackedMetadata && (!hasPerChannelScale || !outputIsInt8 ||
          region.loadBias || region.loadScale || !region.loadMetadata ||
          !packedMetadataBytes || *packedMetadataBytes != packedMetadataSize)) ||
      (!hasPackedMetadata && region.loadMetadata) ||
      (hasPerChannelScale && !hasPackedMetadata &&
          (!outputIsInt8 || region.loadBias || !region.loadScale ||
              !scaleBytes || *scaleBytes != scaleMetadataBytes)) ||
      (!hasPerChannelScale && region.loadScale) ||
      (*bias != 0 && !hasPackedMetadata && (!region.loadBias || !biasBytes ||
          *biasBytes < static_cast<uint32_t>(*n) * 4)) ||
      (needsResadd && (!hasOutputDma || !region.loadResidual || !residualBytes ||
          *residualBytes != static_cast<uint32_t>(*m) * *n *
              (outputIsInt8 ? 1 : outputIsBf16 ? 2 : 4))) ||
      (*bias == 0 && *accumulate != (needsAccumulate ? 1 : 0)) ||
      (*bias != 0 && (bodyAccSlice || finalAccSlice || *accumulate != 1)) ||
      (*relu != 0 && *relu != 1))
    return candidate;

  candidate.hasStaticShapeAndOffset = true;
  candidate.tile = TileRequest{*m, *n, *k};
  candidate.tile.needsBias = *bias != 0;
  candidate.tile.needsAccumulate = needsAccumulate;
  candidate.tile.writePartial = writePartial;
  candidate.tile.needsResadd = needsResadd;
  candidate.tile.relu = *relu != 0;
  candidate.tile.quantization.requireRawInt32 = !outputIsBf16 && !outputIsInt8;
  candidate.tile.quantization.nextConsumesBf16 = outputIsBf16;
  candidate.tile.quantization.nextConsumesInt8 = outputIsInt8;
  candidate.tile.quantization.perChannelScaleAvailable = hasPerChannelScale;
  if (hasPerChannelScale) {
    const uint64_t metadataWords = hasPackedMetadata ?
        2 * static_cast<uint64_t>(perChannelWords) : perChannelWords;
    if (metadataWords > kMetadataWords)
      return StaticGemmCandidate{};
    candidate.tile.quantization.metadataWords =
        static_cast<uint16_t>(metadataWords);
    candidate.tile.metadataDma = AuxMvinConfig{0, 0,
        hasPackedMetadata ? *packedMetadataBytes : *scaleBytes, 0, false, 0};
  }
  if (writePartial) {
    candidate.tile.quantization.requireRawInt32 = true;
    candidate.tile.quantization.nextConsumesBf16 = false;
    candidate.tile.quantization.nextConsumesInt8 = false;
  }
  candidate.tile.aDma = MvinAConfig{0, *aStride, *m, *k, false, 0};
  const uint64_t wSpan =
      static_cast<uint64_t>(*n) * ((*k + kLocalWordBytes - 1) / kLocalWordBytes) *
      kLocalWordBytes;
  if (wSpan > UINT32_MAX)
    return StaticGemmCandidate{};
  candidate.tile.wDma =
      MvinWConfig{0, static_cast<uint32_t>(wSpan), *n, *k, 0};
  candidate.tile.storeToDram = hasOutputDma && !writePartial;
  if (hasOutputDma)
    candidate.tile.outputDma =
        MvoutConfig{0, *outStride, *m, *outBytes, 0};
  if (candidate.tile.needsBias && !hasPackedMetadata) {
    const uint64_t metadataWords =
        (*biasBytes + kLocalWordBytes - 1) / kLocalWordBytes;
    if (metadataWords > kMetadataWords)
      return StaticGemmCandidate{};
    candidate.tile.quantization.metadataWords =
        static_cast<uint16_t>(metadataWords);
    candidate.tile.metadataDma =
        AuxMvinConfig{0, 0, *biasBytes, 0, false, 0};
  }
  if (candidate.tile.needsResadd) {
    const uint32_t rowBytes = static_cast<uint32_t>(*n) *
        (outputIsInt8 ? 1 : outputIsBf16 ? 2 : 4);
    candidate.tile.residualDma = AuxMvinConfig{0, *outStride,
        static_cast<uint32_t>(*m) | (rowBytes << 16), 0, true, 0};
  }
  candidate.tile.localAddresses = GemmLocalAddresses{*aBase, *wBase, *oBase,
      *accumulatorBase, 0, 0, static_cast<uint16_t>(hasPerChannelScale
          ? (hasPackedMetadata ? perChannelWords : 0)
          : outputIsInt8 && tensorScaleQ8_24 ? *tensorScaleQ8_24 / 16 : 0)};
  return candidate;
}

StaticGemmCandidate buildStaticConvCandidate(
    GemmDmaRegion region, bool descriptorAbiEnabled) {
  StaticGemmCandidate candidate;
  candidate.descriptorAbiEnabled = descriptorAbiEnabled;
  auto conv = region.compute;
  candidate.isGemm = conv.getOpType() == ComputeOpType::conv;
  candidate.isCompleteSubgraph = candidate.isGemm && !conv.getPsumMemref();
  if (!candidate.isCompleteSubgraph)
    return candidate;

  auto ifmWidthM1 = nonnegativeU16(conv.getInputAColNumM1());
  auto ifmHeightM1 = nonnegativeU16(conv.getInputARowNumM1());
  auto outputWidth = positiveU16(conv.getBiaspsumWidth());
  auto outputHeight = positiveU16(conv.getBiaspsumHeight());
  auto kernelM1 = u8Value(conv.getWeightShapeM1(), 15);
  auto strideM1 = u8Value(conv.getWeightStrideM1(), 3);
  auto dilationM1 = u8Value(conv.getWeightDilationM1(), 31);
  auto padTop = u8Value(conv.getPadTop(), 3);
  auto padBottom = u8Value(conv.getPadBottom(), 3);
  auto padLeft = u8Value(conv.getPadLeft(), 3);
  auto padRight = u8Value(conv.getPadRight(), 3);
  auto inputChannels = tiledChannels(conv.getInputA(), 1, 4);
  auto outputChannels = tiledChannels(conv.getOutput(), 1, 4);
  const bool hasOutputDma = static_cast<bool>(region.storeO);
  auto aStride = alignedU32(region.loadA.getDramStride());
  std::optional<uint32_t> outStride;
  if (hasOutputDma)
    outStride = alignedU32(region.storeO.getDramStride());
  auto aRows = positiveU16(region.loadA.getRowNum());
  auto aColumns = positiveU16(region.loadA.getColNum());
  auto wRows = positiveU16(region.loadW.getRowNum());
  auto wColumns = positiveU16(region.loadW.getColNum());
  std::optional<uint16_t> outRows;
  std::optional<uint16_t> outColumns;
  if (hasOutputDma) {
    outRows = positiveU16(region.storeO.getRowNum());
    outColumns = positiveU16(region.storeO.getColNum());
  }
  auto aBase = localWordOffset(conv.getInputA());
  auto wBase = localWordOffset(conv.getInputB());
  auto oBase = localWordOffset(conv.getOutput());
  auto accumulate = constantInt(conv.getIsAccumulate());
  auto bias = constantInt(conv.getAccBias());
  auto relu = constantInt(conv.getReluEnable());
  auto isGroup = constantInt(conv.getIsGroupConv());
  std::optional<uint32_t> biasBytes;
  if (region.loadBias)
    biasBytes = staticByteCount(region.loadBias.getSource());
  std::optional<uint32_t> residualBytes;
  if (region.loadResidual)
    residualBytes = staticByteCount(region.loadResidual.getSource());
  const bool needsResadd = conv->hasAttr("npux.versa_p_resadd");
  const uint32_t kernelExtent =
      (static_cast<uint32_t>(kernelM1.value_or(0)) + 1) *
          (static_cast<uint32_t>(dilationM1.value_or(0)) + 1) -
      dilationM1.value_or(0);
  const int32_t outputHeightNumerator =
      static_cast<int32_t>(ifmHeightM1.value_or(0)) + 1 +
      padTop.value_or(0) + padBottom.value_or(0) -
      static_cast<int32_t>(kernelExtent);
  const int32_t outputWidthNumerator =
      static_cast<int32_t>(ifmWidthM1.value_or(0)) + 1 +
      padLeft.value_or(0) + padRight.value_or(0) -
      static_cast<int32_t>(kernelExtent);
  const uint32_t stride = static_cast<uint32_t>(strideM1.value_or(0)) + 1;
  if (!ifmWidthM1 || !ifmHeightM1 || !outputWidth || !outputHeight ||
      !kernelM1 || !strideM1 || !dilationM1 || !padTop || !padBottom ||
      !padLeft || !padRight || !inputChannels || !outputChannels || !aStride ||
      !aRows || !aColumns || !wRows || !wColumns || !aBase || !wBase ||
      !oBase || !accumulate || !bias ||
      !relu || !isGroup || *isGroup != 0 || *inputChannels > 32 ||
      outputHeightNumerator < 0 || outputWidthNumerator < 0 ||
      static_cast<uint32_t>(*outputHeight) !=
          static_cast<uint32_t>(outputHeightNumerator) / stride + 1 ||
      static_cast<uint32_t>(*outputWidth) !=
          static_cast<uint32_t>(outputWidthNumerator) / stride + 1 ||
      (hasOutputDma && (!outStride || !outRows || !outColumns ||
          static_cast<uint32_t>(*outRows) !=
              static_cast<uint32_t>(*outputHeight) * *outputWidth ||
          *outColumns != *outputChannels)) ||
      (*bias != 0 && (!region.loadBias || !biasBytes ||
          *biasBytes < static_cast<uint32_t>(*outputChannels) * 4)) ||
      (needsResadd && (!hasOutputDma || !region.loadResidual ||
          !residualBytes || *residualBytes !=
              static_cast<uint32_t>(*outputHeight) * *outputWidth *
                  *outputChannels)) ||
      (*bias == 0 && *accumulate != 0) ||
      (*bias != 0 && *accumulate != 1) ||
      (*relu != 0 && *relu != 1))
    return candidate;

  const uint64_t wSpan = static_cast<uint64_t>(*wRows) *
      ((*wColumns + kLocalWordBytes - 1) / kLocalWordBytes) *
      kLocalWordBytes;
  if (wSpan > UINT32_MAX)
    return StaticGemmCandidate{};
  candidate.hasStaticShapeAndOffset = true;
  candidate.tile = TileRequest{static_cast<uint16_t>(*ifmHeightM1 + 1),
      static_cast<uint16_t>(*ifmWidthM1 + 1), *inputChannels};
  candidate.tile.operation = SaOperation::Conv;
  candidate.tile.conv = ConvGeometry{*outputChannels, *kernelM1, *strideM1,
      *dilationM1, *padLeft, *padRight, *padTop, *padBottom};
  candidate.tile.needsBias = *bias != 0;
  candidate.tile.needsResadd = needsResadd;
  candidate.tile.relu = *relu != 0;
  candidate.tile.quantization.nextConsumesInt8 = true;
  candidate.tile.aDma = MvinAConfig{0, *aStride, *aRows, *aColumns, false, 0};
  candidate.tile.wDma = MvinWConfig{0, static_cast<uint32_t>(wSpan), *wRows,
      *wColumns, 0};
  candidate.tile.storeToDram = hasOutputDma;
  if (hasOutputDma)
    candidate.tile.outputDma =
        MvoutConfig{0, *outStride, *outRows, *outColumns, 0};
  if (candidate.tile.needsBias) {
    const uint64_t metadataWords =
        (*biasBytes + kLocalWordBytes - 1) / kLocalWordBytes;
    if (metadataWords > kMetadataWords)
      return StaticGemmCandidate{};
    candidate.tile.quantization.metadataWords =
        static_cast<uint16_t>(metadataWords);
    candidate.tile.metadataDma =
        AuxMvinConfig{0, 0, *biasBytes, 0, false, 0};
  }
  if (candidate.tile.needsResadd) {
    candidate.tile.residualDma = AuxMvinConfig{0, *outStride,
        static_cast<uint32_t>(*outRows) |
            (static_cast<uint32_t>(*outColumns) << 16),
        0, true, 0};
  }
  candidate.tile.localAddresses = GemmLocalAddresses{*aBase, *wBase, *oBase,
      0, 0};
  return candidate;
}

StaticGemmCandidate buildStaticAttentionCandidate(
    GemmDmaRegion region, SaOperation operation, bool descriptorAbiEnabled) {
  StaticGemmCandidate candidate;
  candidate.descriptorAbiEnabled = descriptorAbiEnabled;
  candidate.isGemm = region.compute.getOpType() == ComputeOpType::gemm;
  candidate.isCompleteSubgraph = candidate.isGemm &&
      (operation == SaOperation::AttentionQk ||
          operation == SaOperation::AttentionPv);
  if (!candidate.isCompleteSubgraph || region.compute.getPsumMemref())
    return candidate;

  auto m = positiveU16(region.loadA.getRowNum());
  auto k = positiveU16(region.loadA.getColNum());
  auto n = positiveU16(region.loadW.getRowNum());
  auto wK = positiveU16(region.loadW.getColNum());
  auto outRows = positiveU16(region.storeO.getRowNum());
  auto outBytes = positiveU16(region.storeO.getColNum());
  auto aStride = alignedU32(region.loadA.getDramStride());
  auto outStride = alignedU32(region.storeO.getDramStride());
  auto aBase = localWordOffset(region.compute.getInputA());
  auto wBase = localWordOffset(region.compute.getInputB());
  auto oBase = localWordOffset(region.compute.getOutput());
  auto accumulate = constantInt(region.compute.getIsAccumulate());
  auto bias = constantInt(region.compute.getAccBias());
  auto relu = constantInt(region.compute.getReluEnable());
  std::optional<uint32_t> qkGammaQ8_24;
  std::optional<uint32_t> qkGammaBytes;
  if (operation == SaOperation::AttentionQk) {
    auto gamma = region.compute->getAttrOfType<mlir::IntegerAttr>(
        "npux.versa_p_qk_gamma_q8_24");
    if (gamma && gamma.getInt() > 0 && gamma.getInt() < (int64_t{1} << 25))
      qkGammaQ8_24 = static_cast<uint32_t>(gamma.getInt());
    if (region.loadMetadata)
      qkGammaBytes = staticByteCount(region.loadMetadata.getSource());
  }
  if (!m || !n || !k || !wK || !outRows || !outBytes || !aStride ||
      !outStride || !aBase || !wBase || !oBase || !accumulate || !bias ||
      !relu || *wK != *k || *outRows != *m || *accumulate != 0 ||
      *bias != 0 || (*relu != 0 && *relu != 1) ||
      (operation == SaOperation::AttentionQk &&
          (!qkGammaQ8_24 || !qkGammaBytes || *qkGammaBytes < 4 ||
              *qkGammaBytes > kMetadataWords * kLocalWordBytes)))
    return candidate;
  const uint32_t expectedOutputBytes = operation == SaOperation::AttentionQk
      ? static_cast<uint32_t>(*n) * 4 : static_cast<uint32_t>(*n);
  if (*outBytes != expectedOutputBytes)
    return candidate;

  candidate.hasStaticShapeAndOffset = true;
  candidate.tile = TileRequest{*m, *n, *k};
  candidate.tile.operation = operation;
  candidate.tile.quantization.requireRawInt32 =
      operation == SaOperation::AttentionQk;
  candidate.tile.quantization.nextConsumesInt8 =
      operation == SaOperation::AttentionPv;
  if (operation == SaOperation::AttentionQk) {
    candidate.tile.qkGammaQ8_24 = *qkGammaQ8_24;
    candidate.tile.quantization.metadataWords = static_cast<uint16_t>(
        (*qkGammaBytes + kLocalWordBytes - 1) / kLocalWordBytes);
    candidate.tile.metadataDma =
        AuxMvinConfig{0, 0, *qkGammaBytes, 0, false, 0};
  }
  candidate.tile.aDma = MvinAConfig{0, *aStride, *m, *k, false, 0};
  const uint64_t wSpan = static_cast<uint64_t>(*n) *
      ((*k + kLocalWordBytes - 1) / kLocalWordBytes) * kLocalWordBytes;
  if (wSpan > UINT32_MAX)
    return StaticGemmCandidate{};
  candidate.tile.wDma =
      MvinWConfig{0, static_cast<uint32_t>(wSpan), *n, *k, 0};
  candidate.tile.outputDma = MvoutConfig{0, *outStride, *m, *outBytes, 0,
      0, operation == SaOperation::AttentionQk};
  // RTL QK reads scale_addr as a metadata word address. The gamma buffer's
  // first little-endian i32 is therefore placed at word zero.
  candidate.tile.localAddresses = GemmLocalAddresses{*aBase, *wBase, *oBase};
  return candidate;
}

StaticGemvCandidate buildStaticGemvCandidate(
    GemvDmaRegion region, bool descriptorAbiEnabled) {
  StaticGemvCandidate candidate;
  candidate.descriptorAbiEnabled = descriptorAbiEnabled;
  candidate.isCompleteSubgraph = true;
  auto m = positiveU16(region.compute.getM());
  auto k = positiveU16(region.compute.getK());
  auto aRows = positiveU16(region.loadA.getRowNum());
  auto aColumns = positiveU16(region.loadA.getColNum());
  auto wRows = positiveU16(region.loadW.getRowNum());
  auto wColumns = positiveU16(region.loadW.getColNum());
  auto metadataRows = positiveU16(region.loadMetadata.getRowNum());
  auto metadataColumns = positiveU16(region.loadMetadata.getColNum());
  std::optional<uint16_t> outputRows;
  std::optional<uint16_t> outputBytes;
  std::optional<uint32_t> outputStride;
  if (region.storeO) {
    outputRows = positiveU16(region.storeO.getRowNum());
    outputBytes = positiveU16(region.storeO.getColNum());
    outputStride = alignedU32(region.storeO.getDramStride());
  }
  auto aStride = alignedU32(region.loadA.getDramStride());
  auto aBase = localWordOffset(region.compute.getInputA());
  auto wBase = localWordOffset(region.compute.getInputW());
  auto oBase = localWordOffset(region.compute.getOutput());
  auto metadataBase = localWordOffset(region.compute.getMetadata());
  auto groupStride = nonnegativeU16(region.compute.getActivationGroupStrideBytes());
  auto activationScaleBase = nonnegativeU16(region.compute.getActivationScaleBase());
  auto activationScale2Base = nonnegativeU16(region.compute.getActivationScale2Base());
  auto cacheCell = nonnegativeU16(region.compute.getCacheCellIndex());
  auto scaleMetadata = nonnegativeU16(region.compute.getScaleMetadataWord());
  auto resaddBase = nonnegativeU16(region.compute.getResaddBase());
  auto mode = u8Value(region.compute.getMode(), 3);
  auto groupCount = u8Value(region.compute.getGroupCountM1(), 3);
  auto kvColumnScale = constantInt(region.compute.getKvColumnScale());
  auto pvProbabilityQ24 = constantInt(region.compute.getPvProbabilityQ24());
  auto unitWeightScale = constantInt(region.compute.getUnitWeightScale());
  auto activationScale = constantInt(region.compute.getActivationScale());
  auto activationScale2 = constantInt(region.compute.getActivationScale2());
  auto resadd = constantInt(region.compute.getResadd());
  const uint64_t metadataBytes = metadataRows.value_or(0) *
      static_cast<uint64_t>(metadataColumns.value_or(0));
  if (!m || !k || !aRows || !aColumns || !wRows || !wColumns ||
      !metadataRows || !metadataColumns || !aStride || !aBase || !wBase ||
      !oBase || !metadataBase ||
      !groupStride || !activationScaleBase || !activationScale2Base || !cacheCell ||
      !scaleMetadata || !resaddBase || !mode || !groupCount || !kvColumnScale ||
      !pvProbabilityQ24 || !unitWeightScale || !activationScale ||
      !activationScale2 || !resadd || *aRows != 1 || *aColumns != *k ||
      *wRows != *m || *wColumns != *k ||
      (region.storeO && (!outputRows || !outputBytes || !outputStride ||
          *outputRows != 1 || *outputBytes != static_cast<uint32_t>(*m) * 2)) ||
      metadataBytes == 0 ||
      *resadd != 0 ||
      metadataBytes > UINT32_MAX)
    return candidate;
  const uint64_t wSpan = static_cast<uint64_t>(*m) *
      ((*k + kLocalWordBytes - 1) / kLocalWordBytes) * kLocalWordBytes;
  if (wSpan > UINT32_MAX)
    return candidate;
  candidate.hasStaticShapeAndOffset = true;
  candidate.request.config = GemvConfig{*m, *k, *groupStride,
      *activationScaleBase, *aBase, *wBase, *oBase, *activationScale2Base,
      *cacheCell, *scaleMetadata, *resaddBase, 0, 0, 0, 0, *mode, *groupCount,
      *kvColumnScale != 0, *pvProbabilityQ24 != 0, *unitWeightScale != 0,
      *activationScale != 0, *activationScale2 != 0, false, 0, 0, 0,
      *resadd != 0};
  candidate.request.aDma = MvinAConfig{0, *aStride, *aRows, *aColumns, false, 0};
  candidate.request.wDma = MvinWConfig{0, static_cast<uint32_t>(wSpan),
      *wRows, *wColumns, 0};
  candidate.request.metadataDma = AuxMvinConfig{0,
      static_cast<uint32_t>(*metadataBase) * kLocalWordBytes,
      static_cast<uint32_t>(metadataBytes), 0, false, 0};
  candidate.request.storeToDram = static_cast<bool>(region.storeO);
  if (region.storeO)
    candidate.request.outputDma = MvoutConfig{0, *outputStride, *outputRows,
        *outputBytes, 0};
  return candidate;
}

StaticVpuCandidate buildStaticVpuCandidate(
    VpuDmaRegion region, bool descriptorAbiEnabled) {
  StaticVpuCandidate candidate;
  candidate.descriptorAbiEnabled = descriptorAbiEnabled;
  candidate.isCompleteSubgraph = true;
  auto rows = positiveU16(region.compute.getRows());
  auto columns = positiveU16(region.compute.getColumns());
  auto function = u8Value(region.compute.getFunction(),
      static_cast<uint8_t>(VpuSpecialFunction::Sigmoid));
  auto sourcePrecision = u8Value(region.compute.getSourcePrecision(), 3);
  auto destinationPrecision = u8Value(region.compute.getDestinationPrecision(), 3);
  auto inverseScale = constantInt(region.compute.getOutputInverseScaleQ8_24());
  auto sourceAddress = localWordOffset(region.compute.getInput());
  auto destinationAddress = localWordOffset(region.compute.getOutput());
  auto outputRows = positiveU16(region.storeO.getRowNum());
  auto outputBytes = positiveU16(region.storeO.getColNum());
  auto outputStride = alignedU32(region.storeO.getDramStride());
  if (!rows || !columns || !function || !sourcePrecision || !destinationPrecision ||
      !inverseScale || *inverseScale < 0 || *inverseScale > UINT32_MAX ||
      !sourceAddress || !destinationAddress || !outputRows || !outputBytes ||
      !outputStride)
    return candidate;
  const bool transpose = *function ==
      static_cast<uint8_t>(VpuSpecialFunction::Transpose);
  const bool poolMax = *function ==
      static_cast<uint8_t>(VpuSpecialFunction::PoolMax);
  const uint16_t expectedRows = transpose ? *columns
      : poolMax ? *rows / 2 : *rows;
  const uint16_t expectedColumns = transpose ? *rows
      : poolMax ? *columns / 2 : *columns;
  if (expectedRows == 0 || expectedColumns == 0 ||
      *outputRows != expectedRows)
    return candidate;
  const bool int8Output = *destinationPrecision ==
      static_cast<uint8_t>(VpuPrecision::Int8);
  const uint32_t expectedBytes = static_cast<uint32_t>(expectedColumns) *
      (int8Output ? 1 : 2);
  if (*outputBytes != expectedBytes)
    return candidate;
  candidate.hasStaticShapeAndOffset = true;
  candidate.request.config = VpuConfig{
      static_cast<VpuSpecialFunction>(*function), *rows, *columns,
      0, 1, static_cast<VpuPrecision>(*sourcePrecision),
      static_cast<VpuPrecision>(*destinationPrecision), *sourceAddress,
      *destinationAddress, static_cast<uint32_t>(*inverseScale)};
  candidate.request.storeToDram = true;
  candidate.request.outputDma = MvoutConfig{0, *outputStride, *outputRows,
      *outputBytes, 0};
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
