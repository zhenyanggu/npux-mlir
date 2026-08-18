// SPDX-License-Identifier: Apache-2.0

#include "src/Conversion/NpuToLLVM/VersaPBankScheduler.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace npux::versap {
namespace {

constexpr uint32_t kMvinADesc0 = 0x000;
constexpr uint32_t kMvinAStatus = 0x010;
constexpr uint32_t kMvinWDesc0 = 0x020;
constexpr uint32_t kMvinWStatus = 0x030;
constexpr uint32_t kAuxMvinDesc0 = 0x040;
constexpr uint32_t kAuxMvinStatus = 0x050;
constexpr uint32_t kGemmDesc0 = 0x060;
constexpr uint32_t kGemmStatus = 0x078;
constexpr uint32_t kMvoutDesc0 = 0x080;
constexpr uint32_t kMvoutStatus = 0x090;
constexpr uint32_t kGemvDesc0 = 0x098;
constexpr uint32_t kGemvStatus = 0x0b0;
constexpr uint32_t kVpuDesc0 = 0x0c0;
constexpr uint32_t kVpuStatus = 0x0e0;

uint64_t ceilDiv(uint64_t value, uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

llvm::Error invalid(llvm::StringRef detail) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
      "Versa-P bank scheduler: %s", detail.str().c_str());
}

llvm::Error encodeError(llvm::StringRef command, llvm::Error error) {
  return llvm::joinErrors(
      llvm::createStringError(llvm::inconvertibleErrorCode(),
          "Versa-P bank scheduler: tile descriptor encoding failed for %s",
          command.str().c_str()),
      std::move(error));
}

uint8_t engineIndex(EngineKind engine) {
  return static_cast<uint8_t>(engine);
}

void addDependency(std::vector<uint32_t> &dependencies, uint32_t dependency) {
  if (dependency != 0 &&
      std::find(dependencies.begin(), dependencies.end(), dependency) ==
          dependencies.end())
    dependencies.push_back(dependency);
}

} // namespace

VersaPBankScheduler::VersaPBankScheduler() { reset(); }

void VersaPBankScheduler::reset() {
  aBanks = {};
  wBanks = {};
  oBanks = {};
  accumulatorBanks = {};
  resaddBanks = {};
  metadataBanks = {};
  qkBlockMaxSlots = {};
  engineInFlight = {};
  engineInFlightCount = {};
  partialAccumulatorBank.reset();
  nextCommandId = 1;
}

llvm::Expected<GemmTilePlan> VersaPBankScheduler::planGemmTiles(
    const GemmTilingProblem &problem) const {
  if (problem.m == 0 || problem.n == 0 || problem.k == 0 ||
      problem.k > kMaxSaK || problem.operation == SaOperation::Conv)
    return invalid("GEMM tiling requires nonzero M/N/K, K <= 4096, and a non-CONV operation");

  const OutputMode outputMode = chooseOutputMode(problem.quantization);
  const ScaleMode scaleMode = chooseScaleMode(problem.quantization, outputMode);
  const uint64_t outputLanes = outputMode == OutputMode::TensorInt8 ? 32
      : outputMode == OutputMode::Bf16 ? 16 : 8;
  const uint32_t maxM = std::min<uint32_t>(problem.m, kLocalBankWords);
  const uint32_t maxN = std::min<uint32_t>(problem.n, kLocalBankWords);

  struct Score {
    uint64_t transferWords;
    uint64_t descriptorTiles;
    uint64_t microTileWaste;
    uint64_t payload;
  };
  auto isBetter = [](const Score &candidate, const Score &best) {
    if (candidate.transferWords != best.transferWords)
      return candidate.transferWords < best.transferWords;
    if (candidate.descriptorTiles != best.descriptorTiles)
      return candidate.descriptorTiles < best.descriptorTiles;
    if (candidate.microTileWaste != best.microTileWaste)
      return candidate.microTileWaste < best.microTileWaste;
    return candidate.payload > best.payload;
  };
  auto metadataWordsForN = [&](uint32_t n) {
    uint64_t words = problem.quantization.metadataWords;
    if (scaleMode == ScaleMode::PerChannel)
      words = std::max<uint64_t>(words, ceilDiv(n, uint64_t{8}));
    return words;
  };
  auto sumKWords = [&](uint32_t tileK, uint32_t tileCount) {
    const uint32_t tail = problem.k - (tileCount - 1) * tileK;
    return static_cast<uint64_t>(tileCount - 1) *
               ceilDiv(tileK, uint64_t{kLocalWordBytes}) +
           ceilDiv(tail, uint64_t{kLocalWordBytes});
  };
  auto sumNOutputWords = [&](uint32_t tileN, uint32_t tileCount) {
    uint64_t words = 0;
    for (uint32_t index = 0; index < tileCount; ++index) {
      const uint32_t width = std::min(tileN, problem.n - index * tileN);
      words += ceilDiv(width, outputLanes);
    }
    return words;
  };
  auto sumNMetadataWords = [&](uint32_t tileN, uint32_t tileCount) {
    uint64_t words = 0;
    for (uint32_t index = 0; index < tileCount; ++index) {
      const uint32_t width = std::min(tileN, problem.n - index * tileN);
      words += metadataWordsForN(width);
    }
    return words;
  };
  auto sumMicroTileBlocks = [](uint32_t total, uint32_t tile) {
    uint64_t blocks = 0;
    for (uint32_t offset = 0; offset < total; offset += tile)
      blocks += ceilDiv(std::min(tile, total - offset), uint64_t{32});
    return blocks;
  };

  std::vector<uint64_t> outputWordsByN(maxN + 1);
  std::vector<uint64_t> metadataWordsByN(maxN + 1);
  std::vector<uint64_t> microBlocksByM(maxM + 1);
  std::vector<uint64_t> microBlocksByN(maxN + 1);
  for (uint32_t m = 1; m <= maxM; ++m)
    microBlocksByM[m] = sumMicroTileBlocks(problem.m, m);
  for (uint32_t n = 1; n <= maxN; ++n) {
    const uint32_t nTiles = static_cast<uint32_t>(ceilDiv(problem.n, n));
    outputWordsByN[n] = sumNOutputWords(n, nTiles);
    metadataWordsByN[n] = sumNMetadataWords(n, nTiles);
    microBlocksByN[n] = sumMicroTileBlocks(problem.n, n);
  }

  std::optional<GemmTilePlan> bestPlan;
  Score bestScore{std::numeric_limits<uint64_t>::max(), 0, 0, 0};
  for (uint32_t m = 1; m <= maxM; ++m) {
    for (uint32_t n = 1; n <= maxN; ++n) {
      if (static_cast<uint64_t>(m) * ceilDiv(n, outputLanes) >
          kLocalBankWords)
        continue;
      if (metadataWordsForN(n) > kMetadataWords)
        continue;

      const uint32_t kWords = std::min(kLocalBankWords / m,
          kLocalBankWords / n);
      if (kWords == 0)
        continue;
      const uint32_t k = std::min<uint32_t>(problem.k,
          std::min<uint32_t>(kMaxSaK, kWords * kLocalWordBytes));
      const uint32_t mTiles = static_cast<uint32_t>(ceilDiv(problem.m, m));
      const uint32_t nTiles = static_cast<uint32_t>(ceilDiv(problem.n, n));
      const uint32_t kTiles = static_cast<uint32_t>(ceilDiv(problem.k, k));
      const uint64_t descriptorTiles =
          static_cast<uint64_t>(mTiles) * nTiles * kTiles;
      const uint64_t aWords = static_cast<uint64_t>(problem.m) *
          sumKWords(k, kTiles) * nTiles;
      const uint64_t wWords = static_cast<uint64_t>(problem.n) *
          sumKWords(k, kTiles) * mTiles;
      const uint64_t oWords =
          static_cast<uint64_t>(problem.m) * outputWordsByN[n];
      const uint64_t metadataWords =
          metadataWordsByN[n] * mTiles * kTiles;
      const uint64_t rtlMicroTiles =
          microBlocksByM[m] * microBlocksByN[n] * kTiles;
      const Score score{aWords + wWords + oWords + metadataWords,
          descriptorTiles,
          rtlMicroTiles * 32 * 32 -
              static_cast<uint64_t>(problem.m) * problem.n * kTiles,
          static_cast<uint64_t>(m) * n * k};
      if (!bestPlan || isBetter(score, bestScore)) {
        bestPlan = GemmTilePlan{static_cast<uint16_t>(m),
            static_cast<uint16_t>(n), static_cast<uint16_t>(k),
            static_cast<uint16_t>(metadataWordsForN(n)), mTiles, nTiles,
            kTiles, descriptorTiles, score.transferWords};
        bestScore = score;
      }
    }
  }
  if (!bestPlan)
    return invalid(
        "no GEMM tile fits the A/W/O or metadata bank capacities");
  return *bestPlan;
}

OutputMode VersaPBankScheduler::chooseOutputMode(
    const QuantizationHints &hints) const {
  if (hints.requireFp32)
    return OutputMode::Fp32;
  if (hints.requireRawInt32)
    return OutputMode::RawInt32;
  if (hints.nextConsumesBf16)
    return OutputMode::Bf16;
  if (hints.nextConsumesInt8)
    return OutputMode::TensorInt8;
  return OutputMode::RawInt32;
}

ScaleMode VersaPBankScheduler::chooseScaleMode(
    const QuantizationHints &hints, OutputMode outputMode) const {
  if (outputMode == OutputMode::RawInt32 || outputMode == OutputMode::Bf16)
    return ScaleMode::None;
  return hints.perChannelScaleAvailable ? ScaleMode::PerChannel
                                        : ScaleMode::PerTensor;
}

llvm::Error VersaPBankScheduler::validateTile(
    const TileRequest &request, OutputMode outputMode,
    ScaleMode scaleMode) const {
  if (request.m == 0 || request.n == 0 || request.k == 0 ||
      (request.operation != SaOperation::Conv && request.k > kMaxSaK))
    return invalid("M, N, and K must be nonzero and K must not exceed 4096");
  if (request.quantization.requireFp32 &&
      request.quantization.requireRawInt32)
    return invalid("a tile cannot require both FP32 and raw INT32 output");
  if (request.quantization.nextConsumesBf16 &&
      request.quantization.nextConsumesInt8)
    return invalid("a tile cannot target both BF16 and INT8 output");
  if (request.needsAccumulate && request.needsBias)
    return invalid("accumulate and bias cannot be enabled together");
  if (request.needsAccumulate && !partialAccumulatorBank)
    return invalid("ACC accumulation requires a preceding write_partial tile");
  if (request.writePartial && request.storeToDram)
    return invalid("partial GEMM must retain its result in ACC, not MVOUT to DRAM");
  if (request.writePartial &&
      (outputMode != OutputMode::RawInt32 || scaleMode != ScaleMode::None))
    return invalid("partial GEMM requires raw INT32 output without scaling");
  if (request.needsVpu)
    return invalid("use scheduleVpu for a VPU command");
  if (request.operation == SaOperation::Conv && !request.conv)
    return invalid("CONV requires explicit output-channel and geometry information");
  if (request.operation == SaOperation::AttentionQk) {
    if (outputMode != OutputMode::RawInt32 || request.needsAccumulate ||
        request.writePartial || request.needsResadd || !request.outputDma ||
        !request.outputDma->qkMode)
      return invalid("attention QK requires raw INT32 output and QK-mode MVOUT");
    if (request.qkGammaQ8_24 && request.qkRowColumnGamma)
      return invalid("attention QK cannot combine scalar and row-column gamma");
    if (!request.qkGammaQ8_24 && !request.qkRowColumnGamma)
      return invalid("attention QK requires scalar or row-column gamma metadata");
    if (request.qkGammaQ8_24 && (*request.qkGammaQ8_24 == 0 ||
                                  *request.qkGammaQ8_24 >= (uint32_t{1} << 25)))
      return invalid("attention QK requires a positive 25-bit Q8.24 gamma");
    if (!request.metadataDma || request.metadataDma->value < 4 ||
        request.quantization.metadataWords == 0)
      return invalid("attention QK requires gamma in metadata word zero");
    if (request.localAddresses && request.localAddresses->scaleMetadataWord != 0)
      return invalid("attention QK gamma must use metadata word zero");
    if (request.qkRowColumnGamma) {
      const QkRowColumnGamma &gamma = *request.qkRowColumnGamma;
      const uint32_t qWords = ((request.m + 31) / 32) * 4;
      const uint32_t kWords = ((request.n + 31) / 32) * 4;
      const uint32_t qEnd = gamma.qScaleMetadataWord + qWords;
      const uint32_t kEnd = gamma.kScaleMetadataWord + kWords;
      if (gamma.qScaleWordStride != 4 || gamma.kScaleWordStride != 4 ||
          qEnd > kMetadataWords || kEnd > kMetadataWords ||
          (gamma.qScaleMetadataWord < kEnd &&
              gamma.kScaleMetadataWord < qEnd) ||
          request.metadataDma->offsetBytes != 0 ||
          request.quantization.metadataWords < std::max(qEnd, kEnd) ||
          request.metadataDma->value < std::max(qEnd, kEnd) * kLocalWordBytes)
        return invalid("QK row-column gamma requires packed 4-word scale blocks");
    }
    if (request.m > kQkMaxRows || request.n > 2016 ||
        static_cast<uint32_t>(request.m) * ((request.n + 31) / 32) >
            static_cast<uint32_t>(kQkBlockMaxWordsPerSlot) *
                kQkBlockMaxEntriesPerWord)
      return invalid("attention QK exceeds the dedicated block-max slot capacity");
  }
  if (request.operation == SaOperation::AttentionPv &&
      (outputMode != OutputMode::TensorInt8 || request.needsAccumulate ||
       request.writePartial || request.needsBias ||
       request.outputDma && request.outputDma->qkMode))
    return invalid("attention PV requires INT8 output without QK-mode MVOUT");
  if (!request.aDma || !request.wDma || !request.localAddresses)
    return invalid("tile requires explicit A/W DMA and local-word addresses");
  if (request.storeToDram && !request.outputDma)
    return invalid("store-to-DRAM tile requires an explicit MVOUT DMA request");

  if (request.operation != SaOperation::Conv) {
    const uint64_t inputWords = ceilDiv(request.k, kLocalWordBytes);
    const uint64_t aWords = request.m * inputWords;
    const uint64_t wWords = request.n * inputWords;
    const uint64_t outputLanes = outputMode == OutputMode::TensorInt8 ? 32
        : outputMode == OutputMode::Bf16 ? 16 : 8;
    const uint64_t outputWords = request.m * ceilDiv(request.n, outputLanes);
    if (aWords > kLocalBankWords || wWords > kLocalBankWords ||
        outputWords > kLocalBankWords)
      return invalid("tile does not fit in an input or output bank");
    if (request.aDma->rowCount != request.m || request.aDma->colCount != request.k ||
        request.wDma->rowCount != request.n || request.wDma->colCount != request.k)
      return invalid("A/W DMA shape must match the GEMM tile shape");
  }

  const bool usesMetadata = request.needsBias ||
                            request.quantization.metadataWords != 0 ||
                            scaleMode == ScaleMode::PerChannel;
  uint64_t requiredMetadata = request.quantization.metadataWords;
  if (scaleMode == ScaleMode::PerChannel)
    requiredMetadata = std::max<uint64_t>(
        requiredMetadata, ceilDiv(request.n, uint64_t{8}));
  if (requiredMetadata > kMetadataWords)
    return invalid("tile metadata does not fit in one metadata bank");
  if (usesMetadata && !request.metadataDma)
    return invalid("metadata-consuming tile requires an explicit AUX_MVIN request");
  if (request.metadataDma && request.metadataDma->destinationIsOBank)
    return invalid("metadata AUX_MVIN request must target a metadata bank");
  if (request.needsResadd && !request.residualDma)
    return invalid("resadd tile requires an explicit residual AUX_MVIN request");
  if (request.residualDma && !request.residualDma->destinationIsOBank)
    return invalid("residual AUX_MVIN request must target the physical Res bank");

  if (request.outputDma) {
    const uint16_t outputRows = request.operation == SaOperation::Conv
        ? static_cast<uint16_t>(request.outputDma->rowCount)
        : request.m;
    const uint16_t outputColumns = request.operation == SaOperation::Conv
        ? request.conv->outputChannels : request.n;
    const uint32_t rowBytes = outputMode == OutputMode::TensorInt8 ? outputColumns
        : outputMode == OutputMode::Bf16 ? static_cast<uint32_t>(outputColumns) * 2
        : static_cast<uint32_t>(outputColumns) * 4;
    if (rowBytes > std::numeric_limits<uint16_t>::max() ||
        request.outputDma->rowCount != outputRows ||
        request.outputDma->rowBytes != rowBytes)
      return invalid("MVOUT shape must match the SA output mode and tile");
  }
  return llvm::Error::success();
}

uint8_t VersaPBankScheduler::chooseBank(
    BankResource resource, std::initializer_list<uint8_t> excluded) const {
  uint8_t result = 0xff;
  uint32_t earliestUse = std::numeric_limits<uint32_t>::max();
  const uint8_t bankCount = resource == BankResource::O
                                 ? kOutputBankCount
                                 : resource == BankResource::Resadd
                                       ? kResaddBankCount
                                 : resource == BankResource::Metadata
                                      ? kMetadataBankCount
                                      : resource == BankResource::QkBlockMax
                                      ? kQkBlockMaxSlotCount
                                      : kInputBankCount;
  for (uint8_t bank = 0; bank < bankCount; ++bank) {
    if (std::find(excluded.begin(), excluded.end(), bank) != excluded.end())
      continue;
    const uint32_t lastUse = state(resource, bank).lastUse;
    if (lastUse < earliestUse) {
      earliestUse = lastUse;
      result = bank;
    }
  }
  return result;
}

uint8_t VersaPBankScheduler::chooseQkMetadataBank() const {
  uint8_t result = 0xff;
  uint32_t earliestUse = std::numeric_limits<uint32_t>::max();
  for (uint8_t metadataBank = 0; metadataBank < kMetadataBankCount;
       ++metadataBank) {
    const uint8_t qkBlockMaxSlot = metadataBank ^ 1;
    const uint32_t lastUse = std::max(
        state(BankResource::Metadata, metadataBank).lastUse,
        state(BankResource::QkBlockMax, qkBlockMaxSlot).lastUse);
    if (lastUse < earliestUse) {
      earliestUse = lastUse;
      result = metadataBank;
    }
  }
  return result;
}

VersaPBankScheduler::BankState &VersaPBankScheduler::state(
    BankResource resource, uint8_t bank) {
  switch (resource) {
  case BankResource::A:
    return aBanks[bank];
  case BankResource::W:
    return wBanks[bank];
  case BankResource::O:
    return oBanks[bank];
  case BankResource::Accumulator:
    return accumulatorBanks[bank];
  case BankResource::Resadd:
    return resaddBanks[bank];
  case BankResource::Metadata:
    return metadataBanks[bank];
  case BankResource::QkBlockMax:
    return qkBlockMaxSlots[bank];
  }
  llvm_unreachable("unknown bank resource");
}

const VersaPBankScheduler::BankState &VersaPBankScheduler::state(
    BankResource resource, uint8_t bank) const {
  switch (resource) {
  case BankResource::A:
    return aBanks[bank];
  case BankResource::W:
    return wBanks[bank];
  case BankResource::O:
    return oBanks[bank];
  case BankResource::Accumulator:
    return accumulatorBanks[bank];
  case BankResource::Resadd:
    return resaddBanks[bank];
  case BankResource::Metadata:
    return metadataBanks[bank];
  case BankResource::QkBlockMax:
    return qkBlockMaxSlots[bank];
  }
  llvm_unreachable("unknown bank resource");
}

uint32_t VersaPBankScheduler::appendCommand(ScheduledCommandKind kind,
    EngineKind engine, EncodedDescriptor descriptor,
    uint32_t descriptorRegisterOffset, uint32_t statusRegisterOffset,
    std::vector<DescriptorAddressPatch> addressPatches,
    std::vector<BankUse> reads, std::vector<BankUse> writes,
    std::vector<ScheduledCommand> &commands) {
  ScheduledCommand command{nextCommandId++, kind, engine, descriptor,
      descriptorRegisterOffset, statusRegisterOffset, std::move(addressPatches),
      std::move(reads),
      std::move(writes), {}};
  const uint8_t index = engineIndex(engine);
  if (engineInFlightCount[index] == 2)
    addDependency(command.dependencies, engineInFlight[index][0]);
  for (const BankUse &use : command.reads)
    addDependency(command.dependencies, state(use.resource, use.bank).lastUse);
  for (const BankUse &use : command.writes)
    addDependency(command.dependencies, state(use.resource, use.bank).lastUse);

  if (engineInFlightCount[index] < 2) {
    engineInFlight[index][engineInFlightCount[index]++] = command.id;
  } else {
    engineInFlight[index][0] = engineInFlight[index][1];
    engineInFlight[index][1] = command.id;
  }
  for (const BankUse &use : command.reads)
    state(use.resource, use.bank).lastUse = command.id;
  for (const BankUse &use : command.writes)
    state(use.resource, use.bank).lastUse = command.id;
  for (const BankUse &use : command.writes)
    if (use.resource == BankResource::O)
      state(use.resource, use.bank).containsData = true;
  // An O-bank MVOUT with o_ischange transfers ownership to DRAM. The emitted
  // descriptor always carries that flag for scheduler-managed stores.
  if (kind == ScheduledCommandKind::StoreO)
    for (const BankUse &use : command.reads)
      if (use.resource == BankResource::O)
        state(use.resource, use.bank).containsData = false;
  commands.push_back(std::move(command));
  return commands.back().id;
}

llvm::Expected<TileSchedule> VersaPBankScheduler::scheduleGemmTile(
    const TileRequest &request) {
  const OutputMode outputMode = chooseOutputMode(request.quantization);
  const ScaleMode scaleMode = chooseScaleMode(request.quantization, outputMode);
  if (llvm::Error error = validateTile(request, outputMode, scaleMode))
    return std::move(error);

  const uint8_t metadataBank = request.operation == SaOperation::AttentionQk
      ? chooseQkMetadataBank()
      : chooseBank(BankResource::Metadata);
  TileSchedule result{outputMode, scaleMode, chooseBank(BankResource::A),
      chooseBank(BankResource::W), chooseBank(BankResource::O), metadataBank};
  if (request.operation == SaOperation::AttentionQk)
    result.qkBlockMaxSlot = result.metadataBank ^ 1;
  if (request.needsAccumulate)
    result.accumulatorBank = *partialAccumulatorBank;
  if (request.writePartial) {
    if (request.needsAccumulate) {
      result.partialOutputBank = result.accumulatorBank ^ 1;
    } else {
      result.partialOutputBank = chooseBank(BankResource::Accumulator);
      result.accumulatorBank = result.partialOutputBank ^ 1;
    }
  }
  if (request.needsResadd)
    result.residualBank = chooseBank(BankResource::Resadd);
  if (result.accumulatorBank == 0xff || result.residualBank == 0xff)
    return invalid("no bank is available for accumulator or residual");

  const bool usesMetadata = request.needsBias ||
                            request.quantization.metadataWords != 0 ||
                            scaleMode == ScaleMode::PerChannel;
  result.accumulatorIsChange = request.needsAccumulate || request.writePartial;
  result.residualIsChange = request.needsResadd;
  result.metadataIsChange = usesMetadata;
  result.mvoutOIsChange = request.storeToDram;

  MvinAConfig a = *request.aDma;
  a.bank = result.aBank;
  auto aDescriptor = encodeMvinA(a);
  if (!aDescriptor)
    return encodeError("MVIN_A", aDescriptor.takeError());
  appendCommand(ScheduledCommandKind::LoadA, EngineKind::MvinA, *aDescriptor,
      kMvinADesc0, kMvinAStatus, {{DescriptorAddressSource::A, 0, 0, 32}},
      {}, {{BankResource::A, result.aBank}},
      result.commands);

  MvinWConfig w = *request.wDma;
  w.bank = result.wBank;
  auto wDescriptor = encodeMvinW(w);
  if (!wDescriptor)
    return encodeError("MVIN_W", wDescriptor.takeError());
  appendCommand(ScheduledCommandKind::LoadW, EngineKind::MvinW, *wDescriptor,
      kMvinWDesc0, kMvinWStatus, {{DescriptorAddressSource::W, 0, 0, 32}},
      {}, {{BankResource::W, result.wBank}},
      result.commands);

  if (usesMetadata) {
    AuxMvinConfig metadata = *request.metadataDma;
    metadata.destinationIsOBank = false;
    metadata.metadataBank = result.metadataBank;
    auto descriptor = encodeAuxMvin(metadata);
    if (!descriptor)
      return encodeError("AUX_MVIN metadata", descriptor.takeError());
    appendCommand(ScheduledCommandKind::LoadMetadata, EngineKind::AuxMvin,
        *descriptor, kAuxMvinDesc0, kAuxMvinStatus,
        {{DescriptorAddressSource::Metadata, 0, 0, 32}}, {},
        {{BankResource::Metadata, result.metadataBank}}, result.commands);
  }
  if (request.needsResadd) {
    AuxMvinConfig residual = *request.residualDma;
    residual.destinationIsOBank = true;
    residual.auxTypeOrOBank = result.residualBank;
    auto descriptor = encodeAuxMvin(residual);
    if (!descriptor)
      return encodeError("AUX_MVIN residual", descriptor.takeError());
    appendCommand(ScheduledCommandKind::LoadResidual, EngineKind::AuxMvin,
        *descriptor, kAuxMvinDesc0, kAuxMvinStatus,
        {{DescriptorAddressSource::Residual, 0, 0, 32}}, {},
        {{BankResource::Resadd, result.residualBank}}, result.commands);
  }

  const GemmLocalAddresses &local = *request.localAddresses;
  GemmConfig gemm{request.m, request.n, request.k, local.aBase, local.wBase,
      local.oBase, local.accumulatorBase, local.residualBase,
      local.biasMetadataWord, local.scaleMetadataWord, result.aBank,
      result.wBank, result.oBank,
      request.needsAccumulate || request.writePartial
          ? result.accumulatorBank
          : uint8_t{0},
      request.needsResadd ? result.residualBank : uint8_t{0}, result.metadataBank,
      outputMode, scaleMode, request.needsAccumulate, request.needsBias, request.needsResadd,
      result.aIsChange, result.wIsChange, result.oIsChange,
      result.accumulatorIsChange, result.residualIsChange, result.metadataIsChange,
      request.operation == SaOperation::AttentionQk
          ? static_cast<bool>(result.metadataBank)
          : request.writePartial,
      request.relu, request.operation};
  if (request.operation == SaOperation::Conv) {
    const ConvGeometry &conv = *request.conv;
    gemm.d3 = conv.outputChannels;
    gemm.kernelShapeM1 = conv.kernelShapeM1;
    gemm.strideM1 = conv.strideM1;
    gemm.dilationM1 = conv.dilationM1;
    gemm.paddingLeft = conv.paddingLeft;
    gemm.paddingRight = conv.paddingRight;
    gemm.paddingTop = conv.paddingTop;
    gemm.paddingBottom = conv.paddingBottom;
  }
  if (request.operation == SaOperation::AttentionQk &&
      request.qkRowColumnGamma) {
    const QkRowColumnGamma &gamma = *request.qkRowColumnGamma;
    gemm.qkGammaDescriptor = uint64_t{1} |
        (static_cast<uint64_t>(gamma.qScaleMetadataWord) << 1) |
        (static_cast<uint64_t>(gamma.kScaleMetadataWord) << 10) |
        (static_cast<uint64_t>(gamma.qScaleWordStride) << 19) |
        (static_cast<uint64_t>(gamma.kScaleWordStride) << 27);
  }
  auto gemmDescriptor = encodeGemm(gemm);
  if (!gemmDescriptor)
    return encodeError("SA_COMPUTE", gemmDescriptor.takeError());
  std::vector<BankUse> gemmReads = {
      {BankResource::A, result.aBank}, {BankResource::W, result.wBank}};
  if (usesMetadata)
    gemmReads.push_back({BankResource::Metadata, result.metadataBank});
  if (request.needsAccumulate)
    gemmReads.push_back({BankResource::Accumulator, result.accumulatorBank});
  if (request.needsResadd)
    gemmReads.push_back({BankResource::Resadd, result.residualBank});
  std::vector<BankUse> gemmWrites;
  if (request.operation == SaOperation::AttentionQk)
    gemmWrites.push_back({BankResource::QkBlockMax, result.qkBlockMaxSlot});
  if (!request.writePartial)
    gemmWrites.push_back({BankResource::O, result.oBank});
  if (request.writePartial)
    gemmWrites.push_back(
        {BankResource::Accumulator, result.partialOutputBank});
  appendCommand(request.operation == SaOperation::Conv ? ScheduledCommandKind::Conv :
                                                        ScheduledCommandKind::Gemm,
      EngineKind::Sa, *gemmDescriptor,
      kGemmDesc0, kGemmStatus, {}, std::move(gemmReads),
      std::move(gemmWrites), result.commands);
  if (request.writePartial)
    partialAccumulatorBank = result.partialOutputBank;
  else if (request.needsAccumulate)
    partialAccumulatorBank.reset();

  if (request.storeToDram) {
    MvoutConfig output = *request.outputDma;
    output.bank = result.oBank;
    output.oIsChange = result.mvoutOIsChange;
    if (request.operation == SaOperation::AttentionQk)
      output.qkBlockMaxSlot = result.qkBlockMaxSlot;
    auto outputDescriptor = encodeMvout(output);
    if (!outputDescriptor)
      return encodeError("MVOUT", outputDescriptor.takeError());
    std::vector<BankUse> outputReads = {{BankResource::O, result.oBank}};
    if (request.operation == SaOperation::AttentionQk)
      outputReads.push_back({BankResource::QkBlockMax, result.qkBlockMaxSlot});
    appendCommand(ScheduledCommandKind::StoreO, EngineKind::OutputDma,
        *outputDescriptor, kMvoutDesc0, kMvoutStatus,
        {{DescriptorAddressSource::Output, 0, 0, 32}},
        std::move(outputReads), {}, result.commands);
  }
  return result;
}

llvm::Expected<TileSchedule> VersaPBankScheduler::scheduleConvTile(
    const TileRequest &request) {
  if (request.operation != SaOperation::Conv)
    return invalid("scheduleConvTile requires SA_COMPUTE CONV operation");
  return scheduleGemmTile(request);
}

llvm::Expected<TileSchedule> VersaPBankScheduler::scheduleGemv(
    const GemvRequest &request) {
  if (!request.aDma || !request.wDma || !request.metadataDma)
    return invalid("GEMV requires explicit A, W, and metadata DMA requests");
  if (request.storeToDram && !request.outputDma)
    return invalid("GEMV store-to-DRAM requires an explicit MVOUT DMA request");
  if (request.outputDma) {
    const uint32_t outputBytes = static_cast<uint32_t>(request.config.m) * 2;
    if (outputBytes > std::numeric_limits<uint16_t>::max() ||
        request.outputDma->rowCount != 1 ||
        request.outputDma->rowBytes != outputBytes)
      return invalid("GEMV BF16 MVOUT must contain one M-element vector");
  }
  // npu_top defaults GEMV_ELEM_FORMAT to BF16; GEMV writes 16-bit elements
  // directly to the selected O bank.
  TileSchedule result{OutputMode::Bf16, ScaleMode::None,
      chooseBank(BankResource::A), chooseBank(BankResource::W),
      chooseBank(BankResource::O), chooseBank(BankResource::Metadata)};
  GemvConfig gemv = request.config;
  gemv.aBank = result.aBank;
  gemv.wBank = result.wBank;
  gemv.oBank = result.oBank;
  gemv.metadataBank = result.metadataBank;
  MvinAConfig a = *request.aDma;
  a.bank = result.aBank;
  auto aDescriptor = encodeMvinA(a);
  if (!aDescriptor)
    return encodeError("GEMV MVIN_A", aDescriptor.takeError());
  appendCommand(ScheduledCommandKind::LoadA, EngineKind::MvinA, *aDescriptor,
      kMvinADesc0, kMvinAStatus, {{DescriptorAddressSource::A, 0, 0, 32}}, {},
      {{BankResource::A, result.aBank}}, result.commands);
  MvinWConfig w = *request.wDma;
  w.bank = result.wBank;
  auto wDescriptor = encodeMvinW(w);
  if (!wDescriptor)
    return encodeError("GEMV MVIN_W", wDescriptor.takeError());
  appendCommand(ScheduledCommandKind::LoadW, EngineKind::MvinW, *wDescriptor,
      kMvinWDesc0, kMvinWStatus, {{DescriptorAddressSource::W, 0, 0, 32}}, {},
      {{BankResource::W, result.wBank}}, result.commands);
  AuxMvinConfig metadata = *request.metadataDma;
  metadata.destinationIsOBank = false;
  metadata.metadataBank = result.metadataBank;
  auto metadataDescriptor = encodeAuxMvin(metadata);
  if (!metadataDescriptor)
    return encodeError("GEMV AUX_MVIN metadata", metadataDescriptor.takeError());
  appendCommand(ScheduledCommandKind::LoadMetadata, EngineKind::AuxMvin,
      *metadataDescriptor, kAuxMvinDesc0, kAuxMvinStatus,
      {{DescriptorAddressSource::Metadata, 0, 0, 32}}, {},
      {{BankResource::Metadata, result.metadataBank}}, result.commands);
  std::vector<BankUse> reads = {{BankResource::A, result.aBank},
      {BankResource::W, result.wBank}, {BankResource::Metadata, result.metadataBank}};
  if (gemv.resadd) {
    if (!request.residualDma)
      return invalid("GEMV ResAdd requires an explicit residual AUX_MVIN request");
    result.residualBank = chooseBank(BankResource::Resadd);
    AuxMvinConfig residual = *request.residualDma;
    residual.destinationIsOBank = true;
    residual.auxTypeOrOBank = result.residualBank;
    auto residualDescriptor = encodeAuxMvin(residual);
    if (!residualDescriptor)
      return encodeError("GEMV AUX_MVIN residual", residualDescriptor.takeError());
    appendCommand(ScheduledCommandKind::LoadResidual, EngineKind::AuxMvin,
        *residualDescriptor, kAuxMvinDesc0, kAuxMvinStatus,
        {{DescriptorAddressSource::Residual, 0, 0, 32}}, {},
        {{BankResource::Resadd, result.residualBank}}, result.commands);
    reads.push_back({BankResource::Resadd, result.residualBank});
  }
  auto gemvDescriptor = encodeGemv(gemv);
  if (!gemvDescriptor)
    return encodeError("GEMV", gemvDescriptor.takeError());
  appendCommand(ScheduledCommandKind::Gemv, EngineKind::Gemv, *gemvDescriptor,
      kGemvDesc0, kGemvStatus, {}, std::move(reads),
      {{BankResource::O, result.oBank}}, result.commands);
  if (request.storeToDram) {
    MvoutConfig output = *request.outputDma;
    output.bank = result.oBank;
    auto outputDescriptor = encodeMvout(output);
    if (!outputDescriptor)
      return encodeError("GEMV MVOUT", outputDescriptor.takeError());
    std::vector<BankUse> outputReads = {{BankResource::O, result.oBank}};
    appendCommand(ScheduledCommandKind::StoreO, EngineKind::OutputDma,
        *outputDescriptor, kMvoutDesc0, kMvoutStatus,
        {{DescriptorAddressSource::Output, 0, 0, 32}},
        std::move(outputReads), {}, result.commands);
  }
  return result;
}

llvm::Expected<TileSchedule> VersaPBankScheduler::scheduleVpuTile(
    const VpuRequest &request) {
  if (request.storeToDram && !request.outputDma)
    return invalid("VPU store-to-DRAM requires an explicit MVOUT DMA request");
  if (!request.storeToDram && request.outputDma)
    return invalid("VPU output DMA requires store-to-DRAM to be enabled");
  auto descriptor = encodeVpu(request.config);
  if (!descriptor)
    return encodeError("VPU", descriptor.takeError());
  if (!state(BankResource::O, request.config.sourceSelect).containsData)
    return invalid("VPU source O bank has no live tensor data");
  if (state(BankResource::O, request.config.destinationSelect).containsData)
    return invalid("VPU destination O bank must be empty");
  const bool int8Output =
      request.config.destinationPrecision == VpuPrecision::Int8;
  const uint32_t elementBytes = int8Output ? 1 : 2;
  if (request.outputDma) {
    const bool transpose =
        request.config.function == VpuSpecialFunction::Transpose;
    const bool poolMax =
        request.config.function == VpuSpecialFunction::PoolMax;
    const uint16_t outputRows = transpose ? request.config.columns
        : poolMax ? request.config.rows / 2 : request.config.rows;
    const uint16_t outputColumns = transpose ? request.config.rows
        : poolMax ? request.config.columns / 2 : request.config.columns;
    const uint32_t rowBytes =
        static_cast<uint32_t>(outputColumns) * elementBytes;
    if (outputRows == 0 || outputColumns == 0 ||
        request.outputDma->rowCount != outputRows ||
        request.outputDma->rowBytes != rowBytes)
      return invalid("VPU MVOUT shape must match the destination O-bank tensor");
  }
  TileSchedule result{int8Output ? OutputMode::TensorInt8 : OutputMode::Bf16,
      ScaleMode::None, kInputBankCount, kInputBankCount,
      request.config.destinationSelect, kMetadataBankCount};
  result.vpuOutputBank = request.config.destinationSelect;
  std::vector<BankUse> reads;
  std::vector<BankUse> writes;
  // VpuConfig deliberately represents only SPECIAL tensor operations. RTL
  // arbitrates their source and destination as O-bank use/load ownership.
  reads.push_back({BankResource::O, request.config.sourceSelect});
  writes.push_back({BankResource::O, request.config.destinationSelect});
  appendCommand(ScheduledCommandKind::Vpu, EngineKind::Vpu, *descriptor,
      kVpuDesc0, kVpuStatus, {}, std::move(reads), std::move(writes), result.commands);
  // SPECIAL consumes its source tensor. Keep the read dependency, but make
  // that O bank available to later producers after this VPU command.
  state(BankResource::O, request.config.sourceSelect).containsData = false;
  if (request.storeToDram) {
    MvoutConfig output = *request.outputDma;
    output.bank = request.config.destinationSelect;
    auto outputDescriptor = encodeMvout(output);
    if (!outputDescriptor)
      return encodeError("VPU MVOUT", outputDescriptor.takeError());
    appendCommand(ScheduledCommandKind::StoreO, EngineKind::OutputDma,
        *outputDescriptor, kMvoutDesc0, kMvoutStatus,
        {{DescriptorAddressSource::Output, 0, 0, 32}},
        {{BankResource::O, output.bank}}, {}, result.commands);
  }
  return result;
}

llvm::Expected<ScheduledCommand> VersaPBankScheduler::scheduleVpu(
    const VpuRequest &request) {
  auto schedule = scheduleVpuTile(request);
  if (!schedule)
    return schedule.takeError();
  if (schedule->commands.empty())
    return invalid("VPU schedule did not create a command");
  return schedule->commands.front();
}

llvm::Expected<AttentionHeadSchedule>
VersaPBankScheduler::scheduleAttentionHead(
    const AttentionHeadRequest &request) {
  if (request.qk.m != request.pv.m || request.qk.n != request.pv.k)
    return invalid("QK output [M, sequence] must be the PV input [M, K]");
  if (request.qk.operation != SaOperation::AttentionQk ||
      request.pv.operation != SaOperation::AttentionPv)
    return invalid("attention head requires QK and PV SA operation types");
  if (!request.qk.quantization.requireRawInt32 || !request.qk.storeToDram ||
      !request.qk.outputDma || !request.qk.outputDma->qkMode)
    return invalid("QK must use hardware logP MVOUT");
  if (!request.pv.quantization.nextConsumesInt8 ||
      request.pv.quantization.requireRawInt32 ||
      request.pv.quantization.requireFp32)
    return invalid("PV must produce an INT8 tensor for the next attention consumer");
  if (request.qk.needsVpu || request.pv.needsVpu)
    return invalid("attention Softmax is an external boundary until the VPU ABI is defined");

  auto qk = scheduleGemmTile(request.qk);
  if (!qk)
    return qk.takeError();
  auto pv = scheduleGemmTile(request.pv);
  if (!pv)
    return pv.takeError();

  AttentionHeadSchedule result{std::move(*qk), std::move(*pv)};
  if (result.qk.commands.empty() ||
      result.qk.commands.back().kind != ScheduledCommandKind::StoreO)
    return invalid("QK schedule is missing its logP MVOUT command");
  result.qkLogPCommandId = result.qk.commands.back().id;
  if (!result.pv.commands.empty())
    result.pv.commands.front().dependencies.push_back(result.qkLogPCommandId);
  return result;
}

} // namespace npux::versap
