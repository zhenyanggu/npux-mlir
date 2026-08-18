// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <vector>

#include "src/Conversion/NpuToLLVM/VersaPDescriptor.hpp"

namespace npux::versap {

// O and ACC are physically separate ping-pong banks in the RTL. They may use
// the same bank index in one SA descriptor without a resource conflict.
enum class BankResource : uint8_t {
  A,
  W,
  O,
  Accumulator,
  Resadd,
  Metadata,
  QkBlockMax,
};
enum class EngineKind : uint8_t {
  MvinA,
  MvinW,
  AuxMvin,
  Sa,
  OutputDma,
  Gemv,
  Vpu,
  Count
};

constexpr std::size_t kEngineKindCount =
    static_cast<std::size_t>(EngineKind::Count);
enum class ScheduledCommandKind : uint8_t {
  LoadA,
  LoadW,
  LoadMetadata,
  LoadResidual,
  Gemm,
  Conv,
  Gemv,
  Vpu,
  StoreO,
};

// Dynamic descriptor fields are patched by runtime from the matching host
// memref base plus byte offset before writing the descriptor start bit.
enum class DescriptorAddressSource : uint8_t {
  A,
  W,
  Metadata,
  Residual,
  Output,
};

struct DescriptorAddressPatch {
  DescriptorAddressSource source;
  uint8_t descriptorWord;
  uint8_t bitOffset;
  uint8_t bitWidth;
};
struct BankUse {
  BankResource resource;
  uint8_t bank;
};

// Descriptor and status offsets are relative to the Versa-P MMIO base. A
// For a predecessor on the same engine, a dependency requires software to
// wait for STATUS before another command is submitted. For a predecessor on a
// different engine, it is a hardware launch condition: the RTL dispatcher
// keeps the command pending until its bank ownership and valid bits permit
// execution. accepted_count is never a dependency condition.
struct ScheduledCommand {
  uint32_t id;
  ScheduledCommandKind kind;
  EngineKind engine;
  EncodedDescriptor descriptor;
  uint32_t descriptorRegisterOffset;
  uint32_t statusRegisterOffset;
  std::vector<DescriptorAddressPatch> addressPatches;
  std::vector<BankUse> reads;
  std::vector<BankUse> writes;
  std::vector<uint32_t> dependencies;
};

struct QuantizationHints {
  bool requireRawInt32 = false;
  bool requireFp32 = false;
  bool nextConsumesBf16 = false;
  bool nextConsumesInt8 = false;
  bool perChannelScaleAvailable = false;
  uint16_t metadataWords = 0;
};

// QK row-column mode derives a per-score gamma as
// round((Sq_q8_24 * Sk_q8_24) / 2^24), saturated to unsigned Q8.24.
// Both scale vectors are packed as eight i32 Q8.24 values per metadata word.
struct QkRowColumnGamma {
  uint16_t qScaleMetadataWord;
  uint16_t kScaleMetadataWord;
  uint8_t qScaleWordStride = 4;
  uint8_t kScaleWordStride = 4;
};

// A descriptor tile is intentionally larger than the 32x32 physical SA
// array. The RTL iterates that array across the descriptor M/N extent; this
// problem describes the compiler-level tile that must fit the local banks.
struct GemmTilingProblem {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  SaOperation operation = SaOperation::Gemm;
  QuantizationHints quantization;
};

struct GemmTilePlan {
  uint16_t m;
  uint16_t n;
  uint16_t k;
  uint16_t metadataWords;
  uint32_t mTileCount;
  uint32_t nTileCount;
  uint32_t kTileCount;
  uint64_t descriptorTileCount;
  uint64_t estimatedTransferWords;
};

// Local-word offsets used by SA_COMPUTE. All fields are explicit so that a
// scheduler never invents a zero base for an unresolved memory-plan offset.
struct GemmLocalAddresses {
  uint16_t aBase;
  uint16_t wBase;
  uint16_t oBase;
  uint16_t accumulatorBase = 0;
  uint16_t residualBase = 0;
  uint16_t biasMetadataWord = 0;
  uint16_t scaleMetadataWord = 0;
};

struct ConvGeometry {
  uint16_t outputChannels;
  uint8_t kernelShapeM1;
  uint8_t strideM1;
  uint8_t dilationM1;
  uint8_t paddingLeft;
  uint8_t paddingRight;
  uint8_t paddingTop;
  uint8_t paddingBottom;
};

struct TileRequest {
  uint16_t m;
  uint16_t n;
  uint16_t k;
  bool needsBias = false;
  bool needsAccumulate = false;
  bool writePartial = false;
  bool needsResadd = false;
  bool needsVpu = false;
  bool relu = false;
  bool storeToDram = true;
  SaOperation operation = SaOperation::Gemm;
  std::optional<ConvGeometry> conv;
  QuantizationHints quantization;

  // Scalar QK mode reads the low 32 bits of scaleMetadataWord as a positive
  // Q8.24 gamma. Row-column mode instead reads Q/K scale vectors from the
  // bases described by qkRowColumnGamma. Both forms use metadataDma.
  std::optional<uint32_t> qkGammaQ8_24;
  std::optional<QkRowColumnGamma> qkRowColumnGamma;

  // G1 requires concrete static DMA information for every emitted command.
  std::optional<MvinAConfig> aDma;
  std::optional<MvinWConfig> wDma;
  std::optional<AuxMvinConfig> metadataDma;
  std::optional<AuxMvinConfig> residualDma;
  std::optional<MvoutConfig> outputDma;
  std::optional<GemmLocalAddresses> localAddresses;
};

struct GemvRequest {
  GemvConfig config;
  bool storeToDram = true;
  std::optional<MvinAConfig> aDma;
  std::optional<MvinWConfig> wDma;
  std::optional<AuxMvinConfig> metadataDma;
  std::optional<AuxMvinConfig> residualDma;
  std::optional<MvoutConfig> outputDma;
};

struct VpuRequest {
  VpuConfig config;
  // SPECIAL operations always consume one O bank and produce the other one.
  // An optional MVOUT drains that destination bank to DRAM as the next command
  // in the same descriptor DAG.
  bool storeToDram = false;
  std::optional<MvoutConfig> outputDma;
};

struct TileSchedule {
  OutputMode outputMode;
  ScaleMode scaleMode;
  uint8_t aBank;
  uint8_t wBank;
  uint8_t oBank;
  uint8_t metadataBank;
  // The descriptor ACC selector is the source bank. For write_partial, RTL
  // writes the opposite physical bank and reports it here as partialOutputBank.
  uint8_t accumulatorBank = kOutputBankCount;
  uint8_t partialOutputBank = kOutputBankCount;
  uint8_t residualBank = kOutputBankCount;
  uint8_t vpuOutputBank = kOutputBankCount;
  uint8_t qkBlockMaxSlot = kQkBlockMaxSlotCount;
  bool aIsChange = true;
  bool wIsChange = true;
  bool oIsChange = true;
  bool accumulatorIsChange = false;
  bool residualIsChange = false;
  bool metadataIsChange = false;
  bool mvoutOIsChange = false;
  std::vector<ScheduledCommand> commands;
};

struct AttentionHeadRequest {
  TileRequest qk;
  TileRequest pv;
};

struct AttentionHeadSchedule {
  TileSchedule qk;
  TileSchedule pv;

  // QK MVOUT uses the hardware logP-correction path. PV A-load waits for this
  // command and consumes the hardware QK probability encoding; no host
  // Softmax command is inserted between the two slices.
  uint32_t qkLogPCommandId = 0;
  bool usesHardwareQkLogP = true;
};

class VersaPBankScheduler {
public:
  VersaPBankScheduler();

  // Exhaustively scores all local-bank-legal compiler tiles. The score counts
  // actual A/W/O/metadata word transfers after outer tiling, descriptor count,
  // and RTL 32x32 edge waste, so larger tiles win only when they improve reuse.
  llvm::Expected<GemmTilePlan>
  planGemmTiles(const GemmTilingProblem &problem) const;
  llvm::Expected<TileSchedule> scheduleGemmTile(const TileRequest &request);
  llvm::Expected<TileSchedule> scheduleConvTile(const TileRequest &request);
  llvm::Expected<TileSchedule> scheduleGemv(const GemvRequest &request);
  llvm::Expected<TileSchedule> scheduleVpuTile(const VpuRequest &request);
  llvm::Expected<ScheduledCommand> scheduleVpu(const VpuRequest &request);
  llvm::Expected<AttentionHeadSchedule>
  scheduleAttentionHead(const AttentionHeadRequest &request);
  void reset();

private:
  struct BankState {
    uint32_t lastUse = 0;
    bool containsData = false;
  };

  OutputMode chooseOutputMode(const QuantizationHints &hints) const;
  ScaleMode chooseScaleMode(const QuantizationHints &hints,
      OutputMode outputMode) const;
  llvm::Error validateTile(
      const TileRequest &request, OutputMode outputMode,
      ScaleMode scaleMode) const;
  uint8_t chooseBank(BankResource resource,
                     std::initializer_list<uint8_t> excluded = {}) const;
  uint8_t chooseQkMetadataBank() const;
  uint32_t appendCommand(ScheduledCommandKind kind, EngineKind engine,
      EncodedDescriptor descriptor, uint32_t descriptorRegisterOffset,
      uint32_t statusRegisterOffset,
      std::vector<DescriptorAddressPatch> addressPatches,
      std::vector<BankUse> reads,
      std::vector<BankUse> writes, std::vector<ScheduledCommand> &commands);
  BankState &state(BankResource resource, uint8_t bank);
  const BankState &state(BankResource resource, uint8_t bank) const;

  std::array<BankState, kInputBankCount> aBanks;
  std::array<BankState, kInputBankCount> wBanks;
  std::array<BankState, kOutputBankCount> oBanks;
  std::array<BankState, kOutputBankCount> accumulatorBanks;
  std::array<BankState, kResaddBankCount> resaddBanks;
  std::array<BankState, kMetadataBankCount> metadataBanks;
  std::array<BankState, kQkBlockMaxSlotCount> qkBlockMaxSlots;
  // The RTL admits one executing and one pending command per opcode. Keep a
  // two-entry software window; command three waits for the oldest entry.
  std::array<std::array<uint32_t, 2>, kEngineKindCount> engineInFlight{};
  std::array<uint8_t, kEngineKindCount> engineInFlightCount{};
  // write_partial writes the bank opposite to the descriptor ACC-bank field.
  // Keep that concrete producer bank so the next accumulating slice consumes
  // the value actually retained by the RTL.
  std::optional<uint8_t> partialAccumulatorBank;
  uint32_t nextCommandId = 1;
};

} // namespace npux::versap
