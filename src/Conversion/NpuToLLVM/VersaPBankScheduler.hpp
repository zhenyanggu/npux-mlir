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

enum class BankResource : uint8_t { A, W, O, Resadd, Metadata };
enum class EngineKind : uint8_t {
  MvinA,
  MvinW,
  AuxMvin,
  Sa,
  Vpu,
  OutputDma,
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
// dependency means the predecessor STATUS has reported done without error;
// accepted_count is never a dependency condition.
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
  bool nextConsumesInt8 = false;
  bool perChannelScaleAvailable = false;
  uint16_t metadataWords = 0;
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

struct TileRequest {
  uint16_t m;
  uint16_t n;
  uint16_t k;
  bool needsBias = false;
  bool needsAccumulate = false;
  bool needsResadd = false;
  bool needsVpu = false;
  bool relu = false;
  bool storeToDram = true;
  SaOperation operation = SaOperation::Gemm;
  QuantizationHints quantization;

  // G1 requires concrete static DMA information for every emitted command.
  std::optional<MvinAConfig> aDma;
  std::optional<MvinWConfig> wDma;
  std::optional<AuxMvinConfig> metadataDma;
  std::optional<AuxMvinConfig> residualDma;
  std::optional<MvoutConfig> outputDma;
  std::optional<GemmLocalAddresses> localAddresses;
};

struct TileSchedule {
  OutputMode outputMode;
  ScaleMode scaleMode;
  uint8_t aBank;
  uint8_t wBank;
  uint8_t oBank;
  uint8_t metadataBank;
  uint8_t accumulatorBank = kOutputBankCount;
  uint8_t residualBank = kOutputBankCount;
  uint8_t vpuOutputBank = kOutputBankCount;
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

  llvm::Expected<TileSchedule> scheduleGemmTile(const TileRequest &request);
  llvm::Expected<AttentionHeadSchedule>
  scheduleAttentionHead(const AttentionHeadRequest &request);
  void reset();

private:
  struct BankState {
    uint32_t lastUse = 0;
  };

  OutputMode chooseOutputMode(const QuantizationHints &hints) const;
  ScaleMode chooseScaleMode(const QuantizationHints &hints,
      OutputMode outputMode) const;
  llvm::Error validateTile(
      const TileRequest &request, OutputMode outputMode,
      ScaleMode scaleMode) const;
  uint8_t chooseBank(BankResource resource,
      std::initializer_list<uint8_t> excluded = {}) const;
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
  std::array<BankState, kResaddBankCount> resaddBanks;
  std::array<BankState, kMetadataBankCount> metadataBanks;
  std::array<uint32_t, kEngineKindCount> engineLastUse{};
  uint32_t nextCommandId = 1;
};

} // namespace npux::versap
