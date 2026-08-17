// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llvm/Support/Error.h"

namespace npux::versap {

constexpr uint32_t kLocalWordBytes = 32;
constexpr uint32_t kLocalBankWords = 2048;
constexpr uint32_t kMetadataWords = 512;
constexpr uint32_t kMaxSaK = 4096;
constexpr uint8_t kInputBankCount = 2;
constexpr uint8_t kOutputBankCount = 2;
constexpr uint8_t kResaddBankCount = 1;
constexpr uint8_t kMetadataBankCount = 2;

struct EncodedDescriptor {
  uint64_t desc0 = 0;
  uint64_t desc1 = 0;
  uint64_t desc2 = 0;
};

struct MvinAConfig {
  uint32_t dramBase;
  uint32_t dramRowStrideBytes;
  uint16_t rowCount;
  uint16_t colCount;
  bool u8Minus128;
  uint8_t bank;
};

struct MvinWConfig {
  uint32_t dramBase;
  uint32_t dramSpanBytes;
  uint16_t rowCount;
  uint16_t colCount;
  uint8_t bank;
};

// AUX_MVIN writes either normal metadata or one O bank. For metadata,
// offsetBytes and value are the local byte offset and byte count. For O,
// offsetBytes is the DRAM row stride and value packs rowCount in bits [15:0]
// and rowBytes in bits [31:16], as defined by the descriptor ABI.
struct AuxMvinConfig {
  uint32_t dramBase;
  uint32_t offsetBytes;
  uint32_t value;
  uint8_t auxTypeOrOBank;
  bool destinationIsOBank;
  uint8_t metadataBank = 0;
};

struct MvoutConfig {
  uint32_t dramBase;
  uint32_t dramRowStrideBytes;
  uint16_t rowCount;
  uint16_t rowBytes;
  uint8_t bank;
  uint16_t metadataBaseByte = 0;
  bool qkMode = false;
  uint8_t metadataBank = 0;
  bool oIsChange = false;
  bool qkMaskEnable = false;
};

enum class OutputMode : uint8_t {
  RawInt32 = 0,
  TensorInt8 = 1,
  Bf16 = 2,
  Fp32 = 3,
};

enum class ScaleMode : uint8_t { None = 0, PerTensor = 1, PerChannel = 2 };
enum class SaOperation : uint8_t {
  Gemm = 0,
  Conv = 1,
  AttentionQk = 2,
  AttentionPv = 3,
};

struct GemmConfig {
  uint16_t m;
  uint16_t n;
  uint16_t k;
  uint16_t aBase;
  uint16_t wBase;
  uint16_t oBase;
  uint16_t accBase;
  uint16_t resaddBase;
  uint16_t biasAddr;
  uint16_t scale;
  uint8_t aBank;
  uint8_t wBank;
  uint8_t oBank;
  uint8_t accBank;
  uint8_t resaddBank;
  uint8_t metadataBank;
  OutputMode outputMode;
  ScaleMode scaleMode;
  bool accumulate;
  bool bias;
  bool resadd;
  bool aIsChange = false;
  bool wIsChange = false;
  bool oIsChange = false;
  bool accIsChange = false;
  bool resaddIsChange = false;
  bool metadataIsChange = false;
  bool relu = false;
  SaOperation operation = SaOperation::Gemm;
};

llvm::Expected<EncodedDescriptor> encodeMvinA(const MvinAConfig &config);
llvm::Expected<EncodedDescriptor> encodeMvinW(const MvinWConfig &config);
llvm::Expected<EncodedDescriptor> encodeAuxMvin(const AuxMvinConfig &config);
llvm::Expected<EncodedDescriptor> encodeMvout(const MvoutConfig &config);
llvm::Expected<EncodedDescriptor> encodeGemm(const GemmConfig &config);

} // namespace npux::versap
