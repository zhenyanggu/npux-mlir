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
  // DESC2[60]: retain the INT32 postprocess result in the opposite ACC bank
  // for a following accumulation slice.
  bool writePartial = false;
  bool relu = false;
  SaOperation operation = SaOperation::Gemm;
  // CONV reuses SA_COMPUTE. Its dimensions are IFM H/W/C and OFM C.
  uint16_t d3 = 0;
  uint8_t kernelShapeM1 = 0;
  uint8_t strideM1 = 0;
  uint8_t dilationM1 = 0;
  uint8_t paddingLeft = 0;
  uint8_t paddingRight = 0;
  uint8_t paddingTop = 0;
  uint8_t paddingBottom = 0;
};

struct GemvConfig {
  // GEMV writes 16-bit BF16 elements to the selected O bank in the current
  // npu_top configuration. m is the output-vector length and k is the
  // matrix-vector reduction width.
  uint16_t m;
  uint16_t k;
  uint16_t activationGroupStrideBytes;
  uint16_t activationScaleBase;
  uint16_t aBase;
  uint16_t wBase;
  uint16_t oBase;
  uint16_t activationScale2Base;
  uint16_t cacheCellIndex;
  uint16_t scaleMetadataWord;
  uint16_t resaddBase;
  uint8_t metadataBank;
  uint8_t aBank;
  uint8_t wBank;
  uint8_t oBank;
  uint8_t mode = 0;
  uint8_t groupCountM1 = 0;
  bool kvColumnScale = false;
  bool pvProbabilityQ24 = false;
  bool unitWeightScale = false;
  bool activationScale = false;
  bool activationScale2 = false;
  bool preloadAccumulator = false;
  uint8_t preloadAccumulatorId = 0;
  uint8_t preloadAccumulatorRow = 0;
  uint16_t preloadAccumulatorData = 0;
  bool resadd = false;
};

enum class VpuOpcode : uint8_t {
  Load = 0x00, Store = 0x01, Alu = 0x02, Shift = 0x03, Mac = 0x05,
  FpAlu = 0x06, FpMac = 0x07, Sfu = 0x09, ReduceInt = 0x0a,
  ReduceFp = 0x0b, Conv = 0x0c, Move = 0x0d, Mask = 0x0e,
  Special = 0x10,
};

enum class VpuPrecision : uint8_t { Int8 = 0, Bf16 = 1, Fp32 = 2, Fp16 = 3 };

// VPU SPECIAL functions implemented by the current VersaEdge RTL. All VPU
// tensor traffic is between the two physical O banks.
enum class VpuSpecialFunction : uint8_t {
  RmsNorm = 0,
  LayerNorm = 1,
  Softmax = 2,
  Gelu = 3,
  Transpose = 4,
  PoolMax = 5,
  Sigmoid = 6,
};

struct VpuConfig {
  VpuSpecialFunction function;
  // Logical tensor dimensions. RTL encodes both as minus one in DESC0.
  uint16_t rows;
  uint16_t columns;
  uint8_t sourceSelect = 0;
  uint8_t destinationSelect = 0;
  VpuPrecision sourcePrecision = VpuPrecision::Bf16;
  VpuPrecision destinationPrecision = VpuPrecision::Bf16;
  uint16_t sourceAddress = 0;
  uint16_t destinationAddress = 0;
  // Used only when a BF16 SPECIAL function quantizes its output to INT8.
  uint32_t outputInverseScaleQ8_24 = 0;
};

llvm::Expected<EncodedDescriptor> encodeMvinA(const MvinAConfig &config);
llvm::Expected<EncodedDescriptor> encodeMvinW(const MvinWConfig &config);
llvm::Expected<EncodedDescriptor> encodeAuxMvin(const AuxMvinConfig &config);
llvm::Expected<EncodedDescriptor> encodeMvout(const MvoutConfig &config);
llvm::Expected<EncodedDescriptor> encodeGemm(const GemmConfig &config);
llvm::Expected<EncodedDescriptor> encodeGemv(const GemvConfig &config);
llvm::Expected<EncodedDescriptor> encodeVpu(const VpuConfig &config);

} // namespace npux::versap
