// SPDX-License-Identifier: Apache-2.0

#include "src/Conversion/NpuToLLVM/VersaPDescriptor.hpp"

#include <utility>

#include "llvm/ADT/StringRef.h"

namespace npux::versap {
namespace {

llvm::Error invalid(llvm::StringRef detail) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(), "Versa-P descriptor: %s",
      detail.str().c_str());
}

bool isAligned(uint32_t value, uint32_t alignment) {
  return value % alignment == 0;
}

uint64_t ceilDiv(uint64_t value, uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

bool fitsBank(uint32_t base, uint64_t words) {
  return base < kLocalBankWords && words <= kLocalBankWords - base;
}

llvm::Error validateDramBase(uint32_t dramBase) {
  if (!isAligned(dramBase, kLocalWordBytes))
    return invalid("DRAM base must be 32-byte aligned");
  return llvm::Error::success();
}

llvm::Error validateDmaShape(uint16_t rows, uint16_t bytes) {
  if (rows == 0 || bytes == 0)
    return invalid("row count and row bytes must be nonzero");
  if (ceilDiv(bytes, kLocalWordBytes) * rows > kLocalBankWords)
    return invalid("DMA transfer exceeds one local bank");
  return llvm::Error::success();
}

llvm::Error validateBank(
    uint8_t bank, uint8_t bankCount, llvm::StringRef name) {
  if (bank >= bankCount)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "Versa-P descriptor: %s bank is out of range", name.str().c_str());
  return llvm::Error::success();
}

} // namespace

llvm::Expected<EncodedDescriptor> encodeMvinA(const MvinAConfig &config) {
  if (auto error = validateDramBase(config.dramBase))
    return std::move(error);
  if (config.dramRowStrideBytes < config.colCount ||
      !isAligned(config.dramRowStrideBytes, kLocalWordBytes))
    return invalid("A DMA stride must be 32-byte aligned and cover one row");
  if (auto error = validateDmaShape(config.rowCount, config.colCount))
    return std::move(error);
  if (auto error = validateBank(config.bank, kInputBankCount, "A"))
    return std::move(error);

  const uint64_t desc0 = static_cast<uint64_t>(config.dramBase) |
                         (static_cast<uint64_t>(config.dramRowStrideBytes) <<
                             32);
  const uint64_t desc1 = static_cast<uint64_t>(config.rowCount) |
                         (static_cast<uint64_t>(config.colCount) << 16) |
                         (static_cast<uint64_t>(config.u8Minus128) << 32) |
                         (uint64_t{1} << 33) |
                         (static_cast<uint64_t>(config.bank) << 34);
  return EncodedDescriptor{desc0, desc1, 0};
}

llvm::Expected<EncodedDescriptor> encodeMvinW(const MvinWConfig &config) {
  if (auto error = validateDramBase(config.dramBase))
    return std::move(error);
  if (auto error = validateDmaShape(config.rowCount, config.colCount))
    return std::move(error);
  const uint64_t requiredSpanBytes =
      static_cast<uint64_t>(config.rowCount) *
      ceilDiv(config.colCount, kLocalWordBytes) *
      kLocalWordBytes;
  if (config.dramSpanBytes != 0 &&
      (config.dramSpanBytes < requiredSpanBytes ||
          !isAligned(config.dramSpanBytes, kLocalWordBytes)))
    return invalid("W DMA span must be zero or cover the aligned transfer");
  if (auto error = validateBank(config.bank, kInputBankCount, "W"))
    return std::move(error);

  const uint64_t desc0 = static_cast<uint64_t>(config.dramBase) |
                         (static_cast<uint64_t>(config.dramSpanBytes) << 32);
  const uint64_t desc1 = static_cast<uint64_t>(config.rowCount) |
                         (static_cast<uint64_t>(config.colCount) << 16) |
                         (static_cast<uint64_t>(config.bank) << 32) |
                         (uint64_t{1} << 33);
  return EncodedDescriptor{desc0, desc1, 0};
}

llvm::Expected<EncodedDescriptor> encodeAuxMvin(const AuxMvinConfig &config) {
  if (!isAligned(config.dramBase, kLocalWordBytes))
    return invalid("AUX_MVIN DRAM base must be 32-byte aligned");
  if (config.auxTypeOrOBank > 3)
    return invalid("AUX_MVIN aux type or O bank exceeds its 2-bit field");
  if (config.destinationIsOBank) {
    const uint16_t rowCount = static_cast<uint16_t>(config.value);
    const uint16_t rowBytes = static_cast<uint16_t>(config.value >> 16);
    if (config.offsetBytes < rowBytes ||
        !isAligned(config.offsetBytes, kLocalWordBytes))
      return invalid("AUX_MVIN O stride must be 32-byte aligned and cover one row");
    if (rowCount == 0 || rowBytes == 0)
      return invalid("AUX_MVIN O row count and row bytes must be nonzero");
    if (ceilDiv(rowBytes, kLocalWordBytes) * rowCount > kLocalBankWords)
      return invalid("AUX_MVIN O transfer exceeds one local bank");
    if (config.auxTypeOrOBank != 0)
      return invalid("AUX_MVIN O path only supports physical Res bank 0");
  } else {
    if (config.value == 0 || config.value > kMetadataWords * kLocalWordBytes ||
        config.offsetBytes > kMetadataWords * kLocalWordBytes - config.value)
      return invalid("AUX_MVIN metadata range exceeds one metadata bank");
    if (auto error = validateBank(config.metadataBank, kMetadataBankCount,
            "AUX metadata"))
      return std::move(error);
  }

  const uint64_t desc0 = static_cast<uint64_t>(config.dramBase) |
                         (static_cast<uint64_t>(config.offsetBytes) << 32);
  const uint64_t desc1 = static_cast<uint64_t>(config.value) |
                         (static_cast<uint64_t>(config.auxTypeOrOBank) << 32) |
                         (uint64_t{1} << 34) |
                         (static_cast<uint64_t>(config.destinationIsOBank) <<
                             35) |
                         (static_cast<uint64_t>(config.metadataBank) << 36);
  return EncodedDescriptor{desc0, desc1, 0};
}

llvm::Expected<EncodedDescriptor> encodeMvout(const MvoutConfig &config) {
  if (auto error = validateDramBase(config.dramBase))
    return std::move(error);
  if (config.dramRowStrideBytes != 0 &&
      (config.dramRowStrideBytes < config.rowBytes ||
          !isAligned(config.dramRowStrideBytes, kLocalWordBytes)))
    return invalid("MVOUT stride must be zero or a 32-byte aligned row span");
  if (auto error = validateDmaShape(config.rowCount, config.rowBytes))
    return std::move(error);
  if (auto error = validateBank(config.bank, kOutputBankCount, "O"))
    return std::move(error);
  if (config.qkMode &&
      config.metadataBaseByte >= kMetadataWords * kLocalWordBytes)
    return invalid("QK metadata base exceeds the metadata bank");
  if (auto error = validateBank(config.metadataBank, kMetadataBankCount,
          "metadata"))
    return std::move(error);

  const uint64_t desc0 = static_cast<uint64_t>(config.dramBase) |
                         (static_cast<uint64_t>(config.dramRowStrideBytes) <<
                             32);
  const uint64_t desc1 = static_cast<uint64_t>(config.rowCount) |
                         (static_cast<uint64_t>(config.rowBytes) << 16) |
                         (static_cast<uint64_t>(config.metadataBaseByte) <<
                             32) |
                         (static_cast<uint64_t>(config.qkMode) << 48) |
                         (static_cast<uint64_t>(config.metadataBank) << 49) |
                         (uint64_t{1} << 50) |
                         (static_cast<uint64_t>(config.bank) << 51) |
                         (static_cast<uint64_t>(config.oIsChange) << 53) |
                         (static_cast<uint64_t>(config.qkMaskEnable) << 54);
  return EncodedDescriptor{desc0, desc1, 0};
}

llvm::Expected<EncodedDescriptor> encodeGemm(const GemmConfig &config) {
  if (config.m == 0 || config.n == 0 || config.k == 0 ||
      config.k > kMaxSaK)
    return invalid("GEMM M, N, and K must be nonzero and K must not exceed 4096");
  if (static_cast<uint8_t>(config.outputMode) > static_cast<uint8_t>(OutputMode::Fp32) ||
      static_cast<uint8_t>(config.scaleMode) > static_cast<uint8_t>(ScaleMode::PerChannel) ||
      static_cast<uint8_t>(config.operation) > static_cast<uint8_t>(SaOperation::AttentionPv))
    return invalid("GEMM output mode is not defined by the descriptor ABI");
  if (config.operation == SaOperation::AttentionQk &&
      (config.accumulate || config.resadd || config.outputMode != OutputMode::RawInt32))
    return invalid("attention QK requires raw INT32 output without accumulate or resadd");
  if (config.operation == SaOperation::AttentionPv &&
      (config.accumulate || config.bias || config.outputMode != OutputMode::TensorInt8))
    return invalid("attention PV requires INT8 output without accumulate or bias");
  if (config.accumulate && config.bias)
    return invalid("GEMM accumulate and bias cannot be enabled together");
  if (config.biasAddr >= kMetadataWords)
    return invalid("bias metadata address exceeds the normal metadata bank");
  const bool needsScale = config.outputMode == OutputMode::TensorInt8 ||
                          config.outputMode == OutputMode::Fp32;
  if ((config.outputMode == OutputMode::RawInt32 || config.outputMode == OutputMode::Bf16) &&
      (config.scale != 0 || config.scaleMode != ScaleMode::None))
    return invalid("raw INT32 and BF16 GEMM must use scale_mode=none with zero scale");
  if (needsScale && config.scaleMode == ScaleMode::None)
    return invalid("quantized or FP32 GEMM requires a scale mode");
  if (config.scaleMode == ScaleMode::PerTensor && config.scale > 511)
    return invalid("per-tensor scale field exceeds 9 bits");
  if (config.scaleMode == ScaleMode::PerChannel &&
      (config.scale >= kMetadataWords ||
          config.scale + ceilDiv(config.n, 8) > kMetadataWords))
    return invalid("per-channel INT8 scales exceed the normal metadata bank");

  if (auto error = validateBank(config.aBank, kInputBankCount, "A"))
    return std::move(error);
  if (auto error = validateBank(config.wBank, kInputBankCount, "W"))
    return std::move(error);
  if (auto error = validateBank(config.oBank, kOutputBankCount, "O"))
    return std::move(error);
  if (auto error = validateBank(config.accBank, kOutputBankCount, "ACC"))
    return std::move(error);
  if (config.resaddBank != 0)
    return invalid("GEMM ResAdd bank field must select physical Res bank 0");
  if (auto error =
          validateBank(config.metadataBank, kMetadataBankCount, "metadata"))
    return std::move(error);

  const uint64_t inputWords = ceilDiv(config.k, kLocalWordBytes);
  const uint64_t outputWords = ceilDiv(config.n,
      config.outputMode == OutputMode::TensorInt8 ? 32
      : config.outputMode == OutputMode::Bf16 ? 16 : 8);
  if (!fitsBank(config.aBase, config.m * inputWords) ||
      !fitsBank(config.wBase, config.n * inputWords) ||
      !fitsBank(config.oBase, config.m * outputWords))
    return invalid("GEMM A, W, or O range exceeds its local bank");
  if (config.accumulate &&
      (!fitsBank(config.accBase, config.m * ceilDiv(config.n, 8)) ||
          config.accBank == config.oBank))
    return invalid("GEMM accumulator must fit and use a different O bank");
  if (config.resadd &&
      !fitsBank(config.resaddBase, config.m * ceilDiv(config.n, 8)))
    return invalid("GEMM residual must fit in the physical Res bank");

  const uint64_t desc0 = static_cast<uint64_t>(config.m) |
                         (static_cast<uint64_t>(config.n) << 16) |
                         (static_cast<uint64_t>(config.k) << 32);
  const uint64_t desc1 = static_cast<uint64_t>(config.aBase) |
                         (static_cast<uint64_t>(config.wBase) << 12) |
                         (static_cast<uint64_t>(config.oBase) << 24) |
                         (static_cast<uint64_t>(config.accBase) << 36) |
                         (static_cast<uint64_t>(config.resaddBase) << 48);
  const uint64_t desc2 =
      static_cast<uint64_t>(config.operation) |
      (static_cast<uint64_t>(config.accumulate) << 2) |
      (static_cast<uint64_t>(config.bias) << 3) |
      (static_cast<uint64_t>(config.resadd) << 4) |
      (static_cast<uint64_t>(config.outputMode) << 5) |
      (static_cast<uint64_t>(config.biasAddr) << 18) |
      (static_cast<uint64_t>(config.scale) << 27) | (uint64_t{1} << 45) |
      (static_cast<uint64_t>(config.aBank) << 46) |
      (static_cast<uint64_t>(config.wBank) << 47) |
      (static_cast<uint64_t>(config.oBank) << 48) |
      (static_cast<uint64_t>(config.accBank) << 50) |
      (static_cast<uint64_t>(config.resaddBank) << 52) |
      (static_cast<uint64_t>(config.scaleMode) << 54) |
      (static_cast<uint64_t>(config.oIsChange) << 56) |
      (static_cast<uint64_t>(config.accIsChange) << 57) |
      (static_cast<uint64_t>(config.resaddIsChange) << 58) |
      (static_cast<uint64_t>(config.metadataIsChange) << 59) |
      (static_cast<uint64_t>(config.relu) << 61) |
      (static_cast<uint64_t>(config.aIsChange) << 62) |
      (static_cast<uint64_t>(config.wIsChange) << 63);
  return EncodedDescriptor{desc0, desc1, desc2};
}

} // namespace npux::versap
