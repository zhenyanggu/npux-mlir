// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iostream>
#include <string>

#include "src/Conversion/NpuToLLVM/VersaPDescriptor.hpp"

#include "llvm/Support/Error.h"

namespace {

using npux::versap::AuxMvinConfig;
using npux::versap::EncodedDescriptor;
using npux::versap::GemmConfig;
using npux::versap::MvinAConfig;
using npux::versap::MvinWConfig;
using npux::versap::MvoutConfig;
using npux::versap::OutputMode;
using npux::versap::ScaleMode;
using npux::versap::encodeAuxMvin;
using npux::versap::encodeGemm;
using npux::versap::encodeMvinA;
using npux::versap::encodeMvinW;
using npux::versap::encodeMvout;

bool fail(const std::string &message) {
  std::cerr << "VersaPDescriptorTest: " << message << '\n';
  return false;
}

bool expectDescriptor(llvm::Expected<EncodedDescriptor> result,
    uint64_t expectedDesc0, uint64_t expectedDesc1, uint64_t expectedDesc2) {
  if (!result)
    return fail("unexpected error: " + llvm::toString(result.takeError()));
  if (result->desc0 != expectedDesc0 || result->desc1 != expectedDesc1 ||
      result->desc2 != expectedDesc2)
    return fail("descriptor bits differ from the ABI golden value");
  return true;
}

bool expectError(
    llvm::Expected<EncodedDescriptor> result, const std::string &needle) {
  if (result)
    return fail("expected an encoder error");
  const std::string message = llvm::toString(result.takeError());
  if (message.find(needle) == std::string::npos)
    return fail("error message does not contain '" + needle + "': " + message);
  return true;
}

bool testMvinW() {
  if (!expectDescriptor(encodeMvinW(MvinWConfig{
            0x3000, 0x80, 2, 33, 1}),
          0x0000008000003000ULL, 0x0000000300210002ULL, 0))
    return false;
  return expectError(
      encodeMvinW(MvinWConfig{0x3000, 0x60, 2, 33, 1}), "W DMA span");
}

bool testMvinA() {
  if (!expectDescriptor(encodeMvinA(MvinAConfig{
            0x1000, 0x40, 2, 33, true, 1}),
          0x0000004000001000ULL, 0x0000000700210002ULL, 0))
    return false;
  return expectError(encodeMvinA(MvinAConfig{
                         0x1000, 33, 2, 33, false, 0}),
      "A DMA stride");
}

bool testAuxMvin() {
  if (!expectDescriptor(encodeAuxMvin(AuxMvinConfig{
            0x1000, 0x20, 0x40, 0, false, 1}),
          0x0000002000001000ULL, 0x0000001400000040ULL, 0))
    return false;
  if (!expectDescriptor(encodeAuxMvin(AuxMvinConfig{
            0x2000, 0x80, 0x00400002, 0, true, 0}),
          0x0000008000002000ULL, 0x0000000C00400002ULL, 0))
    return false;
  if (!expectError(encodeAuxMvin(AuxMvinConfig{
            0x1004, 0x20, 0x40, 0, false, 0}),
          "AUX_MVIN"))
    return false;
  if (!expectError(encodeAuxMvin(AuxMvinConfig{
            0x1000, 0, 0x40, 4, false, 0}),
          "AUX_MVIN"))
    return false;
  if (!expectError(encodeAuxMvin(AuxMvinConfig{
            0x1000, 0, 0, 0, false, 0}),
          "AUX_MVIN"))
    return false;
  if (!expectError(encodeAuxMvin(AuxMvinConfig{
            0x1000, 16352, 64, 0, false, 0}),
          "AUX_MVIN"))
    return false;
  if (!expectError(encodeAuxMvin(AuxMvinConfig{
            0x2000, 0x40, 0x00800002, 3, true, 0}),
          "AUX_MVIN"))
    return false;
  return expectError(encodeAuxMvin(AuxMvinConfig{
          0x2000, 0x80, 0x00400000, 1, true, 0}),
      "physical Res bank");
}

bool testMvout() {
  if (!expectDescriptor(encodeMvout(MvoutConfig{
            0x5000, 0x80, 2, 64, 0, 0, false, 0, true, false}),
          0x0000008000005000ULL, 0x0024000000400002ULL, 0))
    return false;
  return expectError(encodeMvout(MvoutConfig{
                          0x5000, 0x80, 2, 64, 2, 0, false, 0, false, false}),
      "O bank");
}

bool testGemm() {
  const GemmConfig config{2, 33, 64, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, OutputMode::TensorInt8, ScaleMode::PerTensor,
      false, false, false,
      true, true, true, false, false, false};
  if (!expectDescriptor(encodeGemm(config), 0x0000004000210002ULL, 0,
          0xC140200000000020ULL))
    return false;

  GemmConfig relu = config;
  relu.relu = true;
  if (!expectDescriptor(encodeGemm(relu), 0x0000004000210002ULL, 0,
          0xE140200000000020ULL))
    return false;

  GemmConfig invalid = config;
  invalid.outputMode = OutputMode::Bf16;
  invalid.scaleMode = ScaleMode::None;
  if (!expectDescriptor(encodeGemm(invalid), 0x0000004000210002ULL, 0,
          0xC100000000000040ULL))
    return false;
  invalid = config;
  invalid.k = 4097;
  if (!expectError(encodeGemm(invalid), "GEMM M, N, and K"))
    return false;
  invalid = config;
  invalid.resaddBank = 1;
  return expectError(encodeGemm(invalid), "physical Res bank");
}

} // namespace

int main() {
  return testMvinW() && testMvinA() && testAuxMvin() && testMvout() &&
                 testGemm()
             ? 0
             : 1;
}
