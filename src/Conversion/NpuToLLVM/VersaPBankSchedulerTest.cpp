// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "src/Conversion/NpuToLLVM/VersaPBankScheduler.hpp"
#include "src/Conversion/NpuToLLVM/VersaPStaticEligibility.hpp"

#include "llvm/Support/Error.h"

namespace {

using npux::versap::DescriptorAddressSource;
using npux::versap::AttentionHeadRequest;
using npux::versap::GemmLocalAddresses;
using npux::versap::GemmTilingProblem;
using npux::versap::MvinAConfig;
using npux::versap::MvinWConfig;
using npux::versap::MvoutConfig;
using npux::versap::ScheduledCommand;
using npux::versap::ScheduledCommandKind;
using npux::versap::TileRequest;
using npux::versap::VersaPBankScheduler;

class MockRuntime {
public:
  explicit MockRuntime(uint32_t failCommand = 0) : failCommand(failCommand) {}

  bool submit(const ScheduledCommand &command) {
    if (error)
      return false;
    for (uint32_t dependency : command.dependencies)
      if (!completed.count(dependency))
        return false;
    submitted.push_back(command.id);
    if (command.id == failCommand) {
      error = true;
      return false;
    }
    completed.insert(command.id);
    return true;
  }

  void clearStatus(uint32_t commandId) {
    completed.erase(commandId);
    cleared.insert(commandId);
  }

  void globalClear() {
    completed.clear();
    error = false;
    globalClearCalled = true;
  }

  bool hasCompleted(uint32_t commandId) const { return completed.count(commandId); }
  bool wasCleared(uint32_t commandId) const { return cleared.count(commandId); }
  bool didGlobalClear() const { return globalClearCalled; }
  const std::vector<uint32_t> &submissions() const { return submitted; }

private:
  uint32_t failCommand;
  bool error = false;
  bool globalClearCalled = false;
  std::unordered_set<uint32_t> completed;
  std::unordered_set<uint32_t> cleared;
  std::vector<uint32_t> submitted;
};
bool fail(const std::string &message) {
  std::cerr << "VersaPBankSchedulerTest: " << message << '\n';
  return false;
}

TileRequest makeRequest() {
  TileRequest request{2, 33, 64};
  request.quantization.nextConsumesInt8 = true;
  request.aDma = MvinAConfig{0x1000, 0x40, 2, 64, false, 0};
  request.wDma = MvinWConfig{0x2000, 0x840, 33, 64, 0};
  request.outputDma = MvoutConfig{0x4000, 0x40, 2, 33, 0};
  request.localAddresses = GemmLocalAddresses{0, 0, 0};
  return request;
}

bool hasDependency(const ScheduledCommand &command, uint32_t id) {
  for (uint32_t dependency : command.dependencies)
    if (dependency == id)
      return true;
  return false;
}

bool hasBankUse(const ScheduledCommand &command,
    npux::versap::BankResource resource, uint8_t bank) {
  for (const auto &use : command.reads)
    if (use.resource == resource && use.bank == bank)
      return true;
  for (const auto &use : command.writes)
    if (use.resource == resource && use.bank == bank)
      return true;
  return false;
}

bool hasBankRead(const ScheduledCommand &command,
    npux::versap::BankResource resource, uint8_t bank) {
  for (const auto &use : command.reads)
    if (use.resource == resource && use.bank == bank)
      return true;
  return false;
}

bool hasBankWrite(const ScheduledCommand &command,
    npux::versap::BankResource resource, uint8_t bank) {
  for (const auto &use : command.writes)
    if (use.resource == resource && use.bank == bank)
      return true;
  return false;
}

bool testSingleTileDescriptorsAndDag() {
  VersaPBankScheduler scheduler;
  auto schedule = scheduler.scheduleGemmTile(makeRequest());
  if (!schedule)
    return fail("unexpected error: " + llvm::toString(schedule.takeError()));
  if (schedule->commands.size() != 4)
    return fail("single tile should emit A/W/GEMM/MVOUT commands");
  const auto &a = schedule->commands[0];
  const auto &w = schedule->commands[1];
  const auto &gemm = schedule->commands[2];
  const auto &output = schedule->commands[3];
  if (a.kind != ScheduledCommandKind::LoadA || a.descriptorRegisterOffset != 0 ||
      a.statusRegisterOffset != 0x10 || a.descriptor.desc1 != 0x0000000200400002ULL)
    return fail("MVIN_A descriptor or register mapping differs from ABI");
  if (w.kind != ScheduledCommandKind::LoadW || w.descriptorRegisterOffset != 0x20 ||
      w.statusRegisterOffset != 0x30 || w.descriptor.desc1 != 0x0000000200400021ULL)
    return fail("MVIN_W descriptor or register mapping differs from ABI");
  if (a.addressPatches.size() != 1 ||
      a.addressPatches[0].source != DescriptorAddressSource::A ||
      a.addressPatches[0].descriptorWord != 0 ||
      a.addressPatches[0].bitOffset != 0 || a.addressPatches[0].bitWidth != 32 ||
      output.addressPatches.size() != 1 ||
      output.addressPatches[0].source != DescriptorAddressSource::Output)
    return fail("DMA descriptor template is missing runtime-address patch metadata");
  if (!a.dependencies.empty() || !w.dependencies.empty())
    return fail("independent A/W loads must not have a false engine dependency");
  if (gemm.kind != ScheduledCommandKind::Gemm || gemm.descriptorRegisterOffset != 0x60 ||
      gemm.statusRegisterOffset != 0x78 || !hasDependency(gemm, a.id) ||
      !hasDependency(gemm, w.id))
    return fail("GEMM must wait for both load completions");
  if (output.kind != ScheduledCommandKind::StoreO ||
      output.descriptorRegisterOffset != 0x80 || output.statusRegisterOffset != 0x90 ||
      !hasDependency(output, gemm.id))
    return fail("MVOUT must wait for GEMM completion");
  return true;
}

bool testOutputBankRotation() {
  VersaPBankScheduler scheduler;
  std::array<bool, 2> seen{};
  for (unsigned i = 0; i < 2; ++i) {
    auto schedule = scheduler.scheduleGemmTile(makeRequest());
    if (!schedule)
      return fail("unexpected bank-rotation error: " +
                  llvm::toString(schedule.takeError()));
    if (schedule->oBank >= seen.size())
      return fail("O bank is outside O0/O1");
    seen[schedule->oBank] = true;
  }
  for (bool selected : seen)
    if (!selected)
      return fail("scheduler failed to consider both O0/O1 banks");
  return true;
}

bool testRejectsUnknownDma() {
  VersaPBankScheduler scheduler;
  TileRequest request{2, 33, 64};
  request.localAddresses = GemmLocalAddresses{0, 0, 0};
  auto schedule = scheduler.scheduleGemmTile(request);
  if (schedule)
    return fail("scheduler accepted a tile without explicit DMA information");
  return llvm::toString(schedule.takeError()).find("explicit A/W DMA") !=
                 std::string::npos ||
             fail("missing-DMA error is not actionable");
}

bool testOutputModesAndMetadata() {
  struct ModeCase {
    npux::versap::OutputMode expected;
    bool raw;
    bool fp32;
    bool bf16;
    bool perChannel;
  };
  const std::array<ModeCase, 5> cases = {{{npux::versap::OutputMode::RawInt32,
      true, false, false, false}, {npux::versap::OutputMode::TensorInt8, false,
      false, false, false}, {npux::versap::OutputMode::TensorInt8, false,
      false, false, true}, {npux::versap::OutputMode::Fp32, false, true,
      false, false}, {npux::versap::OutputMode::Bf16, false, false,
      true, false}}};
  for (const ModeCase &mode : cases) {
    VersaPBankScheduler scheduler;
    TileRequest request = makeRequest();
    request.quantization.nextConsumesInt8 =
        !mode.raw && !mode.fp32 && !mode.bf16;
    request.quantization.requireRawInt32 = mode.raw;
    request.quantization.requireFp32 = mode.fp32;
    request.quantization.nextConsumesBf16 = mode.bf16;
    request.quantization.perChannelScaleAvailable = mode.perChannel;
    if (!mode.raw && !mode.bf16 && !mode.perChannel)
      request.localAddresses->scaleMetadataWord = 128;
    if (mode.raw || mode.fp32) {
      request.outputDma->dramRowStrideBytes = 0xA0;
      request.outputDma->rowBytes = 132;
    }
    if (mode.bf16) {
      request.outputDma->dramRowStrideBytes = 0x60;
      request.outputDma->rowBytes = 66;
    }
    if (mode.perChannel) {
      request.quantization.metadataWords = 5;
      request.metadataDma = npux::versap::AuxMvinConfig{
          0x3000, 0, 160, 0, false, 0};
    }
    auto schedule = scheduler.scheduleGemmTile(request);
    if (!schedule)
      return fail("unexpected output-mode error: " +
                  llvm::toString(schedule.takeError()));
    if (schedule->outputMode != mode.expected)
      return fail("scheduler selected an unexpected output mode");
    const ScheduledCommand *gemm = nullptr;
    bool hasMetadata = false;
    for (const ScheduledCommand &command : schedule->commands) {
      if (command.kind == ScheduledCommandKind::Gemm)
        gemm = &command;
      hasMetadata |= command.kind == ScheduledCommandKind::LoadMetadata;
    }
    if (!gemm || ((gemm->descriptor.desc2 >> 5) & 3) !=
                     static_cast<uint64_t>(mode.expected))
      return fail("GEMM output-mode bits differ from the ABI");
    const uint64_t expectedScaleMode = mode.perChannel ? 2 :
        (!mode.raw && !mode.fp32 && !mode.bf16 ? 1 : mode.fp32 ? 1 : 0);
    if (((gemm->descriptor.desc2 >> 54) & 3) != expectedScaleMode)
      return fail("GEMM scale-mode bits differ from the ABI");
    const uint64_t expectedScale = mode.perChannel ? 0 :
        (!mode.raw && !mode.bf16 ? 128 : 0);
    if (((gemm->descriptor.desc2 >> 27) & 0x1ff) != expectedScale)
      return fail("GEMM scale address bits differ from the ABI");
    if (hasMetadata != mode.perChannel)
      return fail("metadata DMA does not match per-channel mode");
  }
  return true;
}

bool testAccumulateAndResadd() {
  VersaPBankScheduler scheduler;
  TileRequest request = makeRequest();
  request.quantization.nextConsumesInt8 = false;
  request.quantization.requireRawInt32 = true;
  request.outputDma->dramRowStrideBytes = 0xA0;
  request.outputDma->rowBytes = 132;
  request.storeToDram = false;
  request.writePartial = true;
  auto partial = scheduler.scheduleGemmTile(request);
  if (!partial || partial->commands.size() != 3 ||
      partial->partialOutputBank >= 2)
    return fail("partial GEMM must retain its result in an ACC ping-pong bank");
  const ScheduledCommand &partialGemm = partial->commands.back();
  const uint8_t firstAccSelector =
      static_cast<uint8_t>((partialGemm.descriptor.desc2 >> 50) & 3);
  if ((partialGemm.descriptor.desc2 & (uint64_t{1} << 60)) == 0 ||
      firstAccSelector != partial->accumulatorBank ||
      partial->partialOutputBank != (firstAccSelector ^ 1) ||
      !hasBankWrite(partialGemm, npux::versap::BankResource::Accumulator,
          partial->partialOutputBank) ||
      hasBankWrite(partialGemm, npux::versap::BankResource::O,
          partial->oBank))
    return fail(
        "first partial GEMM does not target the opposite ACC bank");

  request.writePartial = true;
  request.needsAccumulate = true;
  auto body = scheduler.scheduleGemmTile(request);
  if (!body)
    return fail("scheduler rejected RTL-supported ACC accumulation: " +
                llvm::toString(body.takeError()));
  const ScheduledCommand *bodyGemm = nullptr;
  for (const ScheduledCommand &command : body->commands)
    if (command.kind == ScheduledCommandKind::Gemm)
      bodyGemm = &command;
  const uint8_t bodyAccSelector = bodyGemm
      ? static_cast<uint8_t>((bodyGemm->descriptor.desc2 >> 50) & 3)
      : 2;
  if (!bodyGemm ||
      (bodyGemm->descriptor.desc2 & (uint64_t{1} << 2)) == 0 ||
      (bodyGemm->descriptor.desc2 & (uint64_t{1} << 60)) == 0 ||
      bodyAccSelector != partial->partialOutputBank ||
      body->partialOutputBank != (bodyAccSelector ^ 1) ||
      !hasBankRead(*bodyGemm, npux::versap::BankResource::Accumulator,
          bodyAccSelector) ||
      !hasBankWrite(*bodyGemm, npux::versap::BankResource::Accumulator,
          body->partialOutputBank) ||
      !hasDependency(*bodyGemm, partialGemm.id))
    return fail("ACC body GEMM must read and ping-pong the partial result");

  request.writePartial = false;
  request.storeToDram = true;
  auto final = scheduler.scheduleGemmTile(request);
  if (!final || final->commands.size() != 4)
    return fail("ACC final GEMM must emit a final MVOUT");
  const ScheduledCommand *finalGemm = nullptr;
  for (const ScheduledCommand &command : final->commands)
    if (command.kind == ScheduledCommandKind::Gemm)
      finalGemm = &command;
  const uint8_t finalAccSelector = finalGemm
      ? static_cast<uint8_t>((finalGemm->descriptor.desc2 >> 50) & 3)
      : 2;
  if (!finalGemm || finalAccSelector != body->partialOutputBank ||
      !hasBankRead(*finalGemm, npux::versap::BankResource::Accumulator,
          finalAccSelector) ||
      !hasBankWrite(*finalGemm, npux::versap::BankResource::O,
          final->oBank) ||
      !hasDependency(*finalGemm, bodyGemm->id))
    return fail("ACC final GEMM must consume the last partial result");

  request.needsAccumulate = false;
  request.needsResadd = true;
  request.residualDma = npux::versap::AuxMvinConfig{0x7000, 0xA0,
      0x00840002, 0, true, 0};
  auto schedule = scheduler.scheduleGemmTile(request);
  if (!schedule)
    return fail("unexpected accumulate/resadd error: " +
                llvm::toString(schedule.takeError()));
  if (schedule->residualBank != 0)
    return fail("residual must use physical Res bank 0");
  const ScheduledCommand *residual = nullptr;
  const ScheduledCommand *gemm = nullptr;
  for (const ScheduledCommand &command : schedule->commands) {
    if (command.kind == ScheduledCommandKind::LoadResidual)
      residual = &command;
    if (command.kind == ScheduledCommandKind::Gemm)
      gemm = &command;
  }
  if (!residual || !gemm || !hasDependency(*gemm, residual->id))
    return fail("GEMM must wait for residual AUX_MVIN completion");
  return true;
}
bool testBiasAndMetadataPingPong() {
  VersaPBankScheduler scheduler;
  TileRequest request = makeRequest();
  request.needsBias = true;
  request.quantization.metadataWords = 1;
  request.metadataDma = npux::versap::AuxMvinConfig{
      0x3000, 0, 32, 0, false, 0};
  auto first = scheduler.scheduleGemmTile(request);
  if (!first)
    return fail("unexpected bias schedule error: " +
                llvm::toString(first.takeError()));
  auto second = scheduler.scheduleGemmTile(request);
  if (!second)
    return fail("unexpected metadata ping-pong error: " +
                llvm::toString(second.takeError()));
  if (first->metadataBank == second->metadataBank)
    return fail("metadata DMA must ping-pong across M0/M1");
  const ScheduledCommand *gemm = nullptr;
  const ScheduledCommand *metadata = nullptr;
  for (const ScheduledCommand &command : first->commands) {
    if (command.kind == ScheduledCommandKind::Gemm)
      gemm = &command;
    if (command.kind == ScheduledCommandKind::LoadMetadata)
      metadata = &command;
  }
  if (!gemm || !metadata || !hasDependency(*gemm, metadata->id) ||
      (gemm->descriptor.desc2 & (uint64_t{1} << 3)) == 0)
    return fail("bias GEMM must wait for metadata and set the ABI bias bit");
  return true;
}

bool testOptimalGemmTiling() {
  VersaPBankScheduler scheduler;
  GemmTilingProblem problem{128, 1024, 64};
  problem.quantization.nextConsumesInt8 = true;
  problem.quantization.perChannelScaleAvailable = true;
  auto plan = scheduler.planGemmTiles(problem);
  if (!plan)
    return fail("unexpected optimal-tiling error: " +
                llvm::toString(plan.takeError()));
  if (plan->m != 128 || plan->n != 512 || plan->k != 64 ||
      plan->mTileCount != 1 || plan->nTileCount != 2 ||
      plan->kTileCount != 1 || plan->metadataWords != 64)
    return fail("tiler did not maximize legal descriptor reuse and scale coverage");

  GemmTilingProblem internalTileCheck{64, 128, 64};
  internalTileCheck.quantization.nextConsumesInt8 = true;
  internalTileCheck.quantization.perChannelScaleAvailable = true;
  auto internalTilePlan = scheduler.planGemmTiles(internalTileCheck);
  if (!internalTilePlan || internalTilePlan->m != 64 ||
      internalTilePlan->n != 128 || internalTilePlan->descriptorTileCount != 1)
    return fail("tiler incorrectly treated the RTL 32x32 array as a compiler tile");
  return true;
}

bool testPerEnginePendingWindow() {
  VersaPBankScheduler scheduler;
  auto first = scheduler.scheduleGemmTile(makeRequest());
  auto second = scheduler.scheduleGemmTile(makeRequest());
  auto third = scheduler.scheduleGemmTile(makeRequest());
  if (!first || !second || !third)
    return fail("scheduler failed while filling the per-engine pending window");
  const ScheduledCommand &firstA = first->commands[0];
  const ScheduledCommand &secondA = second->commands[0];
  const ScheduledCommand &thirdA = third->commands[0];
  if (hasDependency(secondA, firstA.id))
    return fail("second same-engine command must use the RTL pending slot");
  if (!hasDependency(thirdA, firstA.id))
    return fail("third same-engine command must wait for the oldest command");
  return true;
}

bool testPackedBiasAndPerChannelScaleMetadata() {
  VersaPBankScheduler scheduler;
  TileRequest request = makeRequest();
  request.needsBias = true;
  request.quantization.nextConsumesInt8 = true;
  request.quantization.perChannelScaleAvailable = true;
  // N=33 uses five 32-byte bias words and five scale words. The per-channel
  // scale base is therefore metadata word five, immediately after bias.
  request.quantization.metadataWords = 10;
  request.metadataDma = npux::versap::AuxMvinConfig{
      0x3000, 0, 320, 0, false, 0};
  request.localAddresses->biasMetadataWord = 0;
  request.localAddresses->scaleMetadataWord = 5;
  auto schedule = scheduler.scheduleGemmTile(request);
  if (!schedule || schedule->commands.size() != 5)
    return fail("packed metadata must emit one AUX_MVIN before GEMM");
  const ScheduledCommand *metadata = nullptr;
  const ScheduledCommand *gemm = nullptr;
  for (const ScheduledCommand &command : schedule->commands) {
    if (command.kind == ScheduledCommandKind::LoadMetadata)
      metadata = &command;
    if (command.kind == ScheduledCommandKind::Gemm)
      gemm = &command;
  }
  if (!metadata || !gemm || !hasDependency(*gemm, metadata->id) ||
      ((gemm->descriptor.desc2 >> 18) & 0x1ff) != 0 ||
      ((gemm->descriptor.desc2 >> 27) & 0x1ff) != 5 ||
      ((gemm->descriptor.desc2 >> 54) & 3) != 2 ||
      (gemm->descriptor.desc2 & (uint64_t{1} << 3)) == 0)
    return fail("packed metadata descriptor addresses differ from the RTL ABI");
  return true;
}
bool testMockRuntimeDoneErrorAndClear() {
  VersaPBankScheduler scheduler;
  auto schedule = scheduler.scheduleGemmTile(makeRequest());
  if (!schedule)
    return fail("unexpected mock-runtime schedule error: " +
                llvm::toString(schedule.takeError()));

  MockRuntime success;
  for (const ScheduledCommand &command : schedule->commands)
    if (!success.submit(command))
      return fail("mock runtime rejected a valid done dependency chain");
  const uint32_t gemmId = schedule->commands[2].id;
  success.clearStatus(gemmId);
  if (success.hasCompleted(gemmId) || !success.wasCleared(gemmId))
    return fail("W1C clear must consume the observed done status");

  MockRuntime failure(gemmId);
  if (!failure.submit(schedule->commands[0]) || !failure.submit(schedule->commands[1]) ||
      failure.submit(schedule->commands[2]))
    return fail("mock runtime did not surface the injected GEMM error");
  if (failure.submit(schedule->commands[3]) || failure.submissions().size() != 3)
    return fail("error must stop submission of subsequent schedule commands");
  failure.globalClear();
  if (!failure.didGlobalClear() || failure.hasCompleted(schedule->commands[0].id) ||
      failure.hasCompleted(schedule->commands[1].id))
    return fail("GLOBAL_CLEAR must invalidate all locally completed state");
  return true;
}
bool testStaticEligibilityFallbacks() {
  using npux::versap::StaticGemmCandidate;
  using npux::versap::VersaPFallbackRule;
  StaticGemmCandidate candidate;
  candidate.tile = makeRequest();
  auto expect = [&](VersaPFallbackRule expected) {
    return npux::versap::evaluateStaticGemmEligibility(candidate).fallback == expected;
  };
  if (!expect(VersaPFallbackRule::AbiDisabled))
    return fail("disabled ABI fallback rule is not stable");
  candidate.descriptorAbiEnabled = true;
  if (!expect(VersaPFallbackRule::UnsupportedOp))
    return fail("unsupported-op fallback rule is not stable");
  candidate.isGemm = true;
  if (!expect(VersaPFallbackRule::DynamicShapeOrOffset))
    return fail("dynamic-shape fallback rule is not stable");
  candidate.hasStaticShapeAndOffset = true;
  if (!expect(VersaPFallbackRule::IncompleteSubgraph))
    return fail("incomplete-subgraph fallback rule is not stable");
  candidate.isCompleteSubgraph = true;
  candidate.tile.aDma->dramBase = 4;
  if (!expect(VersaPFallbackRule::DmaAlignment))
    return fail("DMA alignment fallback rule is not stable");
  candidate.tile.aDma->dramBase = 0x1000;
  auto decision = npux::versap::evaluateStaticGemmEligibility(candidate);
  if (!decision.eligible || decision.fallback != VersaPFallbackRule::None)
    return fail("fully static DMA-compute-DMA candidate was rejected");
  return true;
}

TileRequest makeAttentionQkRequest() {
  TileRequest request{128, 128, 64};
  request.operation = npux::versap::SaOperation::AttentionQk;
  request.quantization.requireRawInt32 = true;
  request.quantization.metadataWords = 1;
  request.qkGammaQ8_24 = 1U << 21; // 1 / sqrt(64), encoded as Q8.24.
  request.aDma = MvinAConfig{0x1000, 64, 128, 64, false, 0};
  // K remains in its native [sequence, headDim] layout: SA computes A*W^T.
  request.wDma = MvinWConfig{0x4000, 64, 128, 64, 0};
  request.metadataDma = npux::versap::AuxMvinConfig{
      0x6000, 0, 4, 0, false, 0};
  request.outputDma = MvoutConfig{0x8000, 128 * 4, 128, 128 * 4, 0,
      0, true};
  request.localAddresses = GemmLocalAddresses{0, 0, 0};
  return request;
}

TileRequest makeAttentionPvRequest() {
  TileRequest request{128, 64, 128};
  request.operation = npux::versap::SaOperation::AttentionPv;
  request.quantization.nextConsumesInt8 = true;
  // A is the QK probability encoding produced by hardware logP MVOUT.
  request.aDma = MvinAConfig{0xC000, 128, 128, 128, false, 0};
  // PV W layout is supplied directly by the memory plan; no scheduler transpose.
  request.wDma = MvinWConfig{0x10000, 128, 64, 128, 0};
  request.outputDma = MvoutConfig{0x14000, 64, 128, 64, 0};
  request.localAddresses = GemmLocalAddresses{0, 0, 0};
  return request;
}

bool testAttentionQkPvSchedule() {
  VersaPBankScheduler scheduler;
  AttentionHeadRequest request{makeAttentionQkRequest(), makeAttentionPvRequest()};
  auto schedule = scheduler.scheduleAttentionHead(request);
  if (!schedule)
    return fail("unexpected QK/PV schedule error: " +
                llvm::toString(schedule.takeError()));
  if (!schedule->usesHardwareQkLogP || schedule->qkLogPCommandId == 0 ||
      schedule->qk.outputMode != npux::versap::OutputMode::RawInt32 ||
      schedule->pv.outputMode != npux::versap::OutputMode::TensorInt8)
    return fail("QK/PV schedule did not select the hardware attention path");
  if (schedule->qk.commands.size() != 5 || schedule->pv.commands.size() != 4 ||
      schedule->qk.commands.back().kind != ScheduledCommandKind::StoreO ||
      schedule->qk.commands[2].kind != ScheduledCommandKind::LoadMetadata ||
      schedule->pv.commands[0].kind != ScheduledCommandKind::LoadA ||
      schedule->pv.commands[1].kind != ScheduledCommandKind::LoadW)
    return fail("QK/PV schedule must emit two complete DMA-GEMM-DMA slices");
  const auto &qkGamma = schedule->qk.commands[2];
  const auto &qkGemm = schedule->qk.commands[3];
  const auto &qkMvout = schedule->qk.commands[4];
  const auto &pvGemm = schedule->pv.commands[2];
  if ((qkGemm.descriptor.desc2 & 3) != 2 ||
      (pvGemm.descriptor.desc2 & 3) != 3 ||
      (qkMvout.descriptor.desc1 & (uint64_t{1} << 48)) == 0 ||
      ((qkGemm.descriptor.desc2 >> 60) & 1) != schedule->qk.metadataBank ||
      ((qkMvout.descriptor.desc1 >> 49) & 1) !=
          schedule->qk.qkBlockMaxSlot ||
      schedule->qk.qkBlockMaxSlot != (schedule->qk.metadataBank ^ 1) ||
      ((qkGemm.descriptor.desc2 >> 27) & 0x1ff) != 0 ||
      !hasDependency(qkGemm, qkGamma.id) ||
      !hasBankUse(qkGemm, npux::versap::BankResource::QkBlockMax,
          schedule->qk.qkBlockMaxSlot) ||
      !hasBankUse(qkMvout, npux::versap::BankResource::QkBlockMax,
          schedule->qk.qkBlockMaxSlot) ||
      schedule->qk.commands.back().id != schedule->qkLogPCommandId ||
      !hasDependency(schedule->pv.commands.front(), schedule->qkLogPCommandId))
    return fail("QK/PV descriptors or hardware logP dependency differ from the ABI");

  auto secondQk = scheduler.scheduleGemmTile(makeAttentionQkRequest());
  if (!secondQk || secondQk->metadataBank == schedule->qk.metadataBank ||
      secondQk->qkBlockMaxSlot == schedule->qk.qkBlockMaxSlot ||
      ((secondQk->commands[3].descriptor.desc2 >> 60) & 1) !=
          secondQk->metadataBank ||
      ((secondQk->commands[4].descriptor.desc1 >> 49) & 1) !=
          secondQk->qkBlockMaxSlot)
    return fail("consecutive QK descriptors did not ping-pong gamma and max slots");

  VersaPBankScheduler rowColumnScheduler;
  TileRequest rowColumn = makeAttentionQkRequest();
  rowColumn.m = 32;
  rowColumn.n = 32;
  rowColumn.aDma = MvinAConfig{0x1000, 64, 32, 64, false, 0};
  rowColumn.wDma = MvinWConfig{0x4000, 64, 32, 64, 0};
  rowColumn.metadataDma = npux::versap::AuxMvinConfig{
      0x6000, 0, 8 * 32, 0, false, 0};
  rowColumn.outputDma = MvoutConfig{0x8000, 32 * 4, 32, 32 * 4, 0,
      0, true};
  rowColumn.quantization.metadataWords = 8;
  rowColumn.qkGammaQ8_24.reset();
  rowColumn.qkRowColumnGamma = npux::versap::QkRowColumnGamma{0, 4};
  auto rowColumnSchedule = rowColumnScheduler.scheduleGemmTile(rowColumn);
  if (!rowColumnSchedule || rowColumnSchedule->commands.size() != 5 ||
      rowColumnSchedule->commands[3].descriptor.desc3 !=
          (uint64_t{1} | (uint64_t{4} << 10) | (uint64_t{4} << 19) |
              (uint64_t{4} << 27)))
    return fail("QK row-column gamma descriptor differs from the RTL ABI");

  VersaPBankScheduler missingGammaScheduler;
  TileRequest missingGamma = makeAttentionQkRequest();
  missingGamma.qkGammaQ8_24.reset();
  auto missingGammaSchedule = missingGammaScheduler.scheduleGemmTile(missingGamma);
  if (missingGammaSchedule ||
      llvm::toString(missingGammaSchedule.takeError()).find("gamma") ==
          std::string::npos)
    return fail("QK schedule accepted an unspecified gamma");

  request.pv.k = 64;
  auto invalidShape = scheduler.scheduleAttentionHead(request);
  if (invalidShape || llvm::toString(invalidShape.takeError()).find("PV input") ==
                          std::string::npos)
    return fail("QK/PV schedule accepted incompatible score and probability shapes");
  return true;
}

bool testConvGemvAndVpuSchedule() {
  VersaPBankScheduler scheduler;
  TileRequest conv{8, 8, 16};
  conv.operation = npux::versap::SaOperation::Conv;
  conv.conv = npux::versap::ConvGeometry{32, 2, 1, 0, 1, 1, 1, 1};
  conv.quantization.nextConsumesInt8 = true;
  conv.aDma = MvinAConfig{0x1000, 32, 8, 16, false, 0};
  conv.wDma = MvinWConfig{0x2000, 512, 32, 144, 0};
  conv.outputDma = MvoutConfig{0x3000, 32, 1, 32, 0};
  conv.localAddresses = GemmLocalAddresses{0, 0, 0};
  auto convSchedule = scheduler.scheduleConvTile(conv);
  if (!convSchedule || convSchedule->commands[2].kind != ScheduledCommandKind::Conv ||
      ((convSchedule->commands[2].descriptor.desc0 >> 48) & 0xffff) != 32)
    return fail("CONV schedule does not emit an SA CONV descriptor");

  VersaPBankScheduler convVpuScheduler;
  TileRequest convNoStore = conv;
  convNoStore.storeToDram = false;
  convNoStore.outputDma.reset();
  auto convVpuProducer = convVpuScheduler.scheduleConvTile(convNoStore);
  if (!convVpuProducer || convVpuProducer->commands.size() != 3)
    return fail("VPU-fed CONV must retain its INT8 result in an O bank");
  auto convVpu = convVpuScheduler.scheduleVpuTile({
      {npux::versap::VpuSpecialFunction::Transpose, 8, 32,
          convVpuProducer->oBank,
          static_cast<uint8_t>(convVpuProducer->oBank ^ 1),
          npux::versap::VpuPrecision::Int8,
          npux::versap::VpuPrecision::Int8, 0, 0, 0},
      true, MvoutConfig{0x3800, 8, 32, 8, 0}});
  if (!convVpu || convVpu->commands.size() != 2 ||
      !hasDependency(convVpu->commands[0],
          convVpuProducer->commands.back().id) ||
      convVpu->vpuOutputBank != (convVpuProducer->oBank ^ 1))
    return fail("INT8 CONV-to-VPU must preserve O-bank ownership and ordering");
  VersaPBankScheduler poolScheduler;
  auto poolProducer = poolScheduler.scheduleConvTile(convNoStore);
  if (!poolProducer)
    return fail("PoolMax requires an INT8 CONV O-bank producer");
  auto convPool = poolScheduler.scheduleVpuTile({
      {npux::versap::VpuSpecialFunction::PoolMax, 8, 32,
          poolProducer->oBank,
          static_cast<uint8_t>(poolProducer->oBank ^ 1),
          npux::versap::VpuPrecision::Int8,
          npux::versap::VpuPrecision::Int8, 0, 0, 0},
      true, MvoutConfig{0x3a00, 16, 4, 16, 0}});
  if (!convPool || convPool->commands.size() != 2 ||
      !hasDependency(convPool->commands[0],
          poolProducer->commands.back().id))
    return fail("PoolMax MVOUT must use the RTL 2x2 stride-2 output shape");

  VersaPBankScheduler gemmVpuScheduler;
  TileRequest bf16Gemm = makeRequest();
  bf16Gemm.quantization.nextConsumesInt8 = false;
  bf16Gemm.quantization.nextConsumesBf16 = true;
  bf16Gemm.storeToDram = false;
  bf16Gemm.outputDma.reset();
  auto gemmVpuProducer = gemmVpuScheduler.scheduleGemmTile(bf16Gemm);
  if (!gemmVpuProducer || gemmVpuProducer->commands.size() != 3 ||
      gemmVpuProducer->outputMode != npux::versap::OutputMode::Bf16)
    return fail("VPU-fed GEMM must retain its BF16 result in an O bank");
  auto gemmVpu = gemmVpuScheduler.scheduleVpuTile({
      {npux::versap::VpuSpecialFunction::Softmax, 2, 33,
          gemmVpuProducer->oBank,
          static_cast<uint8_t>(gemmVpuProducer->oBank ^ 1),
          npux::versap::VpuPrecision::Bf16,
          npux::versap::VpuPrecision::Bf16, 0, 0, 0},
      true, MvoutConfig{0x3c00, 0x60, 2, 66, 0}});
  if (!gemmVpu || gemmVpu->commands.size() != 2 ||
      !hasDependency(gemmVpu->commands[0],
          gemmVpuProducer->commands.back().id) ||
      gemmVpu->vpuOutputBank != (gemmVpuProducer->oBank ^ 1) ||
      gemmVpu->commands[1].kind != ScheduledCommandKind::StoreO)
    return fail("BF16 GEMM-to-VPU must retain O-bank ownership and ordering");
  auto postVpuGemm = gemmVpuScheduler.scheduleGemmTile(bf16Gemm);
  if (!postVpuGemm || postVpuGemm->oBank != gemmVpuProducer->oBank ||
      !hasDependency(postVpuGemm->commands.back(),
          gemmVpu->commands[0].id))
    return fail("a later GEMM must reuse the VPU-consumed source O bank");

  npux::versap::GemvRequest gemv{{32, 64, 64, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0}, false,
      MvinAConfig{0x4000, 64, 1, 64, false, 0},
      MvinWConfig{0x5000, 2048, 32, 64, 0},
      AuxMvinConfig{0x6000, 0, 32, 0, false, 0}, std::nullopt,
      MvoutConfig{0x7000, 64, 1, 64, 0}};
  auto gemvSchedule = scheduler.scheduleGemv(gemv);
  if (!gemvSchedule || gemvSchedule->commands[3].kind != ScheduledCommandKind::Gemv ||
      gemvSchedule->commands[3].descriptorRegisterOffset != 0x98 ||
      gemvSchedule->outputMode != npux::versap::OutputMode::Bf16)
    return fail("GEMV schedule does not emit the GEMV command");

  auto vpu = scheduler.scheduleVpu({npux::versap::VpuConfig{
      npux::versap::VpuSpecialFunction::Transpose, 4, 8,
      gemvSchedule->oBank, static_cast<uint8_t>(gemvSchedule->oBank ^ 1),
      npux::versap::VpuPrecision::Int8, npux::versap::VpuPrecision::Int8,
      0, 0, 0}});
  if (!vpu || vpu->kind != ScheduledCommandKind::Vpu ||
      vpu->descriptorRegisterOffset != 0xc0 ||
      !hasDependency(*vpu, gemvSchedule->commands[3].id))
    return fail("VPU schedule does not emit the VPU command");

  VersaPBankScheduler storeScheduler;
  auto producer = storeScheduler.scheduleGemv(gemv);
  if (!producer)
    return fail("GEMV producer for VPU MVOUT test was rejected");
  auto vpuStore = storeScheduler.scheduleVpuTile({
      {npux::versap::VpuSpecialFunction::Softmax, 1, 32,
          producer->oBank, static_cast<uint8_t>(producer->oBank ^ 1),
          npux::versap::VpuPrecision::Bf16, npux::versap::VpuPrecision::Bf16,
          0, 0, 0},
      true, MvoutConfig{0x8000, 64, 1, 64, 0}});
  if (!vpuStore || vpuStore->commands.size() != 2 ||
      vpuStore->commands[1].kind != ScheduledCommandKind::StoreO ||
      vpuStore->commands[1].reads.front().bank != vpuStore->vpuOutputBank ||
      !hasDependency(vpuStore->commands[1], vpuStore->commands[0].id))
    return fail("VPU MVOUT does not drain the destination O bank");
  return true;
}

bool testMixedEngineOBankLiveness() {
  VersaPBankScheduler scheduler;
  TileRequest gemm = makeRequest();
  gemm.quantization.nextConsumesInt8 = false;
  gemm.quantization.nextConsumesBf16 = true;
  gemm.storeToDram = false;
  gemm.outputDma.reset();
  auto sa = scheduler.scheduleGemmTile(gemm);
  if (!sa || sa->commands.size() != 3)
    return fail("mixed schedule SA producer was rejected");

  auto saVpu = scheduler.scheduleVpuTile({
      {npux::versap::VpuSpecialFunction::Gelu, 2, 33, sa->oBank,
          static_cast<uint8_t>(sa->oBank ^ 1),
          npux::versap::VpuPrecision::Bf16,
          npux::versap::VpuPrecision::Bf16, 0, 0, 0},
      true, MvoutConfig{0x8000, 0x60, 2, 66, 0}});
  if (!saVpu || !hasDependency(saVpu->commands[0], sa->commands.back().id) ||
      saVpu->vpuOutputBank != (sa->oBank ^ 1))
    return fail("mixed schedule VPU did not consume the SA O bank");

  npux::versap::GemvRequest gemv{{32, 64, 64, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0}, false,
      MvinAConfig{0x4000, 64, 1, 64, false, 0},
      MvinWConfig{0x5000, 2048, 32, 64, 0},
      npux::versap::AuxMvinConfig{0x6000, 0, 32, 0, false, 0}, std::nullopt,
      std::nullopt};
  auto vector = scheduler.scheduleGemv(gemv);
  if (!vector || vector->commands.size() != 4 ||
      vector->oBank != sa->oBank ||
      !hasDependency(vector->commands.back(), saVpu->commands[0].id))
    return fail("mixed schedule GEMV did not reuse the VPU-released O bank");

  auto vectorVpu = scheduler.scheduleVpuTile({
      {npux::versap::VpuSpecialFunction::Softmax, 1, 32, vector->oBank,
          static_cast<uint8_t>(vector->oBank ^ 1),
          npux::versap::VpuPrecision::Bf16,
          npux::versap::VpuPrecision::Bf16, 0, 0, 0},
      true, MvoutConfig{0x9000, 0x40, 1, 64, 0}});
  if (!vectorVpu || !hasDependency(vectorVpu->commands[0],
          vector->commands.back().id) ||
      vectorVpu->vpuOutputBank != (vector->oBank ^ 1))
    return fail("mixed schedule GEMV-to-VPU dependency is missing");
  return true;
}

bool testAccPartialCanOverlapVpu() {
  VersaPBankScheduler scheduler;

  TileRequest producerRequest = makeRequest();
  producerRequest.quantization.nextConsumesInt8 = false;
  producerRequest.quantization.nextConsumesBf16 = true;
  producerRequest.storeToDram = false;
  producerRequest.outputDma.reset();
  auto producer = scheduler.scheduleGemmTile(producerRequest);
  if (!producer)
    return fail("VPU source GEMM was rejected in ACC overlap test");

  auto vpu = scheduler.scheduleVpu({npux::versap::VpuConfig{
      npux::versap::VpuSpecialFunction::Gelu, 2, 33, producer->oBank,
      static_cast<uint8_t>(producer->oBank ^ 1),
      npux::versap::VpuPrecision::Bf16,
      npux::versap::VpuPrecision::Bf16, 0, 0, 0}});
  if (!vpu)
    return fail("VPU command was rejected in ACC overlap test");

  TileRequest partialRequest = makeRequest();
  partialRequest.quantization.nextConsumesInt8 = false;
  partialRequest.quantization.requireRawInt32 = true;
  partialRequest.storeToDram = false;
  partialRequest.writePartial = true;
  auto partial = scheduler.scheduleGemmTile(partialRequest);
  if (!partial)
    return fail("ACC-only partial GEMM was rejected in VPU overlap test");
  const ScheduledCommand &partialGemm = partial->commands.back();
  if (hasDependency(partialGemm, vpu->id) ||
      hasBankUse(partialGemm, npux::versap::BankResource::O, 0) ||
      hasBankUse(partialGemm, npux::versap::BankResource::O, 1))
    return fail("ACC-only partial GEMM must not wait for or claim VPU O banks");

  partialRequest.needsAccumulate = true;
  partialRequest.writePartial = false;
  partialRequest.storeToDram = true;
  auto final = scheduler.scheduleGemmTile(partialRequest);
  if (!final)
    return fail("final ACC GEMM was rejected in VPU overlap test");
  const ScheduledCommand *finalGemm = nullptr;
  for (const ScheduledCommand &command : final->commands)
    if (command.kind == ScheduledCommandKind::Gemm)
      finalGemm = &command;
  if (!finalGemm || !hasDependency(*finalGemm, vpu->id))
    return fail("final ACC GEMM must wait for the VPU O-bank ownership");
  return true;
}
} // namespace

int main() {
  return testSingleTileDescriptorsAndDag() && testPerEnginePendingWindow() &&
                 testOutputBankRotation() &&
                 testOutputModesAndMetadata() && testBiasAndMetadataPingPong() &&
                 testPackedBiasAndPerChannelScaleMetadata() &&
                 testAccumulateAndResadd() &&
                 testMockRuntimeDoneErrorAndClear() &&
                 testStaticEligibilityFallbacks() && testRejectsUnknownDma() &&
                 testOptimalGemmTiling() &&
                 testAttentionQkPvSchedule() && testConvGemvAndVpuSchedule() &&
                 testMixedEngineOBankLiveness() &&
                 testAccPartialCanOverlapVpu()
             ? 0
             : 1;
}
