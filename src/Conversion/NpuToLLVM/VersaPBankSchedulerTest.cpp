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
    bool perChannel;
  };
  const std::array<ModeCase, 4> cases = {{{npux::versap::OutputMode::RawInt32,
      true, false, false}, {npux::versap::OutputMode::TensorInt8, false, false,
      false}, {npux::versap::OutputMode::TensorInt8, false, false, true},
      {npux::versap::OutputMode::Fp32, false, true, false}}};
  for (const ModeCase &mode : cases) {
    VersaPBankScheduler scheduler;
    TileRequest request = makeRequest();
    request.quantization.nextConsumesInt8 = !mode.raw && !mode.fp32;
    request.quantization.requireRawInt32 = mode.raw;
    request.quantization.requireFp32 = mode.fp32;
    request.quantization.perChannelScaleAvailable = mode.perChannel;
    if (mode.raw || mode.fp32) {
      request.outputDma->dramRowStrideBytes = 0xA0;
      request.outputDma->rowBytes = 132;
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
        (!mode.raw && !mode.fp32 ? 1 : mode.fp32 ? 1 : 0);
    if (((gemm->descriptor.desc2 >> 54) & 3) != expectedScaleMode)
      return fail("GEMM scale-mode bits differ from the ABI");
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
  request.needsAccumulate = true;
  auto accumulation = scheduler.scheduleGemmTile(request);
  if (accumulation ||
      llvm::toString(accumulation.takeError()).find("write_partial") ==
          std::string::npos)
    return fail("single-tile scheduler must reject ACC input without a prior partial result");

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
  request.aDma = MvinAConfig{0x1000, 64, 128, 64, false, 0};
  // K remains in its native [sequence, headDim] layout: SA computes A*W^T.
  request.wDma = MvinWConfig{0x4000, 64, 128, 64, 0};
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
  if (schedule->qk.commands.size() != 4 || schedule->pv.commands.size() != 4 ||
      schedule->qk.commands.back().kind != ScheduledCommandKind::StoreO ||
      schedule->pv.commands[0].kind != ScheduledCommandKind::LoadA ||
      schedule->pv.commands[1].kind != ScheduledCommandKind::LoadW)
    return fail("QK/PV schedule must emit two complete DMA-GEMM-DMA slices");
  const auto &qkGemm = schedule->qk.commands[2];
  const auto &qkMvout = schedule->qk.commands[3];
  const auto &pvGemm = schedule->pv.commands[2];
  if ((qkGemm.descriptor.desc2 & 3) != 2 ||
      (pvGemm.descriptor.desc2 & 3) != 3 ||
      (qkMvout.descriptor.desc1 & (uint64_t{1} << 48)) == 0 ||
      schedule->qk.commands.back().id != schedule->qkLogPCommandId ||
      !hasDependency(schedule->pv.commands.front(), schedule->qkLogPCommandId))
    return fail("QK/PV descriptors or hardware logP dependency differ from the ABI");

  request.pv.k = 64;
  auto invalidShape = scheduler.scheduleAttentionHead(request);
  if (invalidShape || llvm::toString(invalidShape.takeError()).find("PV input") ==
                          std::string::npos)
    return fail("QK/PV schedule accepted incompatible score and probability shapes");
  return true;
}
} // namespace

int main() {
  return testSingleTileDescriptorsAndDag() && testOutputBankRotation() &&
                 testOutputModesAndMetadata() && testBiasAndMetadataPingPong() &&
                 testAccumulateAndResadd() &&
                 testMockRuntimeDoneErrorAndClear() &&
                 testStaticEligibilityFallbacks() && testRejectsUnknownDma() &&
                 testAttentionQkPvSchedule()
             ? 0
             : 1;
}
