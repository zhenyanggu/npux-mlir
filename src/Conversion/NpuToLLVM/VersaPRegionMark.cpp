// SPDX-License-Identifier: Apache-2.0

#include "src/Compiler/NpuConfig.hpp"
#include "src/Conversion/NpuToLLVM/VersaPDmaRegion.hpp"
#include "src/Conversion/NpuToLLVM/VersaPStaticEligibility.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <map>
#include <string>

namespace {

class VersaPRegionMarkPass
    : public mlir::PassWrapper<VersaPRegionMarkPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VersaPRegionMarkPass)

  llvm::StringRef getArgument() const override {
    return "npux-versa-p-region-mark";
  }
  llvm::StringRef getDescription() const override {
    return "Mark static Versa-P GEMM, CONV, and explicitly paired QK/PV regions";
  }

  void runOnOperation() override {
    if (!npux::useVersaPDescriptorAbi())
      return;

    uint64_t nextRegionId = 0;
    auto mark = [&](mlir::Operation *operation, uint64_t regionId,
                    llvm::StringRef role) {
      operation->setAttr("npux.versa_p_region_id",
          mlir::IntegerAttr::get(mlir::IntegerType::get(&getContext(), 64),
              regionId));
      operation->setAttr("npux.versa_p_role",
          mlir::StringAttr::get(&getContext(), role));
    };

    auto markCommand = [&](mlir::Operation *operation,
                           const npux::versap::ScheduledCommand &command) {
      auto i64 = mlir::IntegerType::get(&getContext(), 64);
      auto i32 = mlir::IntegerType::get(&getContext(), 32);
      operation->setAttr("npux.versa_p_desc0", mlir::IntegerAttr::get(i64, command.descriptor.desc0));
      operation->setAttr("npux.versa_p_desc1", mlir::IntegerAttr::get(i64, command.descriptor.desc1));
      operation->setAttr("npux.versa_p_desc2", mlir::IntegerAttr::get(i64, command.descriptor.desc2));
      operation->setAttr("npux.versa_p_api", mlir::IntegerAttr::get(i32, static_cast<uint32_t>(command.engine)));
      operation->setAttr("npux.versa_p_status_offset", mlir::IntegerAttr::get(i32, command.statusRegisterOffset));
      operation->setAttr("npux.versa_p_command_id",
          mlir::IntegerAttr::get(i32, command.id));
      llvm::SmallVector<mlir::Attribute> dependencies;
      dependencies.reserve(command.dependencies.size());
      for (uint32_t dependency : command.dependencies)
        dependencies.push_back(mlir::IntegerAttr::get(i32, dependency));
      operation->setAttr("npux.versa_p_dependencies",
          mlir::ArrayAttr::get(&getContext(), dependencies));
    };

    auto attentionRole = [](npux::ComputeRunOp compute) -> llvm::StringRef {
      auto role = compute->getAttrOfType<mlir::StringAttr>(
          "npux.versa_p_attention_role");
      return role ? role.getValue() : llvm::StringRef{};
    };
    auto attentionHead = [](npux::ComputeRunOp compute) -> mlir::StringAttr {
      return compute->getAttrOfType<mlir::StringAttr>(
          "npux.versa_p_attention_head");
    };

    auto markAttentionCommand =
        [&](mlir::Operation *operation,
            const npux::versap::ScheduledCommand &command,
            uint64_t regionId, llvm::StringRef role) {
      markCommand(operation, command);
      mark(operation, regionId, role);
    };

    std::map<mlir::Block *, npux::versap::VersaPBankScheduler>
        gemvBlockSchedulers;
    getOperation().walk([&](npux::GemvRunOp compute) {
      for (mlir::Operation *user : compute.getOutput().getUsers())
        if (auto vpu = mlir::dyn_cast<npux::VpuRunOp>(user);
            vpu && vpu.getInput() == compute.getOutput())
          return;
      auto region = npux::versap::extractGemvDmaRegion(compute);
      if (!region) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
            &getContext(), "gemv-incomplete-subgraph"));
        llvm::consumeError(region.takeError());
        return;
      }
      auto candidate = npux::versap::buildStaticGemvCandidate(*region, true);
      auto eligibility = npux::versap::evaluateStaticGemvEligibility(candidate);
      if (!eligibility.eligible) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
            &getContext(), npux::versap::fallbackRuleName(eligibility.fallback)));
        return;
      }
      auto &scheduler = gemvBlockSchedulers[compute->getBlock()];
      auto schedule = scheduler.scheduleGemv(candidate.request);
      if (!schedule || schedule->commands.size() != 5) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
            &getContext(), "gemv-scheduler-failed"));
        if (!schedule)
          llvm::consumeError(schedule.takeError());
        return;
      }
      const uint64_t regionId = nextRegionId++;
      compute->setAttr("npux.versa_p_region",
          mlir::StringAttr::get(&getContext(), "static-bf16-gemv-eligible"));
      markCommand(region->loadA, schedule->commands[0]);
      markCommand(region->loadW, schedule->commands[1]);
      markCommand(region->loadMetadata, schedule->commands[2]);
      markCommand(compute, schedule->commands[3]);
      markCommand(region->storeO, schedule->commands[4]);
      mark(region->loadA, regionId, "gemv-mvin-a");
      mark(region->loadW, regionId, "gemv-mvin-w");
      mark(region->loadMetadata, regionId, "gemv-mvin-metadata");
      mark(compute, regionId, "gemv");
      mark(region->storeO, regionId, "gemv-mvout");
    });

    std::map<mlir::Block *, npux::versap::VersaPBankScheduler>
        saVpuBlockSchedulers;
    getOperation().walk([&](npux::VpuRunOp compute) {
      npux::ComputeRunOp producer;
      for (mlir::Operation *operation = compute->getPrevNode(); operation;
           operation = operation->getPrevNode()) {
        auto candidate = mlir::dyn_cast<npux::ComputeRunOp>(operation);
        if (candidate &&
            (candidate.getOpType() == npux::ComputeOpType::conv ||
                candidate.getOpType() == npux::ComputeOpType::gemm) &&
            candidate.getOutput() == compute.getInput()) {
          producer = candidate;
          break;
        }
      }
      if (!producer)
        return;

      const bool producerIsConv =
          producer.getOpType() == npux::ComputeOpType::conv;
      auto producerRegion =
          npux::versap::extractGemmDmaRegion(producer, false);
      auto vpuRegion = npux::versap::extractVpuDmaRegion(compute);
      if (!producerRegion || producerRegion->storeO || !vpuRegion) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
            &getContext(), "vpu-sa-producer-region-invalid"));
        if (!producerRegion)
          llvm::consumeError(producerRegion.takeError());
        if (!vpuRegion)
          llvm::consumeError(vpuRegion.takeError());
        return;
      }
      auto producerCandidate = producerIsConv
          ? npux::versap::buildStaticConvCandidate(*producerRegion, true)
          : npux::versap::buildStaticRawInt32GemmCandidate(*producerRegion,
              true);
      auto vpu = npux::versap::buildStaticVpuCandidate(*vpuRegion, true);
      const auto int8 = npux::versap::VpuPrecision::Int8;
      const auto bf16 = npux::versap::VpuPrecision::Bf16;
      const auto transpose = npux::versap::VpuSpecialFunction::Transpose;
      const auto poolMax = npux::versap::VpuSpecialFunction::PoolMax;
      const bool validConvVpu = producerIsConv &&
          vpu.request.config.sourcePrecision == int8 &&
          vpu.request.config.destinationPrecision == int8 &&
          (vpu.request.config.function == transpose ||
              vpu.request.config.function == poolMax);
      const bool validGemmVpu = !producerIsConv &&
          producerCandidate.tile.quantization.nextConsumesBf16 &&
          vpu.request.config.sourcePrecision == bf16 &&
          vpu.request.config.function != transpose &&
          vpu.request.config.function != poolMax;
      if (!npux::versap::evaluateStaticGemmEligibility(producerCandidate)
               .eligible ||
          !vpu.hasStaticShapeAndOffset || (!validConvVpu && !validGemmVpu)) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
            &getContext(), "vpu-sa-static-ineligible"));
        return;
      }
      auto &scheduler = saVpuBlockSchedulers[compute->getBlock()];
      auto producerSchedule = producerIsConv
          ? scheduler.scheduleConvTile(producerCandidate.tile)
          : scheduler.scheduleGemmTile(producerCandidate.tile);
      const bool producerNeedsMetadata = producerCandidate.tile.needsBias ||
          producerCandidate.tile.quantization.metadataWords != 0;
      const std::size_t producerCommands = 3 + producerNeedsMetadata +
          producerCandidate.tile.needsResadd;
      if (!producerSchedule ||
          producerSchedule->commands.size() != producerCommands) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
            &getContext(), "vpu-sa-producer-scheduler-failed"));
        if (!producerSchedule)
          llvm::consumeError(producerSchedule.takeError());
        return;
      }
      vpu.request.config.sourceSelect = producerSchedule->oBank;
      vpu.request.config.destinationSelect = producerSchedule->oBank ^ 1;
      auto vpuSchedule = scheduler.scheduleVpuTile(vpu.request);
      if (!vpuSchedule || vpuSchedule->commands.size() != 2) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
            &getContext(), "vpu-sa-scheduler-failed"));
        if (!vpuSchedule)
          llvm::consumeError(vpuSchedule.takeError());
        return;
      }
      const uint64_t regionId = nextRegionId++;
      producer->setAttr("npux.versa_p_region", mlir::StringAttr::get(
          &getContext(), producerIsConv ? "static-conv-vpu-eligible" :
              "static-gemm-vpu-eligible"));
      compute->setAttr("npux.versa_p_region", mlir::StringAttr::get(
          &getContext(), producerIsConv ? "static-conv-vpu-eligible" :
              "static-gemm-vpu-eligible"));
      markCommand(producerRegion->loadA, producerSchedule->commands[0]);
      markCommand(producerRegion->loadW, producerSchedule->commands[1]);
      std::size_t producerIndex = 2;
      if (producerRegion->loadMetadata)
        markCommand(producerRegion->loadMetadata,
            producerSchedule->commands[producerIndex++]);
      else if (producerCandidate.tile.needsBias)
        markCommand(producerRegion->loadBias,
            producerSchedule->commands[producerIndex++]);
      else if (producerNeedsMetadata)
        markCommand(producerRegion->loadScale,
            producerSchedule->commands[producerIndex++]);
      if (producerCandidate.tile.needsResadd)
        markCommand(producerRegion->loadResidual,
            producerSchedule->commands[producerIndex++]);
      markCommand(producer, producerSchedule->commands[producerIndex]);
      markCommand(compute, vpuSchedule->commands[0]);
      markCommand(vpuRegion->storeO, vpuSchedule->commands[1]);
      const llvm::StringRef producerRole = producerIsConv ? "conv" : "gemm";
      mark(producerRegion->loadA, regionId, "sa-vpu-mvin-a");
      mark(producerRegion->loadW, regionId, "sa-vpu-mvin-w");
      if (producerRegion->loadMetadata)
        mark(producerRegion->loadMetadata, regionId, "sa-vpu-mvin-metadata");
      else if (producerCandidate.tile.needsBias)
        mark(producerRegion->loadBias, regionId, "sa-vpu-mvin-bias-metadata");
      else if (producerNeedsMetadata)
        mark(producerRegion->loadScale, regionId, "sa-vpu-mvin-scale-metadata");
      if (producerCandidate.tile.needsResadd)
        mark(producerRegion->loadResidual, regionId, "sa-vpu-mvin-residual");
      mark(producer, regionId, producerRole);
      mark(compute, regionId, "vpu");
      mark(vpuRegion->storeO, regionId, "vpu-mvout");
    });

    std::map<mlir::Block *, npux::versap::VersaPBankScheduler>
        gemvVpuBlockSchedulers;
    getOperation().walk([&](npux::VpuRunOp compute) {
      if (compute->hasAttr("npux.versa_p_region"))
        return;
      auto vpuRegion = npux::versap::extractVpuDmaRegion(compute);
      if (!vpuRegion) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
            &getContext(), "vpu-incomplete-subgraph"));
        llvm::consumeError(vpuRegion.takeError());
        return;
      }
      npux::GemvRunOp producer;
      for (mlir::Operation *operation = compute->getPrevNode(); operation;
           operation = operation->getPrevNode()) {
        auto candidate = mlir::dyn_cast<npux::GemvRunOp>(operation);
        if (candidate && candidate.getOutput() == compute.getInput()) {
          producer = candidate;
          break;
        }
      }
      if (!producer) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
            &getContext(), "vpu-o-bank-producer-missing"));
        return;
      }
      auto gemvRegion = npux::versap::extractGemvDmaRegion(producer);
      if (!gemvRegion || gemvRegion->storeO) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
            &getContext(), "vpu-producer-region-invalid"));
        if (!gemvRegion)
          llvm::consumeError(gemvRegion.takeError());
        return;
      }
      auto gemv = npux::versap::buildStaticGemvCandidate(*gemvRegion, true);
      auto vpu = npux::versap::buildStaticVpuCandidate(*vpuRegion, true);
      if (!npux::versap::evaluateStaticGemvEligibility(gemv).eligible ||
          !vpu.hasStaticShapeAndOffset) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
            &getContext(), "vpu-static-ineligible"));
        return;
      }
      auto &scheduler = gemvVpuBlockSchedulers[compute->getBlock()];
      auto gemvSchedule = scheduler.scheduleGemv(gemv.request);
      if (!gemvSchedule || gemvSchedule->commands.size() != 4) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
            &getContext(), "vpu-producer-scheduler-failed"));
        if (!gemvSchedule)
          llvm::consumeError(gemvSchedule.takeError());
        return;
      }
      vpu.request.config.sourceSelect = gemvSchedule->oBank;
      vpu.request.config.destinationSelect = gemvSchedule->oBank ^ 1;
      auto vpuSchedule = scheduler.scheduleVpuTile(vpu.request);
      if (!vpuSchedule || vpuSchedule->commands.size() != 2) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
            &getContext(), "vpu-scheduler-failed"));
        if (!vpuSchedule)
          llvm::consumeError(vpuSchedule.takeError());
        return;
      }
      const uint64_t regionId = nextRegionId++;
      producer->setAttr("npux.versa_p_region", mlir::StringAttr::get(
          &getContext(), "static-gemv-vpu-eligible"));
      compute->setAttr("npux.versa_p_region", mlir::StringAttr::get(
          &getContext(), "static-gemv-vpu-eligible"));
      markCommand(gemvRegion->loadA, gemvSchedule->commands[0]);
      markCommand(gemvRegion->loadW, gemvSchedule->commands[1]);
      markCommand(gemvRegion->loadMetadata, gemvSchedule->commands[2]);
      markCommand(producer, gemvSchedule->commands[3]);
      markCommand(compute, vpuSchedule->commands[0]);
      markCommand(vpuRegion->storeO, vpuSchedule->commands[1]);
      mark(gemvRegion->loadA, regionId, "gemv-vpu-mvin-a");
      mark(gemvRegion->loadW, regionId, "gemv-vpu-mvin-w");
      mark(gemvRegion->loadMetadata, regionId, "gemv-vpu-mvin-metadata");
      mark(producer, regionId, "gemv");
      mark(compute, regionId, "vpu");
      mark(vpuRegion->storeO, regionId, "vpu-mvout");
    });

    // Scheduler state is scoped to an MLIR block. Commands in distinct blocks
    // may be control-flow alternatives, so carrying bank last-use state across
    // them would create false descriptor dependencies.
    std::map<mlir::Block *, npux::versap::VersaPBankScheduler> blockSchedulers;
    getOperation().walk([&](npux::ComputeRunOp compute) {
      llvm::StringRef role = attentionRole(compute);
      if (role.empty() && compute.getOpType() == npux::ComputeOpType::gemm) {
        auto qkRegion = npux::versap::extractGemmDmaRegion(compute);
        if (qkRegion) {
          npux::ComputeRunOp pv;
          for (mlir::Operation *operation = compute->getNextNode(); operation;
               operation = operation->getNextNode()) {
            auto candidate = mlir::dyn_cast<npux::ComputeRunOp>(operation);
            if (!candidate || candidate.getOpType() != npux::ComputeOpType::gemm ||
                !attentionRole(candidate).empty())
              continue;
            auto pvRegion = npux::versap::extractGemmDmaRegion(candidate);
            if (!pvRegion) {
              llvm::consumeError(pvRegion.takeError());
              continue;
            }
            if (qkRegion->storeO.getHostPtr() == pvRegion->loadA.getHostPtr()) {
              pv = candidate;
              break;
            }
          }
          if (pv) {
            const std::string head = "auto-" + std::to_string(nextRegionId);
            compute->setAttr("npux.versa_p_attention_role",
                mlir::StringAttr::get(&getContext(), "qk"));
            compute->setAttr("npux.versa_p_attention_head",
                mlir::StringAttr::get(&getContext(), head));
            pv->setAttr("npux.versa_p_attention_role",
                mlir::StringAttr::get(&getContext(), "pv"));
            pv->setAttr("npux.versa_p_attention_head",
                mlir::StringAttr::get(&getContext(), head));
            role = "qk";
          }
        } else {
          llvm::consumeError(qkRegion.takeError());
        }
      }
      if (role == "pv")
        return;
      if (role == "qk") {
        auto head = attentionHead(compute);
        if (!head) {
          compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
              &getContext(), "attention-head-missing"));
          return;
        }
        npux::ComputeRunOp pv;
        for (mlir::Operation *operation = compute->getNextNode(); operation;
             operation = operation->getNextNode()) {
          auto candidate = mlir::dyn_cast<npux::ComputeRunOp>(operation);
          if (candidate && attentionRole(candidate) == "pv" &&
              attentionHead(candidate) &&
              attentionHead(candidate).getValue() == head.getValue()) {
            pv = candidate;
            break;
          }
        }
        if (!pv) {
          compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
              &getContext(), "attention-pv-missing"));
          return;
        }
        auto qkRegion = npux::versap::extractGemmDmaRegion(compute);
        auto pvRegion = npux::versap::extractGemmDmaRegion(pv);
        if (!qkRegion || !pvRegion) {
          compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
              &getContext(), "attention-incomplete-subgraph"));
          if (!qkRegion)
            llvm::consumeError(qkRegion.takeError());
          if (!pvRegion)
            llvm::consumeError(pvRegion.takeError());
          return;
        }
        auto qkCandidate = npux::versap::buildStaticAttentionCandidate(
            *qkRegion, npux::versap::SaOperation::AttentionQk, true);
        auto pvCandidate = npux::versap::buildStaticAttentionCandidate(
            *pvRegion, npux::versap::SaOperation::AttentionPv, true);
        if (!npux::versap::evaluateStaticGemmEligibility(qkCandidate)
                 .eligible ||
            !npux::versap::evaluateStaticGemmEligibility(pvCandidate)
                 .eligible) {
          compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
              &getContext(), "attention-static-ineligible"));
          return;
        }
        npux::versap::VersaPBankScheduler scheduler;
        auto attention = scheduler.scheduleAttentionHead(
            {qkCandidate.tile, pvCandidate.tile});
        if (!attention || attention->qk.commands.size() != 5 ||
            attention->pv.commands.size() != 4) {
          compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
              &getContext(), "attention-scheduler-failed"));
          if (!attention)
            llvm::consumeError(attention.takeError());
          return;
        }
        const uint64_t regionId = nextRegionId++;
        compute->setAttr("npux.versa_p_region", mlir::StringAttr::get(
            &getContext(), "attention-qk-pv"));
        pv->setAttr("npux.versa_p_region", mlir::StringAttr::get(
            &getContext(), "attention-qk-pv"));
        markAttentionCommand(qkRegion->loadA, attention->qk.commands[0],
            regionId, "qk-mvin-a");
        markAttentionCommand(qkRegion->loadW, attention->qk.commands[1],
            regionId, "qk-mvin-w");
        markAttentionCommand(qkRegion->loadMetadata, attention->qk.commands[2],
            regionId, "qk-mvin-gamma");
        markAttentionCommand(compute, attention->qk.commands[3], regionId,
            "qk-sa");
        markAttentionCommand(qkRegion->storeO, attention->qk.commands[4],
            regionId, "qk-logp-mvout");
        markAttentionCommand(pvRegion->loadA, attention->pv.commands[0],
            regionId, "pv-mvin-a");
        markAttentionCommand(pvRegion->loadW, attention->pv.commands[1],
            regionId, "pv-mvin-w");
        markAttentionCommand(pv, attention->pv.commands[2], regionId,
            "pv-sa");
        markAttentionCommand(pvRegion->storeO, attention->pv.commands[3],
            regionId, "pv-mvout");
        return;
      }
      if (!role.empty()) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(
            &getContext(), "attention-role-invalid"));
        return;
      }
      for (mlir::Operation *user : compute.getOutput().getUsers())
        if (auto vpu = mlir::dyn_cast<npux::VpuRunOp>(user);
            vpu && vpu.getInput() == compute.getOutput())
          return;
      bool requireOutputDma = true;
      if (auto stage = compute->getAttrOfType<mlir::StringAttr>(
              "npux.versa_p_acc_stage"))
        requireOutputDma = stage.getValue() != "first" &&
            stage.getValue() != "body";
      auto region = npux::versap::extractGemmDmaRegion(compute,
          requireOutputDma);
      if (!region) {
        compute->setAttr("npux.versa_p_fallback",
            mlir::StringAttr::get(&getContext(), "incomplete-subgraph"));
        llvm::consumeError(region.takeError());
        return;
      }

      auto candidate = compute.getOpType() == npux::ComputeOpType::conv
          ? npux::versap::buildStaticConvCandidate(*region, true)
          : npux::versap::buildStaticRawInt32GemmCandidate(*region, true);
      auto eligibility =
          npux::versap::evaluateStaticGemmEligibility(candidate);
      if (!eligibility.eligible) {
        compute->setAttr("npux.versa_p_fallback",
            mlir::StringAttr::get(&getContext(),
                npux::versap::fallbackRuleName(eligibility.fallback)));
        return;
      }

      auto &blockScheduler = blockSchedulers[compute->getBlock()];
      auto schedule = candidate.tile.operation == npux::versap::SaOperation::Conv
          ? blockScheduler.scheduleConvTile(candidate.tile)
          : blockScheduler.scheduleGemmTile(candidate.tile);
      const bool needsMetadata = candidate.tile.needsBias ||
          candidate.tile.quantization.metadataWords != 0;
      const std::size_t expectedCommands = 3 + needsMetadata +
          candidate.tile.needsResadd + candidate.tile.storeToDram;
      if (!schedule || schedule->commands.size() != expectedCommands) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(&getContext(), "scheduler-failed"));
        if (!schedule) llvm::consumeError(schedule.takeError());
        return;
      }
      uint64_t regionId = nextRegionId++;
      compute->setAttr("npux.versa_p_region", mlir::StringAttr::get(
          &getContext(), candidate.tile.writePartial ||
                  candidate.tile.needsAccumulate
              ? "static-acc-gemm-eligible"
              : candidate.tile.operation == npux::versap::SaOperation::Conv
                  ? "static-int8-conv-eligible" : "static-raw-int32-eligible"));
      markCommand(region->loadA, schedule->commands[0]);
      markCommand(region->loadW, schedule->commands[1]);
      std::size_t computeIndex = 2;
      if (needsMetadata) {
        if (region->loadMetadata)
          markCommand(region->loadMetadata, schedule->commands[computeIndex]);
        else if (candidate.tile.needsBias)
          markCommand(region->loadBias, schedule->commands[computeIndex]);
        else
          markCommand(region->loadScale, schedule->commands[computeIndex]);
        ++computeIndex;
      }
      if (candidate.tile.needsResadd)
        markCommand(region->loadResidual, schedule->commands[computeIndex++]);
      markCommand(compute, schedule->commands[computeIndex]);
      if (candidate.tile.storeToDram)
        markCommand(region->storeO, schedule->commands[computeIndex + 1]);
      mark(region->loadA, regionId, "mvin-a");
      mark(region->loadW, regionId, "mvin-w");
      if (region->loadMetadata)
        mark(region->loadMetadata, regionId, "mvin-metadata");
      else if (candidate.tile.needsBias)
        mark(region->loadBias, regionId, "mvin-bias-metadata");
      else if (needsMetadata)
        mark(region->loadScale, regionId, "mvin-scale-metadata");
      if (candidate.tile.needsResadd)
        mark(region->loadResidual, regionId, "mvin-residual");
      mark(compute, regionId,
          candidate.tile.writePartial || candidate.tile.needsAccumulate
              ? "gemm-acc" : candidate.tile.operation ==
                    npux::versap::SaOperation::Conv ? "conv" : "gemm");
      if (candidate.tile.storeToDram)
        mark(region->storeO, regionId, "mvout");
    });
  }
};

static mlir::PassRegistration<VersaPRegionMarkPass> registration;

} // namespace

std::unique_ptr<mlir::Pass> npux::createVersaPRegionMarkPass() {
  return std::make_unique<VersaPRegionMarkPass>();
}
