// SPDX-License-Identifier: Apache-2.0

#include "src/Compiler/NpuConfig.hpp"
#include "src/Conversion/NpuToLLVM/VersaPDmaRegion.hpp"
#include "src/Conversion/NpuToLLVM/VersaPStaticEligibility.hpp"
#include "src/Conversion/NpuToLLVM/VersaPStaticEligibility.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

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
    return "Mark statically eligible raw-INT32 DMA-GEMM-DMA Versa-P regions";
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
    };

    getOperation().walk([&](npux::ComputeRunOp compute) {
      auto region = npux::versap::extractGemmDmaRegion(compute);
      if (!region) {
        compute->setAttr("npux.versa_p_fallback",
            mlir::StringAttr::get(&getContext(), "incomplete-subgraph"));
        llvm::consumeError(region.takeError());
        return;
      }

      auto candidate = npux::versap::buildStaticRawInt32GemmCandidate(
          *region, true);
      auto eligibility =
          npux::versap::evaluateStaticGemmEligibility(candidate);
      if (!eligibility.eligible) {
        compute->setAttr("npux.versa_p_fallback",
            mlir::StringAttr::get(&getContext(),
                npux::versap::fallbackRuleName(eligibility.fallback)));
        return;
      }

      npux::versap::VersaPBankScheduler scheduler;
      auto schedule = scheduler.scheduleGemmTile(candidate.tile);
      if (!schedule || schedule->commands.size() != 4) {
        compute->setAttr("npux.versa_p_fallback", mlir::StringAttr::get(&getContext(), "scheduler-failed"));
        if (!schedule) llvm::consumeError(schedule.takeError());
        return;
      }
      uint64_t regionId = nextRegionId++;
      compute->setAttr("npux.versa_p_region", mlir::StringAttr::get(&getContext(), "static-raw-int32-eligible"));
      markCommand(region->loadA, schedule->commands[0]);
      markCommand(region->loadW, schedule->commands[1]);
      markCommand(compute, schedule->commands[2]);
      markCommand(region->storeO, schedule->commands[3]);
      mark(region->loadA, regionId, "mvin-a");
      mark(region->loadW, regionId, "mvin-w");
      mark(compute, regionId, "gemm");
      mark(region->storeO, regionId, "mvout");
    });
  }
};

static mlir::PassRegistration<VersaPRegionMarkPass> registration;

} // namespace

std::unique_ptr<mlir::Pass> npux::createVersaPRegionMarkPass() {
  return std::make_unique<VersaPRegionMarkPass>();
}