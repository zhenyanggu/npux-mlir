//=============================================================================
// src/Conversion/NpuTiling/FusionAnalysis/NpuFusionAnalysis.cpp
// This file plans profitable NPU fusion groups and records the plan on IR.
//=============================================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Conversion/NpuTiling/FusionAnalysis/FusionEvaluator.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kFusionGroupAttr = "npu.fusion.group";
constexpr llvm::StringLiteral kFusionRoleAttr = "npu.fusion.role";
constexpr llvm::StringLiteral kFusionSeedKindAttr = "npu.fusion.seed_kind";

struct FusionPlan {
  int64_t groupId = -1;
  linalg::LinalgOp seed;
  SmallVector<linalg::LinalgOp> ops;
  double benefit = 0.0;
};

//=============================================================================
// Pass Definition
//=============================================================================

struct NpuFusionAnalysisPass
    : public PassWrapper<NpuFusionAnalysisPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuFusionAnalysisPass)

  llvm::StringRef getArgument() const override { return "npu-fusion-analysis"; }
  
  llvm::StringRef getDescription() const override {
    return "Plan profitable NPU fusion groups based on wait_irq and DMA costs.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    FusionCostEvaluator evaluator(/*cpuWaitIrqTime=*/15.0);
    llvm::SmallPtrSet<Operation *, 32> assignedOps;
    int64_t nextGroupId = 0;

    func.walk([&](linalg::LinalgOp op) {
      if (!isSeedOp(op) || assignedOps.contains(op.getOperation())) {
        return;
      }

      FusionPlan plan = buildGreedyPlan(op, nextGroupId, evaluator, assignedOps);
      if (plan.ops.size() < 2) {
        return;
      }

      markFusionPlan(plan);
      for (linalg::LinalgOp plannedOp : plan.ops) {
        assignedOps.insert(plannedOp.getOperation());
      }
      ++nextGroupId;
      dumpFusionPlan(plan);
    });
  }

private:
  llvm::StringRef getLibraryCallName(linalg::LinalgOp op) const {
    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr) {
      return {};
    }
    return libCallAttr.getValue();
  }

  bool isNpuOp(linalg::LinalgOp op) const {
    llvm::StringRef libCall = getLibraryCallName(op);
    return libCall.starts_with("npu_") || libCall.starts_with("mv_");
  }

  bool isSeedOp(linalg::LinalgOp op) const {
    llvm::StringRef libCall = getLibraryCallName(op);
    return libCall == "npu_conv" || libCall == "npu_gemm" ||
           libCall == "npu_matmul";
  }

  bool isEligibleConsumer(linalg::LinalgOp seed, linalg::LinalgOp consumer,
      const llvm::SmallPtrSetImpl<Operation *> &assignedOps) const {
    if (!consumer || !isNpuOp(consumer)) {
      return false;
    }
    if (consumer->hasAttr("npu.tiled") || consumer->hasAttr(kFusionGroupAttr)) {
      return false;
    }
    if (assignedOps.contains(consumer.getOperation())) {
      return false;
    }
    if (isSeedOp(consumer)) {
      return false;
    }
    return seed->getBlock() == consumer->getBlock();
  }

  SmallVector<linalg::LinalgOp> getDirectConsumers(linalg::LinalgOp op) const {
    SmallVector<linalg::LinalgOp> consumers;
    llvm::SmallPtrSet<Operation *, 4> seen;
    for (OpResult result : op->getResults()) {
      if (!result.hasOneUse()) {
        continue;
      }
      Operation *user = (*result.getUsers().begin());
      auto consumer = dyn_cast<linalg::LinalgOp>(user);
      if (consumer && seen.insert(user).second) {
        consumers.push_back(consumer);
      }
    }
    return consumers;
  }

  linalg::LinalgOp chooseBestConsumer(linalg::LinalgOp seed,
      linalg::LinalgOp tail, FusionCostEvaluator &evaluator,
      const llvm::SmallPtrSetImpl<Operation *> &assignedOps,
      double &bestBenefit) const {
    linalg::LinalgOp bestConsumer;
    bestBenefit = 0.0;

    for (linalg::LinalgOp consumer : getDirectConsumers(tail)) {
      if (!isEligibleConsumer(seed, consumer, assignedOps)) {
        continue;
      }

      double benefit = evaluator.evaluateFusionBenefit(tail, consumer);
      llvm::outs() << "[NpuFusionAnalysis] candidate "
                   << getLibraryCallName(tail) << " -> "
                   << getLibraryCallName(consumer) << " benefit=" << benefit
                   << "\n";
      if (benefit > bestBenefit) {
        bestBenefit = benefit;
        bestConsumer = consumer;
      }
    }

    return bestConsumer;
  }

  FusionPlan buildGreedyPlan(linalg::LinalgOp seed, int64_t groupId,
      FusionCostEvaluator &evaluator,
      const llvm::SmallPtrSetImpl<Operation *> &assignedOps) const {
    FusionPlan plan;
    plan.groupId = groupId;
    plan.seed = seed;
    plan.ops.push_back(seed);

    linalg::LinalgOp tail = seed;
    while (true) {
      double stepBenefit = 0.0;
      linalg::LinalgOp consumer =
          chooseBestConsumer(seed, tail, evaluator, assignedOps, stepBenefit);
      if (!consumer || stepBenefit <= 0.0) {
        break;
      }

      plan.ops.push_back(consumer);
      plan.benefit += stepBenefit;
      tail = consumer;
    }

    return plan;
  }

  void markFusionPlan(const FusionPlan &plan) const {
    Builder builder(plan.seed->getContext());
    StringAttr seedKind = builder.getStringAttr(getLibraryCallName(plan.seed));

    for (auto indexedOp : llvm::enumerate(plan.ops)) {
      linalg::LinalgOp op = indexedOp.value();
      int64_t order = indexedOp.index();
      StringRef role = "body";
      if (order == 0) {
        role = "seed";
      } else if (order == static_cast<int64_t>(plan.ops.size()) - 1) {
        role = "root";
        op->setAttr(kFusionSeedKindAttr, seedKind);
      }

      op->setAttr(kFusionGroupAttr, builder.getI64IntegerAttr(plan.groupId));
      op->setAttr(kFusionRoleAttr, builder.getStringAttr(role));
    }

  }

  void dumpFusionPlan(const FusionPlan &plan) const {
    llvm::outs() << "[NpuFusionAnalysis] group=" << plan.groupId
                 << " seed_kind=" << getLibraryCallName(plan.seed)
                 << " benefit=" << plan.benefit << " chain=";
    for (auto indexedOp : llvm::enumerate(plan.ops)) {
      if (indexedOp.index() != 0) {
        llvm::outs() << " -> ";
      }
      llvm::outs() << getLibraryCallName(indexedOp.value());
    }
    llvm::outs() << "\n";
  }
};

} // namespace

//=============================================================================
// Factory Function Implementation
//=============================================================================
std::unique_ptr<Pass> npux::createNpuFusionAnalysisPass() {
  return std::make_unique<NpuFusionAnalysisPass>();
}
