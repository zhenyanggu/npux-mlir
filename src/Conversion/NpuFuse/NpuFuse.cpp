//======================================
// src/Conversion/NpuFuse/NpuFuse.cpp
// this file implements the fusion of NPU
// operations for optimization
//======================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"

#include "src/Conversion/NpuFuse/FuseHelper.hpp"

using namespace mlir;

namespace {
struct NpuFusePass
    : public PassWrapper<NpuFusePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuFusePass)

  StringRef getArgument() const override { return "npu-fuse"; }
  StringRef getDescription() const override {
    return "Fuse operations for NPU execution.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto targetAttr = func->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu") {
      return;
    }

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    npux::populateNpuFusePatterns(patterns);

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true)
        .enableFolding(true);

    if (failed(applyPatternsGreedily(func.getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  };
};

} // namespace

std::unique_ptr<Pass> npux::createNpuFusePass() {
  return std::make_unique<NpuFusePass>();
}