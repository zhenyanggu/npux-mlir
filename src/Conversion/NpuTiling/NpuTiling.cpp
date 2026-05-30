//===================================================
// src/Conversion/NpuTiling/NpuTiling.cpp
// this file implements the all the tiling patterns for NPU
//===================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {
struct NpuTilingPass
    : public PassWrapper<NpuTilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuTilingPass)

  StringRef getArgument() const override { return "npu-tiling"; }
  StringRef getDescription() const override {
    return "Tile npucore Gelu operations for NPU execution.";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    npux::populateNpuTilingPatterns(patterns, context);

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true).enableFolding(true);

    if (failed(applyPatternsGreedily(
            getOperation().getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  };
};

} // namespace

std::unique_ptr<Pass> npux::createNpuTilingPass() {
  return std::make_unique<NpuTilingPass>();
}
