//===================================================
// src/Conversion/NpuTiling/NpuTiling.cpp
// this file implements the all the tiling patterns for NPU
//===================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h" 
#include "src/Pass/Passes.hpp"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

using namespace mlir;

namespace npux {
void populateNpuTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  populateElemWiseTilingPatterns(patterns, context);
  populateConvTilingPatterns(patterns, context);
  populateGemmTilingPatterns(patterns, context);
  populateLayoutTilingPatterns(patterns, context);
  populateMaxPoolTilingPatterns(patterns, context);
};
} // namespace npux

namespace {
struct NpuTilingPass
    : public PassWrapper<NpuTilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuTilingPass)

  StringRef getArgument() const override { return "npu-tiling"; }
  StringRef getDescription() const override {
    return "Tile operations for NPU execution.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto targetAttr = func->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu") {
      return;
    }

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 添加 Tiling Patterns
    npux::populateNpuTilingPatterns(patterns, context);
    //patterns.add<linalg::ExtractSliceOfPadTensorSwapPattern>(context);

    // 修改：配置 GreedyRewriteConfig (参考你的例子)
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true)
        .enableFolding(true);

    // 修改：使用 applyPatternsGreedily 替代 applyPartialConversion
    if (failed(applyPatternsGreedily(func.getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  };
};

} // namespace

std::unique_ptr<Pass> npux::createNpuTilingPass() {
  return std::make_unique<NpuTilingPass>();
}