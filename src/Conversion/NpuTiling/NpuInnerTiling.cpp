//===================================================
// src/Conversion/NpuTiling/NpuInnerTiling.cpp
// this file implements the tiling of the Channel/32 dim
//===================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"

#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

using namespace mlir;

namespace npux {
void populateNpuInnerTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  //populateElemWiseInnerTilingPatterns(patterns, context);
  populateConvInnerTilingPatterns(patterns, context);
};
} // namespace npux

namespace {
struct NpuInnerTilingPass
    : public PassWrapper<NpuInnerTilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuInnerTilingPass)

  StringRef getArgument() const override { return "npu-inner-tiling"; }
  StringRef getDescription() const override {
    return "Tile Channel/32 dimension since the kernel can only handle 32 "
           "channel at one time.";
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
    npux::populateNpuInnerTilingPatterns(patterns, context);

    // 修改：配置 GreedyRewriteConfig (参考你的例子)
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true).enableFolding(true);

    // 修改：使用 applyPatternsGreedily 替代 applyPartialConversion
    if (failed(applyPatternsGreedily(
            func.getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  };
};

} // namespace

std::unique_ptr<Pass> npux::createNpuInnerTilingPass() {
  return std::make_unique<NpuInnerTilingPass>();
}