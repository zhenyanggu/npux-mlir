//===============================================
// src/Conversion/NpuPartition/RemoveRedundantDma.cpp
// this file removes redundant mvin/mvout operations
//===============================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include "src/Pass/Passes.hpp" 

using namespace mlir;

namespace {

//=============================================================================
// Pattern: RemoveRedundantNpucoreDma
// 匹配并消除 npucore.dma_mvin 直接读取 npucore.dma_mvout 结果的冗余模式
//=============================================================================
struct RemoveRedundantNpucoreDma
    : public OpRewritePattern<npucore::DmaMvinOp> {
  using OpRewritePattern<npucore::DmaMvinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(npucore::DmaMvinOp mvinOp,
      PatternRewriter &rewriter) const override {
    if (mvinOp.getInputs().size() != 1 || mvinOp.getResultTensors().size() != 1)
      return failure();

    Value mvinInput = mvinOp.getInputs().front();
    auto mvoutOp = mvinInput.getDefiningOp<npucore::DmaMvoutOp>();
    if (!mvoutOp || mvoutOp.getInputs().size() != 1)
      return failure();

    Value originalTensor = mvoutOp.getInputs().front();
    rewriter.replaceOp(mvinOp, originalTensor);
    return success();
  }
};

//=============================================================================
// Pass Definition
//=============================================================================
struct NpuRemoveRedundantDmaPass
    : public PassWrapper<NpuRemoveRedundantDmaPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuRemoveRedundantDmaPass)

  llvm::StringRef getArgument() const override { return "npu-remove-redundant-dma"; }
  llvm::StringRef getDescription() const override {
    return "Eliminate redundant npu_dma_mvin operations that directly read from npu_dma_mvout.";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 注册我们写好的 Pattern
    patterns.add<RemoveRedundantNpucoreDma>(context);

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true)
        .enableFolding(true)
        .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Aggressive);

    if (failed(applyPatternsGreedily(
            getOperation().getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuRemoveRedundantDmaPass() {
  return std::make_unique<NpuRemoveRedundantDmaPass>();
}
