//===============================================
// src/Conversion/NpuPartition/RemoveRedundantDma.cpp
// this file removes redundant mvin/mvout operations
//===============================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp" 

using namespace mlir;

namespace {

static LogicalResult removeRedundantDmaLikeOp(
    Operation *mvinOp, Value mvinInput, Value mvinResult,
    PatternRewriter &rewriter) {
  auto libCall = mvinOp->getAttrOfType<StringAttr>("library_call");
  if (!libCall || libCall.getValue() != "npu_dma_mvin")
    return failure();

  Operation *mvoutOp = mvinInput.getDefiningOp();
  if (!mvoutOp)
    return failure();

  auto mvoutLibCall = mvoutOp->getAttrOfType<StringAttr>("library_call");
  if (!mvoutLibCall || mvoutLibCall.getValue() != "npu_dma_mvout")
    return failure();

  Value originalTensor;
  if (auto genericMvout = dyn_cast<linalg::GenericOp>(mvoutOp)) {
    if (genericMvout.getNumDpsInputs() != 1)
      return failure();
    originalTensor = genericMvout.getDpsInputOperand(0)->get();
  } else if (auto copyMvout = dyn_cast<linalg::CopyOp>(mvoutOp)) {
    if (copyMvout.getNumDpsInputs() != 1)
      return failure();
    originalTensor = copyMvout.getDpsInputOperand(0)->get();
  } else {
    return failure();
  }

  if (originalTensor.getType() != mvinResult.getType())
    return failure();

  rewriter.replaceOp(mvinOp, originalTensor);
  return success();
}

//=============================================================================
// Pattern: RemoveRedundantDma
// 匹配并消除 mvin 直接读取 mvout 结果的冗余模式
//=============================================================================
struct RemoveRedundantDma : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp mvinOp,
                                PatternRewriter &rewriter) const override {
    if (mvinOp.getNumDpsInputs() != 1 || mvinOp.getNumResults() != 1)
      return failure();
    return removeRedundantDmaLikeOp(
        mvinOp, mvinOp.getDpsInputOperand(0)->get(), mvinOp.getResult(0),
        rewriter);
  }
};

struct RemoveRedundantCopyDma : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern<linalg::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::CopyOp mvinOp,
                                PatternRewriter &rewriter) const override {
    if (mvinOp.getNumDpsInputs() != 1 || mvinOp.getNumResults() != 1)
      return failure();
    return removeRedundantDmaLikeOp(
        mvinOp, mvinOp.getDpsInputOperand(0)->get(), mvinOp.getResult(0),
        rewriter);
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
    patterns.add<RemoveRedundantDma, RemoveRedundantCopyDma>(context);

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
