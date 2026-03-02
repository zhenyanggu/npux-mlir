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

//=============================================================================
// Pattern: RemoveRedundantDma
// 匹配并消除 mvin 直接读取 mvout 结果的冗余模式
//=============================================================================
struct RemoveRedundantDma : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp mvinOp,
                                PatternRewriter &rewriter) const override {
    // 1. 检查当前 Op 是否为 "npu_dma_mvin"
    auto libCall = mvinOp->getAttrOfType<StringAttr>("library_call");
    if (!libCall || libCall.getValue() != "npu_dma_mvin")
      return failure();

    // 2. 确保它符合单输入单输出的 Tensor 语义
    if (mvinOp.getNumDpsInputs() != 1 || mvinOp.getNumResults() != 1)
      return failure();

    Value mvinInput = mvinOp.getDpsInputOperand(0)->get();

    // 3. 顺藤摸瓜，检查输入是否由一个 "npu_dma_mvout" 产生
    auto mvoutOp = mvinInput.getDefiningOp<linalg::GenericOp>();
    if (!mvoutOp)
      return failure();

    auto mvoutLibCall = mvoutOp->getAttrOfType<StringAttr>("library_call");
    if (!mvoutLibCall || mvoutLibCall.getValue() != "npu_dma_mvout")
      return failure();

    // 4. 获取 mvoutOp 最原始的输入 (即驻留在 memory_space=2 中的 tensor)
    if (mvoutOp.getNumDpsInputs() != 1)
      return failure();

    Value originalTensor = mvoutOp.getDpsInputOperand(0)->get();

    // 5. 旁路 (Bypass)：将 mvinOp 的所有使用者直接替换为 originalTensor
    // 替换后，mvinOp 变成死代码被删除。如果 mvoutOp 的结果也没有其他使用者，
    // 它也会被贪心重写驱动器 (Greedy Pattern Rewrite Driver) 自动消除。
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
    patterns.add<RemoveRedundantDma>(context);

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