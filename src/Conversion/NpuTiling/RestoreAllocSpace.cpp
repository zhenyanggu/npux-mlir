#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "src/Pass/Passes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace {

// 专门抓捕丢失 memory_space 属性的 alloc_tensor
struct RestoreAllocSpacePattern : public OpRewritePattern<bufferization::AllocTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::AllocTensorOp op, PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type) return failure();
    
    // 检查 Tensor 基因里有没有存 memory space
    auto enc = type.getEncoding();
    if (!enc) return failure();

    // 如果外挂属性完好无损，就不管它
    if (op.getMemorySpace() && op.getMemorySpace() == enc)
      return failure();

    // 发现属性丢失！立刻用基因里的值强行恢复它
    rewriter.modifyOpInPlace(op, [&]() {
      op.setMemorySpaceAttr(enc);
    });
    return success();
  }
};

// 包装成一个 Pass
struct RestoreAllocSpacePass : public PassWrapper<RestoreAllocSpacePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RestoreAllocSpacePass)
  
  StringRef getArgument() const final { return "restore-alloc-space"; }
  StringRef getDescription() const final { return "Restore memory_space attribute for alloc_tensor from its type encoding."; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<RestoreAllocSpacePattern>(&getContext());
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true)
        .enableFolding(true);

    // 修改：使用 applyPatternsGreedily 替代 applyPartialConversion
    if (failed(applyPatternsGreedily(func.getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

// 暴露创建接口
std::unique_ptr<Pass> npux::createRestoreAllocSpacePass() {
  return std::make_unique<RestoreAllocSpacePass>();
}