//===========================================
// src/Transforms/RemoveDuplicateMvinBias.cpp
//
// Description:
// This pass removes redundant npux.mvin_bias operations within the same block.
// It relies on the standard MLIR CSE (-cse) pass running beforehand to 
// deduplicate the preceding affine.apply and memref.subview operations 
// into identical SSA values.
//===========================================

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "src/Dialect/Npux/NpuxOps.hpp" 
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace npux;

namespace {

struct RemoveDuplicateMvinBiasPattern : public OpRewritePattern<npux::MvinBiasOp> {
  using OpRewritePattern<npux::MvinBiasOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(npux::MvinBiasOp mvinOp,
                                PatternRewriter &rewriter) const override {
    Block *block = mvinOp->getBlock();
    
    // 往前遍历同一个 Block 里的指令，寻找在它之前是否已经有相同的 mvin_bias
    for (Operation &op : *block) {
      // 如果按顺序遍历遇到了自己，说明在自己前面并没有出现过重复的，直接退出检查
      if (&op == mvinOp.getOperation()) {
        break;
      }
      
      if (auto prevMvinOp = dyn_cast<npux::MvinBiasOp>(&op)) {
        // 比较操作数 (Operands) 和属性 (Attributes) 是否完全一致
        // 注意：这里能用 `==` 比较的前提是前面已经跑过 `-cse` Pass，
        // `%subview_8` 和 `%subview_9` 已经被 CSE 合并成了同一个 SSA Value。
        if (prevMvinOp->getOperands() == mvinOp->getOperands() && 
            prevMvinOp->getAttrs() == mvinOp->getAttrs()) {
          
          // 发现重复，删除当前的 mvinOp
          rewriter.eraseOp(mvinOp);
          return success();
        }
      }
    }

    return failure();
  }
};

struct RemoveDuplicateMvinBiasPass : public PassWrapper<RemoveDuplicateMvinBiasPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveDuplicateMvinBiasPass)

  StringRef getArgument() const override { return "npu-remove-duplicate-mvin-bias"; }
  
  StringRef getDescription() const override { 
    return "Remove redundant npux.mvin_bias operations within the same block (requires prior CSE)"; 
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    
    patterns.add<RemoveDuplicateMvinBiasPattern>(context);
    
    // 使用 Greedy Rewrite 框架
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true)
          .enableFolding(true);

    if (failed(applyPatternsGreedily(
            getOperation().getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createRemoveDuplicateMvinBiasPass() {
  return std::make_unique<RemoveDuplicateMvinBiasPass>();
}