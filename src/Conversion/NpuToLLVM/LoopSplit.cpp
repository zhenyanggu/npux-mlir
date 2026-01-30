//===========================================
// src/Conversion/NpuToLLVM/LoopSplit.cpp
//
// Description:
// This pass splits the IC (Inner Channel) loop into Head, Body, and Tail parts
// to facilitate NPU pipeline optimizations (e.g., Preload/Init in Head, 
// Writeback in Tail).
// 
// It handles the "Single Iteration" edge case by branching:
// - If (iter <= 1): Generate "single" loop (Init + Compute + Writeback).
// - Else: Generate "head" (Init), "body" (Compute), "tail" (Writeback).
//===========================================

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/IRMapping.h" 
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

struct SplitICLoopPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loopOp,
                                PatternRewriter &rewriter) const override {
    
    // 1. 过滤掉已经处理过的循环
    if (loopOp->hasAttr("npu.peeled_split")) return failure();

    // 2. 核心条件A：必须带有 npu.computeop 标记 (支持 conv, gemm, matmul)
    if (!loopOp->hasAttr("npu.computeop")) return failure();

    // 3. 核心条件B：必须是“带有该标记的循环”中的“最内层”
    bool hasInnerComputeLoop = false;
    loopOp.walk([&](scf::ForOp nestedLoop) {
        if (nestedLoop == loopOp) return WalkResult::advance(); 
        if (nestedLoop->hasAttr("npu.computeop")) {
            hasInnerComputeLoop = true;
            return WalkResult::interrupt(); 
        }
        return WalkResult::advance();
    });

    if (hasInnerComputeLoop) return failure();

    // ==============================================================
    // 能走到这里，说明 loopOp 是 ComputeOp 的最内层循环
    // ==============================================================

    auto loopDimAttr = loopOp->getAttrOfType<StringAttr>("npu.loop_dim");
    if (!loopDimAttr) return failure(); 

    StringRef loopDim = loopDimAttr.getValue();
    Location loc = loopOp.getLoc();

    // [关键修改] 定义什么是归约维度
    // Conv 的归约维是 IC (Input Channel)
    // Gemm/Matmul 的归约维是 K
    bool isReductionDim = (loopDim == "IC" || loopDim == "K");

    // ==============================================================
    // Path A: 空间维度 (M, N, OH, OW, OC...) -> 包一层 Single
    // 如果它不是归约维度，我们就认为它是空间维度
    // ==============================================================
    if (!isReductionDim) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(loopOp.getBody());

        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        
        // 创建 0..1 的伪循环 (Single Iteration Loop)
        auto innerLoop = rewriter.create<scf::ForOp>(loc, zero, one, one);
        
        innerLoop->setAttr("npu.loop_part", rewriter.getStringAttr("single"));
        innerLoop->setAttr("npu.peeled_split", rewriter.getUnitAttr());
        innerLoop->setAttr("npu.computeop", loopOp->getAttr("npu.computeop"));
        
        // 传递维度信息，方便后续调试
        if(loopDimAttr) innerLoop->setAttr("npu.loop_dim", loopDimAttr);

        // 剪切(Splice)当前循环体的内容到新循环中
        Block *outerBody = loopOp.getBody();
        Block *innerBody = innerLoop.getBody();
        
        auto &outerOps = outerBody->getOperations();
        auto &innerOps = innerBody->getOperations();

        auto moveStart = ++Block::iterator(innerLoop); 
        auto moveEnd = std::prev(outerOps.end()); 

        if (moveStart != outerOps.end() && moveStart != moveEnd->getIterator()) {
            innerOps.splice(innerBody->getTerminator()->getIterator(), outerOps, moveStart, moveEnd);
        }

        loopOp->setAttr("npu.peeled_split", rewriter.getUnitAttr());
        
        return success();
    }

    // ==============================================================
    // Path B: 归约维度 (IC, K) -> 执行 Split (Head/Body/Tail)
    // ==============================================================
    if (loopOp.getNumRegionIterArgs() > 0) return failure(); 

    Value lb = loopOp.getLowerBound();
    Value ub = loopOp.getUpperBound();
    Value step = loopOp.getStep();

    std::optional<int64_t> lbConst = getConstantIntValue(lb);
    std::optional<int64_t> ubConst = getConstantIntValue(ub);
    std::optional<int64_t> stepConst = getConstantIntValue(step);

    auto createLabeledLoop = [&](OpBuilder &builder, Value s, Value e, StringRef label) {
      OpBuilder::InsertionGuard guard(builder);
      auto newLoop = builder.create<scf::ForOp>(loc, s, e, step);
      
      Block *oldBody = loopOp.getBody();
      Block *newBody = newLoop.getBody();
      builder.setInsertionPointToStart(newBody);
      
      IRMapping mapper;
      mapper.map(loopOp.getInductionVar(), newLoop.getInductionVar());
      for (auto &op : oldBody->without_terminator()) {
        builder.clone(op, mapper);
      }
      
      newLoop->setAttrs(loopOp->getAttrs());
      newLoop->setAttr("npu.loop_part", builder.getStringAttr(label));
      newLoop->setAttr("npu.peeled_split", builder.getUnitAttr());
    };

    bool generateSinglePath = false;
    bool generateSplitPath = false;
    bool needsRuntimeCheck = true; 

    if (lbConst && ubConst && stepConst) {
        int64_t range = *ubConst - *lbConst;
        int64_t stepVal = *stepConst;
        needsRuntimeCheck = false; 
        if (range <= stepVal) generateSinglePath = true; 
        else generateSplitPath = true;
    }

    if (!needsRuntimeCheck && generateSinglePath) {
        createLabeledLoop(rewriter, lb, ub, "single");
    }
    else if (!needsRuntimeCheck && generateSplitPath) {
        Value headEnd = rewriter.create<arith::AddIOp>(loc, lb, step);
        Value tailStart = rewriter.create<arith::SubIOp>(loc, ub, step);
        createLabeledLoop(rewriter, lb, headEnd, "head");
        
        // 注意：这里需要确保中间确实有 Body
        if ((*ubConst - *lbConst) > 2 * *stepConst) {
             createLabeledLoop(rewriter, headEnd, tailStart, "body");
        }
        createLabeledLoop(rewriter, tailStart, ub, "tail");
    }
    else {
        // 动态检查
        Value range = rewriter.create<arith::SubIOp>(loc, ub, lb);
        Value isSingle = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, range, step);
        auto ifOp = rewriter.create<scf::IfOp>(loc, isSingle, /*withElseRegion=*/true);

        {
            OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
            createLabeledLoop(thenBuilder, lb, ub, "single");
        }
        {
            OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
            Value headEnd = elseBuilder.create<arith::AddIOp>(loc, lb, step);
            Value tailStart = elseBuilder.create<arith::SubIOp>(loc, ub, step);
            createLabeledLoop(elseBuilder, lb, headEnd, "head");
            createLabeledLoop(elseBuilder, headEnd, tailStart, "body");
            createLabeledLoop(elseBuilder, tailStart, ub, "tail");
        }
    }

    rewriter.eraseOp(loopOp);
    return success();
  }
};

// ... Pass 注册代码保持不变 ...
struct NpuICLoopSplitPass : public PassWrapper<NpuICLoopSplitPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuICLoopSplitPass)
  StringRef getArgument() const override { return "npu-split-loop"; }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<SplitICLoopPattern>(context);
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true).enableFolding(true);
    if (failed(applyPatternsGreedily(getOperation().getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> npux::createNpuICLoopSplitPass() {
  return std::make_unique<NpuICLoopSplitPass>();
}