//=============================================================
// src/Conversion/NpuTiling/NpuSpatialPeeling.cpp
//=============================================================

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace mlir::linalg;

namespace {

//----------------------------------------------------------------------------//
// Helper Function: Create Peeled Body for Spatial Dims
//----------------------------------------------------------------------------//

Value createSpatialPeeledLoopBody(OpBuilder &b, scf::ForOp originalLoop, Value iv,
                                  ValueRange iterArgs, IRMapping &mapping) {
  Block *originalBody = originalLoop.getBody();

  // 1. 映射 Induction Variable
  mapping.map(originalLoop.getInductionVar(), iv);

  // 2. 映射 iter_args
  unsigned argIdx = 0;
  for (Value arg : originalLoop.getRegionIterArgs()) {
    if (argIdx < iterArgs.size()) {
      mapping.map(arg, iterArgs[argIdx++]);
    }
  }

  // 3. 克隆 Body
  for (auto &op : originalBody->without_terminator()) {
    b.clone(op, mapping);
  }

  // 4. 处理 Terminator (scf.yield)
  if (auto yieldOp = dyn_cast<scf::YieldOp>(originalBody->getTerminator())) {
    if (yieldOp.getNumOperands() > 0) {
      return mapping.lookup(yieldOp.getOperand(0));
    }
  }
  return nullptr;
}

//----------------------------------------------------------------------------//
// Pattern: NpuConvSpatialTilingPattern
//----------------------------------------------------------------------------//

struct NpuConvSpatialTilingPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // 1. Check Loop Dimension Attribute
    StringRef loopDim = "";
    if (auto loopDimAttr = op->getAttrOfType<StringAttr>("npu.loop_dim")) {
      loopDim = loopDimAttr.getValue();
      // Only process OH (Height) and OW (Width)
      if (loopDim != "OH" && loopDim != "OW") {
        return failure();
      }
    } else {
      return failure();
    }

    Location loc = op.getLoc();
    Value lb = op.getLowerBound();
    Value ub = op.getUpperBound();
    Value step = op.getStep();
    Value initOutput = op.getInitArgs()[0];

    // 2. Strict Check for Static Loop Bounds
    std::optional<int64_t> staticLb = getConstantIntValue(lb);
    std::optional<int64_t> staticUb = getConstantIntValue(ub);
    std::optional<int64_t> staticStep = getConstantIntValue(step);

    bool isStatic = staticLb.has_value() && staticUb.has_value() &&
                    staticStep.has_value();

    if (!isStatic) {
      return failure();
    }

    // Calculate iteration count
    int64_t diff = *staticUb - *staticLb;
    int64_t constNumIters = (diff + *staticStep - 1) / *staticStep;

    if (constNumIters <= 0) return failure();

    // ====================================================
    // Case A: Single Iteration -> Inline Body
    // ====================================================
    if (constNumIters == 1) {
      IRMapping mapping;
      Value res = createSpatialPeeledLoopBody(rewriter, op, lb, {initOutput}, mapping);
      rewriter.replaceOp(op, res);
      return success();
    }

    // ====================================================
    // Case B: Multi Iterations -> Head -> Body -> Tail
    // ====================================================
    
    // --- 1. Head (First Iteration) ---
    IRMapping mapHead;
    Value headResult = createSpatialPeeledLoopBody(rewriter, op, lb, {initOutput}, mapHead);

    Value currentAcc = headResult;

    // --- 2. Body Loop (Middle Iterations) ---
    if (constNumIters > 2) {
      int64_t bodyStartVal = *staticLb + *staticStep;
      int64_t lastIterIdxVal = *staticLb + (constNumIters - 1) * *staticStep;

      Value bodyLb = rewriter.create<arith::ConstantIndexOp>(loc, bodyStartVal);
      Value bodyUb = rewriter.create<arith::ConstantIndexOp>(loc, lastIterIdxVal);
      
      auto bodyLoop = rewriter.create<scf::ForOp>(
          loc, bodyLb, bodyUb, step, ValueRange{currentAcc},
          [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
            IRMapping mapBody;
            Value res = createSpatialPeeledLoopBody(b, op, iv, args, mapBody);
            b.create<scf::YieldOp>(loc, res);
          });
      
      // Copy attributes but remove npu.loop_dim to prevent infinite recursion
      bodyLoop->setAttrs(op->getAttrs());
      bodyLoop->removeAttr("npu.loop_dim");

      currentAcc = bodyLoop.getResult(0);
    }

    // --- 3. Tail (Last Iteration) ---
    int64_t lastIterIdxVal = *staticLb + (constNumIters - 1) * *staticStep;
    Value lastIdx = rewriter.create<arith::ConstantIndexOp>(loc, lastIterIdxVal);
    
    IRMapping mapTail;
    Value finalRes = createSpatialPeeledLoopBody(rewriter, op, lastIdx, {currentAcc}, mapTail);

    rewriter.replaceOp(op, finalRes);
    return success();
  }
};

void populateNpuSpatialPeelingPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  patterns.add<NpuConvSpatialTilingPattern>(context);
}

//----------------------------------------------------------------------------//
// Pass Definition
//----------------------------------------------------------------------------//

struct NpuSpatialPeelingPass
    : public PassWrapper<NpuSpatialPeelingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuSpatialPeelingPass)

  StringRef getArgument() const override { return "npu-spatial-peeling"; }
  StringRef getDescription() const override {
    return "Peel loops for OH/OW spatial dimensions when bounds are static.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto targetAttr = func->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu") {
      return;
    }

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 添加 Spatial Peeling Patterns
    populateNpuSpatialPeelingPatterns(patterns, context);

    // 配置 GreedyRewriteConfig
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true).enableFolding(true);

    // 应用 Patterns
    if (failed(applyPatternsGreedily(
            func.getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  };
};

} // namespace

std::unique_ptr<Pass> npux::createNpuSpatialPeelingPass() {
  return std::make_unique<NpuSpatialPeelingPass>();
}
