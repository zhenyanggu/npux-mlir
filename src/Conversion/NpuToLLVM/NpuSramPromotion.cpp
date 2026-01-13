//======================================================
// src/Conversion/NpuToLLVM/NpuSramPromotion.cpp
// Promotes DRAM subviews to SRAM buffers for computation
//======================================================

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

class PromoteLinalgToSramPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    
    if (!op->hasAttr("npu.target")) return failure();

    if (op.getInputs().empty()) return failure();
    auto inType = cast<MemRefType>(op.getInputs()[0].getType());
    if (inType.getMemorySpaceAsInt() == 2) return failure(); 

    Location loc = op.getLoc();
    Value dramInput = op.getInputs()[0];
    Value dramOutput = op.getOutputs()[0]; // Init Tensor


    auto shape = inType.getShape();

    auto sramType = MemRefType::get(shape, inType.getElementType(), {}, 2); 

    Value sramIn = rewriter.create<memref::AllocOp>(loc, sramType);
    Value sramOut = rewriter.create<memref::AllocOp>(loc, sramType);


    rewriter.create<memref::CopyOp>(loc, dramInput, sramIn);


    Operation *newOp = rewriter.clone(*op.getOperation());
    newOp->setOperand(0, sramIn);     // Input -> SRAM
    newOp->setOperand(1, sramOut);    // Output -> SRAM


    rewriter.create<memref::CopyOp>(loc, sramOut, dramOutput);

    rewriter.eraseOp(op);

    return success();
  }
};

struct SramPromotionPass : public PassWrapper<SramPromotionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SramPromotionPass)

  StringRef getArgument() const override {
    return "npu-sram-promotion";
  }
  StringRef getDescription() const override {
    return "Promote DRAM subviews to SRAM buffers for computation";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<PromoteLinalgToSramPattern>(context);
    
    // 使用 GreedyRewrite 跑 pattern
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuSramPromotionPass() {
  return std::make_unique<SramPromotionPass>();
}