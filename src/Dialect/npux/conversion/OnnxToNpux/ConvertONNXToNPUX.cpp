#include "Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/npux/ir/npuxDialect.h"
#include "src/Dialect/npux/ir/npuxOps.h"
// ONNX dialect headers come from onnx-mlir source tree
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace {
struct ConvertONNXGeluOp : public OpRewritePattern<ONNXGeluOp> {
  using OpRewritePattern<ONNXGeluOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ONNXGeluOp op,
                                PatternRewriter &rewriter) const override {
    // npux.gelu takes the same input and produces same type
    Value in = op.getX();
    Type resultType = op.getResult().getType();
    auto newOp = rewriter.create<npux::geluOp>(op.getLoc(), resultType, in);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertONNXToNPUXPass
    : public PassWrapper<ConvertONNXToNPUXPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertONNXToNPUXPass)
  StringRef getArgument() const final { return "convert-onnx-to-npux"; }
  StringRef getDescription() const final { return "Lower ONNX to NPUX"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<npux::npuxDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<npux::npuxDialect, func::FuncDialect>();
    // Mark ONNX ops illegal so they must be rewritten.
  target.addIllegalOp<ONNXGeluOp>();
    patterns.add<ConvertONNXGeluOp>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace npux {
std::unique_ptr<mlir::Pass> createConvertONNXToNPUXPass() {
  return std::make_unique<ConvertONNXToNPUXPass>();
}

void registerONNXToNPUXPasses() {
  PassRegistration<ConvertONNXToNPUXPass>();
}
} // namespace npux
