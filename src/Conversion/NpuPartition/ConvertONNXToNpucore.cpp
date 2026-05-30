//=============================================================================
// /src/Conversion/NpuPartition/ConvertONNXToNpucore.cpp
// this file implements the NPU partitioning pass that lowers ONNX SFU
// operations directly to the npucore dialect.
//=============================================================================

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/NpuPartition/NpuCoreConversionHelper.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

// ============================================================================
// Pass: ONNX Gelu/Softmax/LayerNorm/Add participate in the native npucore
// experiment.
// ============================================================================
struct ONNXToNpucorePass
    : public PassWrapper<ONNXToNpucorePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXToNpucorePass)

  StringRef getArgument() const override { return "convert-npu-onnx-to-npucore"; }
  StringRef getDescription() const override {
    return "Lower ONNX Gelu/Softmax/LayerNorm/Add ops to npucore.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    bool isEmpty = onnx_mlir::NpuOps.empty();

    ConversionTarget target(*context);
    target.addLegalDialect<npucore::NpucoreDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<ONNXDialect>();

    if (isEmpty || onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Conv)) {
      target.addDynamicallyLegalOp<ONNXConvOp>(
          [](Operation *op) { return !npux::hasStrictQDQContext(op); });
    }
    if (isEmpty || onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Gelu)) {
      target.addDynamicallyLegalOp<ONNXGeluOp>(
          [](Operation *op) { return !npux::hasStrictQDQContext(op); });
    }
    if (isEmpty || onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Softmax)) {
      target.addDynamicallyLegalOp<ONNXSoftmaxOp>(
          [](Operation *op) { return !npux::hasStrictQDQContext(op); });
    }
    if (isEmpty || onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::LayerNorm)) {
      target.addDynamicallyLegalOp<ONNXLayerNormalizationOp>(
          [](Operation *op) { return !npux::hasStrictQDQContext(op); });
    }
    if (isEmpty || onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Add)) {
      target.addDynamicallyLegalOp<ONNXAddOp>([](ONNXAddOp op) {
        return !npux::supportsNpucoreMatAdd(op);
      });
    }
    if (isEmpty || onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Gemm)) {
      target.addDynamicallyLegalOp<ONNXGemmOp>(
          [](Operation *op) { return !npux::hasStrictQDQContext(op); });
    }
    if (isEmpty || onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::MatMul)) {
      target.addDynamicallyLegalOp<ONNXMatMulOp>(
          [](Operation *op) { return !npux::hasStrictQDQContext(op); });
      target.addIllegalOp<ONNXQLinearMatMulOp>();
    }
    if (isEmpty || onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::MaxPool)) {
      target.addDynamicallyLegalOp<ONNXMaxPoolSingleOutOp>(
          [](Operation *op) { return !npux::hasStrictQDQContext(op); });
    }

    RewritePatternSet patterns(context);
    npux::populateNpucoreConversionPatterns(patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> npux::createONNXToNpucorePass() {
  return std::make_unique<ONNXToNpucorePass>();
}
