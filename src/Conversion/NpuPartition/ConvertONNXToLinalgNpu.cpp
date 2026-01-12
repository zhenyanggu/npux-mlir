//=============================================================================
// /src/Conversion/NpuPartition/ConvertONNXToLinalgNpu.cpp
// this file implements the NPU partitioning pass that labels
// ONNX operations for NPU execution based on a conversion registry.
//=============================================================================

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/NPUConversionRegistry.hpp"


using namespace mlir;

namespace npux {
void registerNpuOpConversions();
}

namespace {
struct ONNXToLinalgNpuPass
    : public PassWrapper<ONNXToLinalgNpuPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXToLinalgNpuPass)

  StringRef getArgument() const override {
    return "convert-npu-onnx-to-linalg";
  }
  StringRef getDescription() const override {
    return "Lower ONNX ops to Linalg (Int8) inside NPU kernels.";
  }

  void runOnOperation() override {
    npux::registerNpuOpConversions();
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();


    ConversionTarget target(*context);

    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addLegalDialect<scf::SCFDialect>();

    npux::NPUConversionRegistry::setIllegalOps(target, context);



    RewritePatternSet patterns(context);

    npux::NPUConversionRegistry::populatePatterns(patterns);


    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createONNXToLinalgNpuPass() {
  return std::make_unique<ONNXToLinalgNpuPass>();
}
