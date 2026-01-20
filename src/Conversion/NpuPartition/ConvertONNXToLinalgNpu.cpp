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
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/NpuPartition/LinalgConversionHelper.hpp"


using namespace mlir;


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
    target.addLegalDialect<ONNXDialect>();

    if(onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Conv))
    {
      target.addIllegalOp<ONNXConvOp>();
    }
    if(onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::MatMul))
    {
      target.addIllegalOp<ONNXMatMulOp>();
    }
    if(onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::LayerNorm))
    {
      target.addIllegalOp<ONNXLayerNormalizationOp>();
    }
    if(onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Softmax))
    {
      target.addIllegalOp<ONNXSoftmaxOp>();
    }
    if(onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Gelu))
    {
      target.addIllegalOp<ONNXGeluOp>();
    }
    if(onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Gemm))
    {
      target.addIllegalOp<ONNXGemmOp>();
    }

    if(onnx_mlir::NpuOps.empty())
    {
      target.addIllegalOp<ONNXConvOp, ONNXLayerNormalizationOp,
                          ONNXSoftmaxOp, ONNXGeluOp>();
    }
    



    RewritePatternSet patterns(context);

    npux::populateLinalgConversionPatterns(patterns);



    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createONNXToLinalgNpuPass() {
  return std::make_unique<ONNXToLinalgNpuPass>();
}
