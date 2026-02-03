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
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/NpuPartition/LinalgConversionHelper.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

static bool isSupportedPooling(Operation *op) {
  // 获取属性
  ArrayAttr kernelShape, strides;
  if (auto maxPool = dyn_cast<ONNXMaxPoolSingleOutOp>(op)) {
    kernelShape = maxPool.getKernelShapeAttr();
    strides = maxPool.getStridesAttr();
  } else if (auto avgPool = dyn_cast<ONNXAveragePoolOp>(op)) {
    kernelShape = avgPool.getKernelShapeAttr();
    strides = avgPool.getStridesAttr();
  } else {
    return false;
  }

  if (!kernelShape || !strides) return false;

  // 1. 检查维度是否为 2D
  if (kernelShape.size() != 2 || strides.size() != 2) return false;

  // 2. 检查数值是否全为 2
  auto checkAttr = [](ArrayAttr attr) {
    for (auto val : attr) {
      if (cast<IntegerAttr>(val).getInt() != 2) return false;
    }
    return true;
  };

  return checkAttr(kernelShape) && checkAttr(strides);
}

static bool isSupportedResize(ONNXResizeOp op) {
  // 1. Check Mode
  if (op.getMode() != "nearest") return false;

  // 2. Check Scales
  Value scales = op.getScales();
  auto constOp = scales.getDefiningOp<ONNXConstantOp>();
  if (!constOp) return false;

  ElementsAttr valueAttr = dyn_cast<ElementsAttr>(constOp.getValueAttr());
  if (!valueAttr) return false;
  bool hasScale2 = false;
  for (auto val : valueAttr.getValues<float>()) {
    float s = std::abs(val);
    if (std::abs(s - 2.0f) < 1e-5) hasScale2 = true;
    else if (std::abs(s - 1.0f) > 1e-5) return false; 
  }
  return hasScale2;
}

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

    bool isEmpty=onnx_mlir::NpuOps.empty();

    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Conv)) {
      target.addIllegalOp<ONNXConvOp>();
    }
    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::MatMul)) {
      target.addIllegalOp<ONNXQLinearMatMulOp>();
    }
    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::LayerNorm)) {
      target.addIllegalOp<ONNXLayerNormalizationOp>();
    }
    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Softmax)) {
      target.addIllegalOp<ONNXSoftmaxOp>();
    }
    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Gelu)) {
      target.addIllegalOp<ONNXGeluOp>();
    }
    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Gemm)) {
      target.addIllegalOp<ONNXGemmOp>();
    }
    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Relu)) {
      target.addIllegalOp<ONNXReluOp, ONNXLeakyReluOp>();
    }
    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Transpose)) {
      target.addIllegalOp<ONNXTransposeOp>();
    }


    if(isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::MaxPool)) {
      target.addDynamicallyLegalOp<ONNXMaxPoolSingleOutOp>(
          [](ONNXMaxPoolSingleOutOp op) {
            return !isSupportedPooling(op);
          });
    }
    if(isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::AveragePool)) {
      target.addDynamicallyLegalOp<ONNXAveragePoolOp>(
          [](ONNXAveragePoolOp op) {
            return !isSupportedPooling(op);
          });
    }
    if(isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Resize)) {
      target.addDynamicallyLegalOp<ONNXResizeOp>(
          [](ONNXResizeOp op) {
            return !isSupportedResize(op);
          });
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
