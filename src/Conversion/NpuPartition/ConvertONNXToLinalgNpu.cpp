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

// ============================================================================
// 辅助函数：严格检查上下游是否有 Dequantize -> Op -> Quantize 模式并报警
// ============================================================================
static bool hasStrictQDQContext(Operation *op) {
  if (op->getNumOperands() == 0 || op->getNumResults() == 0) return false;

  // 1. 检查上游 Operand 0 是否有 DequantizeLinearOp
  Value input = op->getOperand(0);
  bool hasDq = (input.getDefiningOp<ONNXDequantizeLinearOp>() != nullptr);

  // 2. 检查下游 Result 0 是否有且仅有一个 QuantizeLinearOp
  Value result = op->getResult(0);
  bool hasQ = false;
  if (result.hasOneUse() && isa<ONNXQuantizeLinearOp>(*result.getUsers().begin())) {
    hasQ = true;
  }

  // 3. 满足严格条件，放行
  if (hasDq && hasQ) {
    return true; 
  }

  // 4. 不满足条件，利用 Attr 避免重复报 Warning
  StringRef warningAttrName = "npu_qdq_warning_emitted";
  if (!op->hasAttr(warningAttrName)) {
    op->emitWarning() << "Operation '" << op->getName() 
                      << "' lacks strict Dequantize -> Op -> Quantize context. "
                      << "Skipping conversion to NPU Linalg.";
    // 打上标签，标记该 Op 已经报过警
    op->setAttr(warningAttrName, UnitAttr::get(op->getContext()));
  }
  
  return false;
}

static bool isSupportedPooling(Operation *op) {
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

  if (kernelShape.size() != 2 || strides.size() != 2) return false;

  auto checkAttr = [](ArrayAttr attr) {
    for (auto val : attr) {
      if (cast<IntegerAttr>(val).getInt() != 2) return false;
    }
    return true;
  };

  return checkAttr(kernelShape) && checkAttr(strides);
}

static bool isSupportedResize(ONNXResizeOp op) {
  if (op.getMode() != "nearest") return false;

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

    // --- 所有目标 Op 全部修改为 DynamicallyLegalOp，强制绑定 QDQ 检查 ---

    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Conv)) {
      target.addDynamicallyLegalOp<ONNXConvOp>([](Operation *op) { return !hasStrictQDQContext(op); });
    }
    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::MatMul)) {
      target.addIllegalOp<ONNXQLinearMatMulOp>();
    }
    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::LayerNorm)) {
      target.addDynamicallyLegalOp<ONNXLayerNormalizationOp>([](Operation *op) { return !hasStrictQDQContext(op); });
    }
    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Softmax)) {
      target.addDynamicallyLegalOp<ONNXSoftmaxOp>([](Operation *op) { return !hasStrictQDQContext(op); });
    }
    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Gelu)) {
      target.addDynamicallyLegalOp<ONNXGeluOp>([](Operation *op) { return !hasStrictQDQContext(op); });
    }
    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Gemm)) {
      target.addDynamicallyLegalOp<ONNXGemmOp>([](Operation *op) { return !hasStrictQDQContext(op); });
    }
    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Relu)) {
      target.addDynamicallyLegalOp<ONNXReluOp, ONNXLeakyReluOp>([](Operation *op) { return !hasStrictQDQContext(op); });
    }
    if (isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Transpose)) {
      target.addDynamicallyLegalOp<ONNXTransposeOp>([](Operation *op) { return !hasStrictQDQContext(op); });
    }

    // 组合判定：满足属性支持 且 满足 QDQ 上下文，才视作 Illegal(被 Pattern 拦截改写)
    if(isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::MaxPool)) {
      target.addDynamicallyLegalOp<ONNXMaxPoolSingleOutOp>(
          [](ONNXMaxPoolSingleOutOp op) {
            return !(isSupportedPooling(op) && hasStrictQDQContext(op));
          });
    }
    if(isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::AveragePool)) {
      target.addDynamicallyLegalOp<ONNXAveragePoolOp>(
          [](ONNXAveragePoolOp op) {
            return !(isSupportedPooling(op) && hasStrictQDQContext(op));
          });
    }
    if(isEmpty||onnx_mlir::hasNpuOp(onnx_mlir::NpuOp::Resize)) {
      target.addDynamicallyLegalOp<ONNXResizeOp>(
          [](ONNXResizeOp op) {
            return !(isSupportedResize(op) && hasStrictQDQContext(op));
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