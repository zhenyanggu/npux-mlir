//======================================================
// src/Conversion/NpuBufferization/NpuBufferizationHelper.cpp
// this file implements the NPU bufferization pass
//======================================================

#include "src/Conversion/NpuBufferization/NpuBufferizationHelper.hpp" // 引入对应的头文件

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace mlir::bufferization;

// 修复 1: 添加返回值 LogicalResult
LogicalResult RunNpuBufferization(ModuleOp module) {
  bufferization::OneShotBufferizationOptions options;
  
  options.allowUnknownOps = true;
  options.bufferizeFunctionBoundaries = false;

  // --- 模拟官方 use-encoding-for-memory-space=true 的逻辑 ---
  options.defaultMemorySpaceFn = [](TensorType t) -> std::optional<Attribute> {
    // 官方源码逻辑：如果是 RankedTensorType，返回其 Encoding
    if (auto rtt = llvm::dyn_cast<RankedTensorType>(t))
      return rtt.getEncoding();
    return std::nullopt;
  };
  // -------------------------------------------------------

  options.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);

  options.unknownTypeConverterFn =
      [](TensorType tensorType, Attribute memorySpace,
         const bufferization::BufferizationOptions &options) {
        // 这里的 memorySpace 会接收来自上面 Lambda 返回的 encoding
        return bufferization::getMemRefTypeWithStaticIdentityLayout(
            tensorType, memorySpace);
      };

  bufferization::BufferizationState state;
  if (failed(bufferization::runOneShotBufferize(module, options, state))) {
    module.emitError("NPU Kernel One-Shot Bufferization failed");
    return failure();
  }

  return success();
}

namespace {
// 把 Pattern 放在匿名空间，通过下面的 populate 函数暴露给外部
struct DowngradeToBufferPattern
    : public OpRewritePattern<bufferization::ToBufferOp> {
  using OpRewritePattern<bufferization::ToBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      bufferization::ToBufferOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, op.getType(), op.getOperand());
    return success();
  }
};

struct DowngradeToTensorPattern
    : public OpRewritePattern<bufferization::ToTensorOp> {
  using OpRewritePattern<bufferization::ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      bufferization::ToTensorOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, op.getType(), op.getOperand());
    return success();
  }
};
} // namespace

// 辅助函数：让外部文件也能注册这两个 Pattern
void populateBufferizationCleanUpHelperPatterns(RewritePatternSet &patterns) {
  patterns.insert<DowngradeToBufferPattern, DowngradeToTensorPattern>(
      patterns.getContext());
}

// 你的 Pass 定义保持不变
namespace {
struct NpuDPSConversionPass
    : public PassWrapper<NpuDPSConversionPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuDPSConversionPass)

  StringRef getArgument() const override { return "npu-dps-convert"; }
  StringRef getDescription() const override {
    return "Promote buffer results to out params for NPU kernels";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bufferization::BufferResultsToOutParamsOpts opts;

    opts.hoistStaticAllocs = true;
    opts.filterFn = [](func::FuncOp *func) {
      return (*func)->hasAttr("npu.target");
    };

    if (failed(bufferization::promoteBufferResultsToOutParams(module, opts))) {
      return signalPassFailure();
    }
  }
};


struct LowerCustomMHAPattern : public OpConversionPattern<ONNXCustomOp> {
  using OpConversionPattern<ONNXCustomOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXCustomOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 验证算子属性
    auto funcNameAttr = op->getAttrOfType<StringAttr>("function_name");
    if (!funcNameAttr || funcNameAttr.getValue() != "MultiHeadAttention") {
      return failure();
    }

    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    // 修复 Error 3: 严格转换为 FloatType
    FloatType f32Type = cast<FloatType>(rewriter.getF32Type());
    Type i64Type = rewriter.getI64Type();

    // 2. 获取 Bufferization 后的输入 (MemRef 类型)
    ValueRange operands = adaptor.getOperands();
    Value qMemRef = operands[0];
    Value kMemRef = operands[1];
    Value vMemRef = operands[2];
    Type memRefType = qMemRef.getType();

    // 3. 声明外部函数
    StringRef funcName = "mha_cpu_standalone_wrapper"; 
    auto funcOp = module.lookupSymbol<func::FuncOp>(funcName);
    if (!funcOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      // 签名：4个 MemRef，4个 i64，1个 f32
      FunctionType funcType = rewriter.getFunctionType(
          {memRefType, memRefType, memRefType, memRefType, 
           i64Type, i64Type, i64Type, i64Type, f32Type}, 
          {} // 无返回值
      );
      funcOp = rewriter.create<func::FuncOp>(loc, funcName, funcType);
      funcOp.setPrivate();
    }

    // 4. 分配 Output MemRef
    auto resultTensorType = cast<RankedTensorType>(op.getResult(0).getType());
    auto outMemRefType = MemRefType::get(resultTensorType.getShape(), f32Type);
    
    SmallVector<Value> dynSizes;
    for (int i = 0; i < resultTensorType.getRank(); ++i) {
      if (resultTensorType.isDynamicDim(i)) {
        dynSizes.push_back(rewriter.create<memref::DimOp>(loc, qMemRef, i));
      }
    }
    Value outMemRef = rewriter.create<memref::AllocOp>(loc, outMemRefType, dynSizes);

    // 5. 辅助 Lambda: 获取维度并转为 i64
    auto getDimAsI64 = [&](Value memref, int dimIdx) -> Value {
      Value dimValue = rewriter.create<memref::DimOp>(loc, memref, dimIdx);
      return rewriter.create<arith::IndexCastOp>(loc, i64Type, dimValue);
    };

    Value batchSize = getDimAsI64(qMemRef, 0);
    Value seqLen    = getDimAsI64(qMemRef, 1);
    Value hiddenDim = getDimAsI64(qMemRef, 2);

    // 修复 Error 2: 正确提取未在 ODS 中定义的通用属性
    auto numHeadsAttr = op->getAttrOfType<IntegerAttr>("num_heads");
    int64_t numHeadsInt = numHeadsAttr ? numHeadsAttr.getInt() : 1;
    Value numHeads = rewriter.create<arith::ConstantIntOp>(loc, numHeadsInt, 64);

    auto scaleAttr = op->getAttrOfType<FloatAttr>("scale");
    // 将 double 显式转换为 float，确保精度匹配
    float scaleFloat = scaleAttr ? static_cast<float>(scaleAttr.getValueAsDouble()) : 0.0f;
    
    // 使用通用的 arith::ConstantOp 和内置的 getF32FloatAttr 生成器
    Value scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(scaleFloat));

    Value headSize = rewriter.create<arith::DivSIOp>(loc, hiddenDim, numHeads);

    // 6. 避开裸指针提取，直接传递 MemRef 到 func.call
    SmallVector<Value, 9> callOperands = {
        outMemRef, qMemRef, kMemRef, vMemRef,
        batchSize, seqLen, numHeads, headSize, scale
    };
    rewriter.create<func::CallOp>(loc, funcName, TypeRange{}, callOperands);

    // 7. 替换原操作
    SmallVector<Value, 3> replValues(3, Value());
    replValues[0] = outMemRef;
    rewriter.replaceOp(op, replValues);
    return success();
  }
};

} // namespace
void populateLowerCustomMHAPattern(RewritePatternSet &patterns) {
  patterns.insert<LowerCustomMHAPattern>(patterns.getContext());
}
// 注册 Pass
std::unique_ptr<Pass> npux::createNpuDPSConversionPass() {
  return std::make_unique<NpuDPSConversionPass>();
}