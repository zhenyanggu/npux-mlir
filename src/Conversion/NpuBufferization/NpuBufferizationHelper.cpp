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
} // namespace

// 注册 Pass
std::unique_ptr<Pass> npux::createNpuDPSConversionPass() {
  return std::make_unique<NpuDPSConversionPass>();
}