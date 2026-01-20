//=============================================================================
// src/Conversion/NpuPartition/NpuLowerPack.cpp
// This pass lowers linalg.pack and linalg.unpack ops to tensor ops 
// (pad, expand_shape, transpose, etc.) using the provided utility functions.
//=============================================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace mlir::linalg;

namespace {

//=============================================================================
// Pattern: LowerPackPattern
// Wraps linalg::lowerPack to convert linalg.pack into pad + expand + transpose
//=============================================================================
struct LowerPackPattern : public OpRewritePattern<linalg::PackOp> {
  using OpRewritePattern<linalg::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::PackOp packOp, PatternRewriter &rewriter) const override {
    
    // 调用提供的 lowerPack 函数
    // 参数 lowerPadLikeWithInsertSlice = true (通常设为 true 以优化纯 Padding 场景)
    FailureOr<LowerPackResult> res = linalg::lowerPack(
        rewriter, packOp, /*lowerPadLikeWithInsertSlice=*/true);

    if (failed(res)) {
      return failure();
    }

    // lowerPack 内部已经完成了 replaceOp，这里只需要返回 success
    return success();
  }
};

//=============================================================================
// Pattern: LowerUnPackPattern
// Wraps linalg::lowerUnPack to convert linalg.unpack into transpose + collapse + slice
//=============================================================================
struct LowerUnPackPattern : public OpRewritePattern<linalg::UnPackOp> {
  using OpRewritePattern<linalg::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::UnPackOp unPackOp, PatternRewriter &rewriter) const override {
    
    // 调用提供的 lowerUnPack 函数
    // 参数 lowerUnpadLikeWithExtractSlice = true
    FailureOr<LowerUnPackOpResult> res = linalg::lowerUnPack(
        rewriter, unPackOp, /*lowerUnpadLikeWithExtractSlice=*/true);

    if (failed(res)) {
      return failure();
    }

    return success();
  }
};

//=============================================================================
// Pass Definition
//=============================================================================
struct NpuLowerPackPass
    : public PassWrapper<NpuLowerPackPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuLowerPackPass)
  
  llvm::StringRef getArgument() const override { return "npu-lower-pack"; }
  
  llvm::StringRef getDescription() const override {
    return "Lower linalg.pack and linalg.unpack to tensor ops (pad, reshape, transpose).";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 添加 Lowering Patterns
    patterns.add<LowerPackPattern, LowerUnPackPattern>(context);

    // 使用 Greedy Driver 应用 Patterns
    // 这里只进行基本的 Canonicalization 即可
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true)
        .enableFolding(true);

    if (failed(applyPatternsGreedily(
            getOperation().getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

// 工厂函数实现
std::unique_ptr<Pass> npux::createNpuLowerPackPass() {
  return std::make_unique<NpuLowerPackPass>();
}