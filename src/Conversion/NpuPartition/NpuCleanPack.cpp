//=============================================================================
// src/Conversion/NpuPartition/NpuCleanPack.cpp
// This file eliminates redundant pack/unpack pairs that occur between regions.
// E.g., unpack (NCHWc32 -> NCHW) followed by pack (NCHW -> NCHWc32)
//=============================================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

//=============================================================================
// Pattern: RemoveRedundantPackUnpack
// Matches: %repacked = linalg.pack(%unpacked) ...
//          where %unpacked = linalg.unpack(%original) ...
// Actions: Replace %repacked with %original if configs match.
//=============================================================================
struct RemoveRedundantPackUnpack : public OpRewritePattern<linalg::PackOp> {
  using OpRewritePattern<linalg::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::PackOp packOp, PatternRewriter &rewriter) const override {
    
    // 1. 获取 Pack 的输入源
    Value packInput = packOp.getSource();
    
    // 2. 向上追踪，查看输入源是否来自 UnPackOp
    auto unPackOp = packInput.getDefiningOp<linalg::UnPackOp>();
    if (!unPackOp)
      return failure();

    // 3. 获取原始的 Packed Tensor (UnPack 的输入)
    Value originalPacked = unPackOp.getSource();

    // 4. 验证类型兼容性 (Type Consistency)
    if (originalPacked.getType() != packOp.getResult().getType()) {
      return failure();
    }

    // 5. 验证参数一致性 (Configuration Consistency)
    
    // 5.1 检查 Inner Dimensions Position (e.g., [1])
    if (packOp.getInnerDimsPos() != unPackOp.getInnerDimsPos()) {
      return failure();
    }

    // 5.2 检查 Inner Tiles (替换了 getMixedInnerTiles)
    // 策略：分别比较“静态定义”和“动态数值”，这在所有版本 MLIR 中都可用。

    // A. 比较静态属性 (Static Attributes)
    // 这里的数组包含了静态数值（如 32）和动态占位符（通常是 -1）
    // 如果两个 Op 的配置一样，这个数组必须完全相同。
    if (packOp.getStaticInnerTiles() != unPackOp.getStaticInnerTiles()) {
      return failure();
    }

    // B. 比较动态操作数 (Dynamic Operands)
    // 如果有动态 Tile，这里存储了对应的 SSA Value。
    // 我们需要确保它们引用的是同一个 Value。
    auto packDynamicTiles = packOp.getInnerTiles();
    auto unPackDynamicTiles = unPackOp.getInnerTiles();

    if (packDynamicTiles.size() != unPackDynamicTiles.size()) {
      return failure();
    }

    for (auto it : llvm::zip(packDynamicTiles, unPackDynamicTiles)) {
      if (std::get<0>(it) != std::get<1>(it)) {
        return failure();
      }
    }

    // 5.3 检查 Outer Dims Permutation (如果有)
    if (packOp.getOuterDimsPerm() != unPackOp.getOuterDimsPerm()) {
       return failure();
    }

    // 6. 执行替换 (Optimization)
    // 用 originalPacked 直接替换 packOp 的结果
    rewriter.replaceOp(packOp, originalPacked);

    return success();
  }
};

//=============================================================================
// Pass Definition
//=============================================================================

struct NpuCleanPackPass
    : public PassWrapper<NpuCleanPackPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuCleanPackPass)
  llvm::StringRef getArgument() const override { return "npu-clean-pack"; }
  llvm::StringRef getDescription() const override {
    return "Remove redundant linalg.unpack followed by linalg.pack operations.";
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 添加消除冗余 Pack/Unpack 的 Pattern
    patterns.add<RemoveRedundantPackUnpack>(context);

    // 配置 Greedy Driver
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true)
        .enableFolding(true) 
        .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);

    if (failed(applyPatternsGreedily(
            getOperation().getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

// 工厂函数实现
std::unique_ptr<Pass> npux::createNpuCleanPackPass() {
  return std::make_unique<NpuCleanPackPass>();
}