//===========================================
// src/Conversion/NpuPartition/ModifyScfRegionEncoding.cpp
//
// Description:
// This pass directly mutates the tensor encodings of scf.execute_region's
// external inputs (captured values) and yielded outputs to encoding=1,
// WITHOUT using any cast operations.
//===========================================

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

#include "src/Pass/Passes.hpp"  

using namespace mlir;

namespace {

// 辅助函数：检查 RankedTensorType 是否已经具有 encoding = 1
bool hasEncoding1(RankedTensorType type) {
  if (!type.getEncoding()) return false;
  if (auto intAttr = dyn_cast<IntegerAttr>(type.getEncoding())) {
    return intAttr.getInt() == 1;
  }
  return false;
}

// 辅助函数：将 RankedTensorType 的 encoding 修改为 1
RankedTensorType addEncoding1(RankedTensorType type, PatternRewriter &rewriter) {
  auto encoding = rewriter.getI64IntegerAttr(1);
  return RankedTensorType::get(type.getShape(), type.getElementType(), encoding);
}

struct ModifyScfRegionEncodingPattern : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern<scf::ExecuteRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp execOp,
                                PatternRewriter &rewriter) const override {
    bool needsChange = false;

    // ==============================================================
    // Logic 1: Determine New Result Types
    // ==============================================================
    SmallVector<Type> newResultTypes;
    for (Type type : execOp.getResultTypes()) {
      auto tensorType = dyn_cast<RankedTensorType>(type);
      if (tensorType && !hasEncoding1(tensorType)) {
        newResultTypes.push_back(addEncoding1(tensorType, rewriter));
        needsChange = true;
      } else {
        newResultTypes.push_back(type);
      }
    }

    // ==============================================================
    // Logic 2: Find Captured External Inputs
    // ==============================================================
    llvm::SetVector<Value> capturedInputs;
    execOp.getRegion().walk([&](Operation *innerOp) {
      for (Value operand : innerOp->getOperands()) {
        // 如果该 operand 的定义不在当前的 scf.execute_region 内，则是外部捕获
        if (operand.getParentRegion() != &execOp.getRegion()) {
          auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
          if (tensorType && !hasEncoding1(tensorType)) {
            capturedInputs.insert(operand);
            needsChange = true;
          }
        }
      }
    });

    if (!needsChange) {
      return failure();
    }

    // ==============================================================
    // Logic 3: In-place Type Mutation (No Casts)
    // ==============================================================
    
    // 3.1 强行修改外部捕获变量的类型
    for (Value captured : capturedInputs) {
      auto tensorType = cast<RankedTensorType>(captured.getType());
      captured.setType(addEncoding1(tensorType, rewriter));
    }

    // 3.2 强行修改 Region 内 scf.yield 操作数的类型 (即 yield 前产生这些值的 Op 的返回类型)
    execOp.getRegion().walk([&](scf::YieldOp yieldOp) {
      for (Value operand : yieldOp.getOperands()) {
        auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
        if (tensorType && !hasEncoding1(tensorType)) {
           // 这会直接修改定义这个 operand 的算子 (例如你的 linalg.generic) 的 Result Type
           operand.setType(addEncoding1(tensorType, rewriter));
        }
      }
    });

    // ==============================================================
    // Logic 4: Recreate scf.execute_region to update its return types
    // ==============================================================
    // 因为 Op 的 Result Type 无法直接原地修改，必须通过创建一个新 Op 来替换旧 Op
    auto newExecOp = rewriter.create<scf::ExecuteRegionOp>(execOp.getLoc(), newResultTypes);
    
    // 把原 region 里的代码直接搬进新的 region
    rewriter.inlineRegionBefore(execOp.getRegion(), newExecOp.getRegion(), newExecOp.getRegion().end());

    // 替换旧的 execute_region
    // 外部消费 execOp 返回值的算子会自动接管 newExecOp 的结果（这些结果现在已经是 encoding=1）
    rewriter.replaceOp(execOp, newExecOp.getResults());

    return success();
  }
};

struct ModifyScfRegionEncodingPass : public PassWrapper<ModifyScfRegionEncodingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModifyScfRegionEncodingPass)

  StringRef getArgument() const override { return "npu-modify-scf-encoding"; }
  
  StringRef getDescription() const override { 
    return "Directly mutate scf.execute_region external inputs and yielded outputs to encoding 1 without casts"; 
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    
    patterns.add<ModifyScfRegionEncodingPattern>(context);
    
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true).enableFolding(true);

    if (failed(applyPatternsGreedily(getOperation().getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createModifyScfRegionEncodingPass() {
  return std::make_unique<ModifyScfRegionEncodingPass>();
}