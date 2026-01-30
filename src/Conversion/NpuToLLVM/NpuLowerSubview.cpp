//===========================================
// src/Conversion/NpuToLLVM/LowerNpuSubview.cpp
//
// Description:
// This pass lowers standard memref.subview operations on NPU-managed memory
// (SRAM: Space 2, ACC: Space 3) to a custom 'npux.subview' operation.
//
// Standard memref.subview implies pointer arithmetic which is complex to 
// lower for hardware with simple "Base + Offset" addressing.
// By converting to npux.subview, we materialize all offsets (static and dynamic)
// into SSA values, facilitating direct offset calculation in later stages.
//===========================================

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

#include "src/Dialect/Npux/NpuxOps.hpp" 
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace npux;

namespace {

struct LowerSramSubviewPattern : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp subviewOp,
                                PatternRewriter &rewriter) const override {
    
    // 1. 获取源 MemRef 类型并检查内存空间
    Value source = subviewOp.getSource();
    auto sourceType = cast<MemRefType>(source.getType());
    int memorySpace = sourceType.getMemorySpaceAsInt();

    // 只处理 NPU 内部存储：2 (SRAM) 和 3 (ACC)
    if (memorySpace != 2 && memorySpace != 3) {
      return failure();
    }

    Location loc = subviewOp.getLoc();

    // ==============================================================
    // Logic: Materialize Offsets (Static -> ConstantOp)
    // ==============================================================
    // memref.subview 的 offset 可能是 Attribute (Static) 或 Value (Dynamic)。
    // npux.subview 要求全都是 Value，以便统一处理。
    
    SmallVector<Value> newOffsets;
    
    for (OpFoldResult ofr : subviewOp.getMixedOffsets()) {
      if (auto val = ofr.dyn_cast<Value>()) {
        // 已经是动态值，直接使用
        newOffsets.push_back(val);
      } else {
        // 是静态属性，创建 arith.constant
        int64_t staticOffset = cast<IntegerAttr>(ofr.get<Attribute>()).getInt();
        Value constOffset = rewriter.create<arith::ConstantIndexOp>(loc, staticOffset);
        newOffsets.push_back(constOffset);
      }
    }

    // ==============================================================
    // Logic: Replace Op
    // ==============================================================
    // 我们保留原本的 Result Type (包含 strided layout)，以便后续 pass 
    // 可能需要利用 stride 信息进行 DMA 配置计算。
    
    rewriter.replaceOpWithNewOp<npux::SubviewOp>(
        subviewOp,
        subviewOp.getType(), // Result Type
        source,              // Source MemRef
        newOffsets           // Materialized Offsets
    );

    return success();
  }
};

struct LowerNpuSubviewPass : public PassWrapper<LowerNpuSubviewPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerNpuSubviewPass)

  StringRef getArgument() const override { return "npu-lower-subview"; }
  
  StringRef getDescription() const override { 
    return "Convert memref.subview on SRAM/ACC to npux.subview for offset calculation"; 
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    
    patterns.add<LowerSramSubviewPattern>(context);
    
    // 使用与 reference 相同的贪婪策略配置
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true)
          .enableFolding(true);

    // 应用 Pattern 到当前 FuncOp 的 Body
    if (failed(applyPatternsGreedily(
            getOperation().getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createLowerNpuSubviewPass() {
  return std::make_unique<LowerNpuSubviewPass>();
}