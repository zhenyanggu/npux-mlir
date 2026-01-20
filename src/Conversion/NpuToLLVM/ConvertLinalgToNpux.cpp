//=====================================================
// src/Conversion/NpuToLLVM/ConvertLinalgToNpux.cpp
// this file implements convert linalg ops to custom npux ops
//=====================================================

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Pass/Passes.hpp"

#include "src/Conversion/NpuToLLVM/NpuxConversionHelper.hpp"
#include "src/Dialect/Npux/NpuxOps.hpp"

using namespace mlir;
using namespace npux;

namespace {

// =========================================================
// 辅助函数：判断 Op 是否在 NPU Kernel 内部
// =========================================================
static bool isInNpuKernel(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) return false;
  // 只要函数有 npu.target = "npu" 属性，就认为是 Kernel
  if (auto attr = funcOp->getAttrOfType<StringAttr>("npu.target")) {
    return attr.getValue() == "npu";
  }
  return false;
}

struct ConvertLinalgToNpuPass
    : public PassWrapper<ConvertLinalgToNpuPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToNpuPass)

  StringRef getArgument() const override {
    return "convert-linalg-to-npux";
  }
  StringRef getDescription() const override {
    return "Lower npu-related linalgs op to npux ops";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    // 1. 定义转换目标
    ConversionTarget target(*context);

    // A. Npux Dialect 是合法的 (目标)
    target.addLegalDialect<NpuxDialect>();
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect>();

    // B. Linalg Generic 限制
    target.addDynamicallyLegalOp<linalg::GenericOp>(
        [](linalg::GenericOp op) { return !op->hasAttr("npu.target"); });

    // C. FuncOp 限制 (用于插入 npux.init)
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        if (op.getName() == "main_graph" || op.getName() == "main") {
            if (!op.getBody().empty()) {
                auto &entryBlock = op.getBody().front();
                if (entryBlock.getOps<npux::InitOp>().empty()) {
                    return false; 
                }
            }
        }
        return true;
    });

    // =========================================================
    // D. AllocOp 限制 (简化版)
    // 逻辑：Space 0 + 在 NPU Kernel 内 = 非法 (需转 npux.alloc)
    // =========================================================
    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      if (isInNpuKernel(op)) {
          return false; 
      }
      return true; 
    });


    target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
        Value memref = op.getMemref();
        auto type = cast<MemRefType>(memref.getType());
        int space = type.getMemorySpaceAsInt();
        
        // 1. SRAM (Space 2) 的 dealloc 必须被移除 -> 非法
        if (space == 2) return false;

        // 2. DRAM (Space 0) 且在 Kernel 内 -> 必须转 npux.free -> 非法
        if (space == 0 && isInNpuKernel(op)) {
            return false;
        }
        
        return true; 
    });

    // =========================================================
    // F. CopyOp 限制
    // 逻辑：只要涉及 SRAM (Space 2)，就是 DMA -> 非法
    // =========================================================
    target.addDynamicallyLegalOp<memref::CopyOp>([&](memref::CopyOp op) {
        auto srcSpace = cast<MemRefType>(op.getSource().getType()).getMemorySpaceAsInt();
        auto dstSpace = cast<MemRefType>(op.getTarget().getType()).getMemorySpaceAsInt();
        
        // 只要源或目的有一个是 Space 2，就需要转换为 DMA Op
        if (srcSpace == 2 || dstSpace == 2) return false;
        
        return true;
    });

    // 2. 收集 Patterns
    RewritePatternSet patterns(context);
    
    npux::populateLinalgToNpuxPatterns(patterns);

    if (failed(applyPartialConversion(
            getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> npux::createConvertLinalgToNpuPass() {
  return std::make_unique<ConvertLinalgToNpuPass>();
}