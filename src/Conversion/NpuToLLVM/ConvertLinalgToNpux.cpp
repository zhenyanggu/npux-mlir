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

    // =========================================================
    // B. Linalg Generic 限制 (修改点 1)
    // =========================================================
    target.addDynamicallyLegalOp<linalg::GenericOp>(
        [](linalg::GenericOp op) { 
            // 如果 Op 标记为 npu.target (Conv/Elewise)，则非法，需转换
            if (op->hasAttr("npu.target")) return false;
            
            // 如果 Op 标记为 npu.pp_stage (我们新加的量化节点)，也非法，需转换
            if (op->hasAttr("npu.pp_stage")) return false;

            return true; 
        });

    // C. FuncOp 限制 (保持不变)
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
    // D. AllocOp 限制 (修改点 2: 增加 Space 3)
    // =========================================================
    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      if (!isInNpuKernel(op)) return true;

      // 在 Kernel 内：
      // Space 0 (Host) -> 需转 npux.alloc
      // Space 2 (SPM)  -> 需转 npux.sram_alloc
      // Space 3 (ACC)  -> 需转 npux.acc_alloc
      // 只要是在 Kernel 里的 Alloc，基本上都要接管
      return false; 
    });

    // =========================================================
    // E. DeallocOp 限制 (修改点 2: 增加 Space 3)
    // =========================================================
    target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
        Value memref = op.getMemref();
        auto type = cast<MemRefType>(memref.getType());
        int space = type.getMemorySpaceAsInt();
        
        // 1. SRAM (Space 2) 和 ACC (Space 3) 的 dealloc 必须被转换
        if (space == 2 || space == 3) return false;

        // 2. DRAM (Space 0) 且在 Kernel 内 -> 需转 npux.free
        if (space == 0 && isInNpuKernel(op)) {
            return false;
        }
        
        return true; 
    });

    // =========================================================
    // F. CopyOp 限制 (已涵盖 Space 3)
    // =========================================================
    target.addDynamicallyLegalOp<memref::CopyOp>([&](memref::CopyOp op) {
        auto srcSpace = cast<MemRefType>(op.getSource().getType()).getMemorySpaceAsInt();
        auto dstSpace = cast<MemRefType>(op.getTarget().getType()).getMemorySpaceAsInt();
        
        // 只要源或目的涉及 Space 2 或 3，就需要转换为 DMA Op
        if (srcSpace == 2 || dstSpace == 2 || srcSpace == 3 || dstSpace == 3) return false;
        
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