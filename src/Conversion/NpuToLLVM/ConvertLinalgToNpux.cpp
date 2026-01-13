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

  static bool isNpuManagedMemref(Value memref) {
  // 1. 必须是 Space 0 (Host Memory)
  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type || type.getMemorySpaceAsInt() != 0) return false;

  // 2. 检查用途：是否被 NPU 相关的 Op 使用
  for (Operation *user : memref.getUsers()) {
    // A. 直接传给 NPU Kernel (CallOp)
    if (auto callOp = dyn_cast<func::CallOp>(user)) {
      auto module = callOp->getParentOfType<ModuleOp>();
      auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
      if (callee && callee->hasAttr("npu.target")) return true;
    }
    
    // B. 被用作 NPU DMA (CopyOp)
    // 检查这个 memref 是否参与了和 SRAM (Space 2) 的交互
    if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
       Value src = copyOp.getSource();
       Value dst = copyOp.getTarget();
       auto srcType = cast<MemRefType>(src.getType());
       auto dstType = cast<MemRefType>(dst.getType());
       
       // 如果我是源，且目标是 SRAM -> 我是 Input
       if (src == memref && dstType.getMemorySpaceAsInt() == 2) return true;
       // 如果我是目标，且源是 SRAM -> 我是 Output
       if (dst == memref && srcType.getMemorySpaceAsInt() == 2) return true;
    }

    // C. 如果在同一个 Pass 中 alloc 已经被转成了 npux.alloc，
    // 那么它的 DeallocOp 的操作数可能已经是 npux.alloc 的结果了
    // 这种情况下，user 不再是判定标准，而是 definingOp
  }
  
  // 3. 补充检查：DefiningOp 是否已经是 npux.alloc
  // 这是为了处理 DialectConversion 中途状态或多次 Pass 的情况
  if (memref.getDefiningOp<npux::AllocOp>()) return true;

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

    target.addDynamicallyLegalOp<linalg::GenericOp>(
        [](linalg::GenericOp op) { return !op->hasAttr("npu.target"); });


    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        if (op.getName() == "main_graph" || op.getName() == "main") {
            // 如果函数体非空，且开头不是 npux.init，则认为非法，触发 InsertNpuLifecyclePattern
            if (!op.getBody().empty()) {
                auto &entryBlock = op.getBody().front();
                if (entryBlock.getOps<npux::InitOp>().empty()) {
                    return false; 
                }
            }
        }
        return true;
    });

    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      // 1. 如果已经是 NPU Space (1) 或者 SRAM (2)，则是合法的
      if (op.getType().getMemorySpaceAsInt() != 0) return true;

      // 2. 检查用途：如果被传入了带有 "npu.target" 的函数，则非法
      for (Operation *user : op->getUsers()) {
        if (auto callOp = dyn_cast<func::CallOp>(user)) {
           // 查找被调用的函数
           auto module = op->getParentOfType<ModuleOp>();
           auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
           if (callee && callee->hasAttr("npu.target")) {
             return false; // 非法！强制触发 HostAllocToNpuxPattern
           }
        }
      }
      return true; // 其他情况（纯CPU使用）是合法的
    });

    // 4. AllocOp 限制 (Space 0 且被 NPU 使用的必须转 npux.alloc)
    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      if (op.getType().getMemorySpaceAsInt() != 0) return true; 
      // 使用提取出来的逻辑：如果是 NPU 内存，则非法 -> 转 npux.alloc
      if (isNpuManagedMemref(op.getResult())) return false; 
      return true;
    });

    // =========================================================
    // 5. 新增: CopyOp 限制
    // 任何涉及 Space 2 (SRAM) 的 Copy 都是非法的，必须转为 DMA
    // =========================================================
    target.addDynamicallyLegalOp<memref::CopyOp>([&](memref::CopyOp op) {
        auto srcSpace = cast<MemRefType>(op.getSource().getType()).getMemorySpaceAsInt();
        auto dstSpace = cast<MemRefType>(op.getTarget().getType()).getMemorySpaceAsInt();
        if (srcSpace == 2 || dstSpace == 2) return false;
        return true;
    });

    // =========================================================
    // 6. 新增: DeallocOp 限制
    // =========================================================
    target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
        Value memref = op.getMemref();
        auto type = cast<MemRefType>(memref.getType());
        
        // A. SRAM (Space 2) Dealloc -> 非法 (Pattern里会 erase 掉)
        if (type.getMemorySpaceAsInt() == 2) return false;

        // B. Host (Space 0) Dealloc
        // 如果这个 memref 被判定为 NPU 管理的内存 -> 非法 (必须转 npux.free)
        if (isNpuManagedMemref(memref)) return false; 
        
        return true; // 普通 CPU Dealloc -> 合法
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