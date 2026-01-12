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