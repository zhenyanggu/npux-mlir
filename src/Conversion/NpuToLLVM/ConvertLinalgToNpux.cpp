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

  StringRef getArgument() const override { return "convert-linalg-to-npux"; }
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
    target.addDynamicallyLegalOp<linalg::GenericOp>([](linalg::GenericOp op) {
      // 如果 Op 标记为 npu.target (Conv/Elewise)，则非法，需转换
      if (op->hasAttr("npu.target"))
        return false;
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
      Value memref = op.getMemref();
      auto type = cast<MemRefType>(memref.getType());
      int space = type.getMemorySpaceAsInt();
      return space == 0;
    });

    // =========================================================
    // E. DeallocOp 限制 (修改点 2: 增加 Space 3)
    // =========================================================
    target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
      Value memref = op.getMemref();
      auto type = cast<MemRefType>(memref.getType());
      int space = type.getMemorySpaceAsInt();
      return space == 0;
    });

    // =========================================================
    // F. CopyOp 限制 (已涵盖 Space 3)
    // =========================================================
    target.addDynamicallyLegalOp<memref::CopyOp>([&](memref::CopyOp op) {
      auto srcSpace =
          cast<MemRefType>(op.getSource().getType()).getMemorySpaceAsInt();
      auto dstSpace =
          cast<MemRefType>(op.getTarget().getType()).getMemorySpaceAsInt();

      // 只要源或目的涉及 Space 2 或 3，就需要转换为 DMA Op
      if (srcSpace == 2 || dstSpace == 2 || srcSpace == 3 || dstSpace == 3)
        return false;

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