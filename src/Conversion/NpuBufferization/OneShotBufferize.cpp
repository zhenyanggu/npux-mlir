//======================================================
// src/Conversion/NpuBufferization/OneShotBufferize.cpp
// this file implements the NPU bufferization pass
//======================================================

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace mlir::bufferization;

namespace {

// 【改回 ModuleOp】必须是 Module 级别，才能同时更新 Caller 和 Callee
struct NpuBufferizationPass : public PassWrapper<NpuBufferizationPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuBufferizationPass)

    StringRef getArgument() const override { return "npu-bufferize"; }
    StringRef getDescription() const override { return "Bufferize NPU kernels and update call sites"; }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        
        OneShotBufferizationOptions options;
        options.bufferizeFunctionBoundaries = true; 

        // 【关键】精确制导的白名单过滤器
        options.opFilter.allowOperation([&](Operation *op) {
            // 1. 对于 FuncOp：只允许带 "npu.target" 的函数进行 Bufferization
            //    这意味着 main_graph (没有该属性) 将会被跳过，其签名(Tensor)保持不变。
            if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
                return funcOp->hasAttr("npu.target");
            }

            // 2. 必须允许 CallOp
            //    即使它在 main_graph 里面，我们也需要处理它，以便更新它调用的目标符号
            //    并在调用点插入 cast (to_memref)。
            if (isa<func::CallOp>(op)) {
                return true;
            }

            // 3. 允许 Module 和 Return (为了结构完整性)
            if (isa<ModuleOp, func::ReturnOp>(op)) {
                return true;
            }

            // 4. 允许 NPU Kernel 内部的所有 Op
            if (auto parentFunc = op->getParentOfType<func::FuncOp>()) {
                if (parentFunc->hasAttr("npu.target")) {
                    return true;
                }
            }

            // 5. 其他一切 (如 main_graph 里的 onnx.Conv 等) 禁止
            return false;
        });

        BufferizationState state;
        
        // 现在 Bufferize 既不会报错 "onnx op not bufferized" (因为被 filter 掉了)，
        // 也不会报错 "call site mismatch" (因为 CallOp 被允许处理，它会自动插入 cast)
        if (failed(bufferization::runOneShotBufferize(module, options, state))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuBufferizationPass() {
  return std::make_unique<NpuBufferizationPass>();
}

