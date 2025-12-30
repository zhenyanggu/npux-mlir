//==================================================================
// src/Conversion/NpuToLLVM/NpuInline.cpp
// Force inline functions marked with "npu.target"
//==================================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h" // 【核心】必须包含这个头文件
#include "src/Pass/Passes.hpp"
#include "mlir/IR/IRMapping.h"


using namespace mlir;

namespace {

struct NpuInlinePass : public PassWrapper<NpuInlinePass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuInlinePass)

    StringRef getArgument() const override { return "npu-inline"; }
    StringRef getDescription() const override { return "Manually inline NPU kernels without generic inliner interface"; }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        SymbolTable symbolTable(module);

        SmallVector<func::CallOp> callsToInline;

        // 1. 收集所有目标调用
        module.walk([&](func::CallOp call) {
            auto callee = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
                call, StringAttr::get(call.getContext(), call.getCallee()));
            if (callee && callee->hasAttr("npu.target")) {
                callsToInline.push_back(call);
            }
        });

        for (func::CallOp call : callsToInline) {
            auto callee = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
                call, StringAttr::get(call.getContext(), call.getCallee()));
            
            // --------------------------------------------------------
            // 手动内联核心逻辑
            // --------------------------------------------------------
            OpBuilder b(call);
            Region &calleeRegion = callee.getBody();
            IRMapping mapper;

            // A. 映射参数：Kernel 的 Argument -> Call 的 Operand
            Block &calleeEntryBlock = calleeRegion.front();
            for (unsigned i = 0; i < calleeEntryBlock.getNumArguments(); ++i) {
                mapper.map(calleeEntryBlock.getArgument(i), call.getOperand(i));
            }

            // B. 搬运 Op：将 Kernel 里的所有 Op 克隆到 Call 的位置
            // 注意：我们假设 Kernel 只有一个 Block (或者简单的 CFG)，这对 NPU Kernel 通常成立
            // 如果有多个 Block，我们需要 splitBlock 并 splice
            
            // 简单起见，我们只克隆 Entry Block 里的操作 (跳过最后的 Return)
            for (Operation &op : calleeEntryBlock.without_terminator()) {
                b.clone(op, mapper);
            }

            // C. 处理返回值
            // 找到 Kernel 的 Terminator (通常是 func.return)
            Operation *terminator = calleeEntryBlock.getTerminator();
            if (auto retOp = dyn_cast<func::ReturnOp>(terminator)) {
                // 如果 Call 有返回值，用 Kernel Return 的操作数替换它
                if (call.getNumResults() > 0) {
                    for (unsigned i = 0; i < retOp.getNumOperands(); ++i) {
                        Value retVal = retOp.getOperand(i);
                        // 查找克隆后的新 Value
                        Value newRetVal = mapper.lookup(retVal);
                        call.getResult(i).replaceAllUsesWith(newRetVal);
                    }
                }
            }

            // D. 删除 Call Op
            call.erase();
        }

        // 2. 清理无用的 Kernel 函数
        module.walk([&](func::FuncOp func) {
            // 只删除标记了 npu.target 且是私有的函数
            if (func->hasAttr("npu.target") && func.isPrivate()) {
                
                // 【关键修改】使用 symbolKnownUseEmpty 来判断是否还有人用它
                // 注意：这里使用静态方法，传入 module 作为查找范围
                if (SymbolTable::symbolKnownUseEmpty(func, module)) {
                    func.erase();
                }
            }
        });
    }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuInlinePass() {
  return std::make_unique<NpuInlinePass>();
}

