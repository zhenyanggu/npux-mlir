//=============================================================================
// /src/Conversion/NpuPartition/ConvertONNXToLinalgNpu.cpp
// this file implements the NPU partitioning pass that labels
// ONNX operations for NPU execution based on a conversion registry.
//=============================================================================


#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h" // 引入贪婪重写驱动
#include "src/Support/NPUConversionRegistry.hpp"     // 引入你的 Registry
#include "src/Pass/Passes.hpp" // Pass 声明
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace npux {
    void registerNpuOpConversions();
}


namespace {

struct ONNXToLinalgNpuPass : public PassWrapper<ONNXToLinalgNpuPass, OperationPass<func::FuncOp>> {
    
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXToLinalgNpuPass)

    StringRef getArgument() const override { return "convert-npu-onnx-to-linalg"; }
    StringRef getDescription() const override { return "Lower ONNX ops to Linalg inside NPU kernels."; }

    void runOnOperation() override {

        npux::registerNpuOpConversions();
        
        func::FuncOp func = getOperation();

        // 1. 【卫语句】只处理 NPU 函数
        // 如果这个函数没有 target="npu" 属性，直接跳过，不碰 CPU 代码
        if (auto attr = func->getAttrOfType<StringAttr>("npu.target")) {
            if (attr.getValue() != "npu") return;
        } else {
            return;
        }

        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);

        // 2. 【这里就是 Populate 的位置！】
        // 从 Registry 加载所有去中心化注册的 Pattern (如 GeluToLinalg)
        npux::NPUConversionRegistry::populatePatterns(patterns);

        // 3. 应用转换
        // 使用贪婪策略：反复应用 Pattern 直到收敛
        // FrozenRewritePatternSet 有助于多线程性能，但这里直接用 patterns 也可以
        FrozenRewritePatternSet frozenPatterns(std::move(patterns));

        if (failed(applyPatternsGreedily(func, frozenPatterns))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> npux::createONNXToLinalgNpuPass() {
    return std::make_unique<ONNXToLinalgNpuPass>();
}

static PassRegistration<ONNXToLinalgNpuPass> pass;