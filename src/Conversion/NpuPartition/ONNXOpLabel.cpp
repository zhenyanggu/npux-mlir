//=============================================================================
// /src/Conversion/NpuPartition/ONNXOpLabel.cpp
// this file implements the NPU partitioning pass that labels
// ONNX operations for NPU execution based on a conversion registry.
//=============================================================================

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "src/Pass/Passes.hpp"

#include "src/Support/NPUConversionRegistry.hpp"


using namespace mlir;

namespace {

// 定义 Pass
struct ONNXOpLabelPass : public PassWrapper<ONNXOpLabelPass, OperationPass<ModuleOp>> {
    
    // Pass 的元数据
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXOpLabelPass)
    
    StringRef getArgument() const override { return "onnx-npu-op-label"; }
    StringRef getDescription() const override { return "Label ONNX ops for NPU execution based on registry checks."; }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext *context = &getContext();

        // 用于分配唯一的 Cluster ID
        // 从 0 开始计数，每次发现新的 NPU 孤岛就 +1
        int32_t globalClusterCounter = 0;

        // 核心逻辑：Walk (后序遍历通常比较方便处理 Producer-Consumer)
        // 但对于贪心聚类，前序(PreOrder)或默认顺序通常也可以，
        // 这里我们用默认顺序遍历，因为我们是向前看 Producer。
        module.walk([&](Operation *op) {
            
            // 1. 【核心检查】询问 Registry：这个 Op 硬件支持吗？
            // 这一步会触发你之前写的 isHardwareSupported 逻辑
            bool supported = npux::NPUConversionRegistry::isSupported(op);
    llvm::outs() << "Checking Op: " << op->getName() << " -> Supported? " << supported << "\n";
    
            if (!npux::NPUConversionRegistry::isSupported(op)) {
                return; // 不支持，跳过，留给 CPU
            }

            // 2. 【贪心聚类】尝试加入现有的 Cluster
            int32_t myClusterId = -1;

            // 遍历当前 Op 的所有输入操作数 (Operands)
            for (Value operand : op->getOperands()) {
                Operation *producer = operand.getDefiningOp();
                
                // 必须满足三个条件才能合并：
                // a. Producer 存在 (不是函数参数)
                // b. Producer 已经被标记为 NPU (说明它支持且已处理)
                // c. Producer 和当前 Op 在同一个 Block (防止跨越控制流边界，比如合并了循环内外的 Op)
                if (producer && 
                    producer->hasAttr("npu.target") && 
                    producer->getBlock() == op->getBlock()) {
                    
                    // 获取 Producer 的 ID
                    if (auto attr = producer->getAttrOfType<IntegerAttr>("npu.cluster_id")) {
                        myClusterId = attr.getInt();
                        // 策略：贪心。找到第一个能抱的大腿就抱上去，然后停止。
                        // (更复杂的策略可以使用 Union-Find 合并多个输入的 Cluster，但通常不需要)
                        break; 
                    }
                }
            }

            // 3. 【新立门户】如果没找到能加入的 Cluster，自己成为一个新的起点
            if (myClusterId == -1) {
                myClusterId = ++globalClusterCounter;
            }

            // 4. 【打标】写入 Attribute
            // 这些 Attribute 就是后续 Outline Pass 抓人的依据
            op->setAttr("npu.target", StringAttr::get(context, "npu"));
            op->setAttr("npu.cluster_id", IntegerAttr::get(IntegerType::get(context, 32), myClusterId));
            
            // 可选：打印调试信息
            // llvm::outs() << "Tagging " << op->getName() << " into Cluster " << myClusterId << "\n";
        });
    }
};

} // namespace

// 暴露创建函数
std::unique_ptr<Pass> npux::createONNXOpLabelPass() {
    return std::make_unique<ONNXOpLabelPass>();
}


static PassRegistration<ONNXOpLabelPass> pass;
