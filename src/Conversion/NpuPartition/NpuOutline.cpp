//=============================================================================
// /src/Conversion/NpuPartition/NpuOutline.cpp
// this file implements the NPU outlining pass that extracts
// NPU clusters into independent functions.
//=============================================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

struct NpuOutlinePass
    : public PassWrapper<NpuOutlinePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuOutlinePass)

  llvm::StringRef getArgument() const override { return "onnx-npu-outline"; }
  llvm::StringRef getDescription() const override {
    return "Outline NPU clusters into independent functions.";
  }

  // 辅助函数：处理单个 Cluster
  void outlineCluster(ModuleOp module, func::FuncOp mainFunc, int clusterId,
      const std::vector<Operation *> &clusterOps) {
    OpBuilder builder(module.getContext());

    // === 1. 分析输入输出 ===
    llvm::SetVector<Value> inputs;
    llvm::SetVector<Value> outputs;
    llvm::DenseSet<Operation *> clusterOpSet(
        clusterOps.begin(), clusterOps.end());

    for (Operation *op : clusterOps) {
      // A. 找输入：如果操作数定义在 Cluster 外部，就是输入
      for (Value operand : op->getOperands()) {
        // 如果 operand 是 Block 参数 (比如函数参数)，肯定是输入
        // 如果 operand 的 DefiningOp 不在 clusterOpSet 里，也是输入
        Operation *defOp = operand.getDefiningOp();
        if (!defOp || clusterOpSet.find(defOp) == clusterOpSet.end()) {
          inputs.insert(operand);
        }
      }

      // B. 找输出：如果结果被 Cluster 外部使用了，就是输出
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (clusterOpSet.find(user) == clusterOpSet.end()) {
            outputs.insert(result);
            break; // 只要有一个外部 User，它就是 Output
          }
        }
      }
    }

    // === 2. 创建新函数 ===
    // 名字：npu_kernel_<id>
    std::string funcName = "npu_kernel_" + std::to_string(clusterId);

    // 类型：(inputs) -> (outputs)
    SmallVector<Type, 4> inputTypes;
    for (Value v : inputs)
      inputTypes.push_back(v.getType());

    SmallVector<Type, 4> outputTypes;
    for (Value v : outputs)
      outputTypes.push_back(v.getType());

    auto funcType = builder.getFunctionType(inputTypes, outputTypes);

    // 插入到 Module 末尾
    builder.setInsertionPointToEnd(module.getBody());
    auto newFunc =
        builder.create<func::FuncOp>(mainFunc.getLoc(), funcName, funcType);

    // 【关键】加上 target="npu" 属性，为了给后续的 Conversion Pass 识别
    newFunc->setAttr("npu.target", builder.getStringAttr("npu"));
    // 私有函数，不导出符号
    newFunc.setVisibility(SymbolTable::Visibility::Private);

    // === 3. 填充函数体 (Clone Ops) ===
    Block *entryBlock = newFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // 映射表：旧值 (Main里的) -> 新值 (Kernel里的 BlockArg 或 Clone 结果)
    IRMapping mapping; // 旧版 MLIR 请用 BlockAndValueMapping

    // 3.1 映射输入参数
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      mapping.map(input, entryBlock->getArgument(idx));
    }

    // 3.2 克隆 Op
    for (Operation *op : clusterOps) {
      builder.clone(*op, mapping);
    }

    // 3.3 创建 Return Op
    SmallVector<Value, 4> newReturns;
    for (Value oldOutput : outputs) {
      newReturns.push_back(mapping.lookup(oldOutput));
    }
    builder.create<func::ReturnOp>(mainFunc.getLoc(), newReturns);

    // === 4. 在原位置创建 Call Op 并替换 ===
// 获取 Cluster 的最后一个 Op
    Operation *lastOp = clusterOps.back();

    // 【核心修正】：将插入点设置在 lastOp 的“前面” (SetInsertionPoint)
    // 1. 依赖安全：lastOp 能运行，说明此时所有 Input 都 ready 了。
    // 2. 结构安全：即使 lastOp 是 Return，插在它前面也不会破坏 Block 的 Terminator 约束。
    builder.setInsertionPoint(lastOp);

    auto callOp = builder.create<func::CallOp>(
        lastOp->getLoc(), 
        newFunc, 
        inputs.getArrayRef()
    );

    for (auto [idx, oldOutput] : llvm::enumerate(outputs)) {
      // 这里的 oldOutput 可能是 const Value，导致无法调用 replaceAllUsesWith
      // 解决方法：直接复制一份 Value (Value
      // 本质是轻量级指针，复制是廉价且合法的)
      Value mutableOutput = oldOutput;
      mutableOutput.replaceAllUsesWith(callOp.getResult(idx));
    }

    bool erasedTerminator = false;
    for (Operation *op : clusterOps) {
      if (op->hasTrait<OpTrait::IsTerminator>()) {
        erasedTerminator = true;
      }
    }


    // 4.2 删掉旧 Op (注意要按反向顺序删，防止 use-def
    // 链报错，或者直接最后统一删)
    for (auto it = clusterOps.rbegin(); it != clusterOps.rend(); ++it) {
      (*it)->erase();
    }

    // 【补丁】：如果刚才删掉了 Terminator，现在 Block 结尾应该是 CallOp。
    // 我们需要在 CallOp 后面补一个 Return。
    if (erasedTerminator) {
        // 将 Builder 移到 Block 的最末端 (也就是 CallOp 后面)
        builder.setInsertionPointToEnd(lastOp->getBlock());
        // 创建新的 Return
        builder.create<func::ReturnOp>(mainFunc.getLoc(), callOp.getResults());
    }

  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // 我们假设所有的 NPU Op 都在 FuncOp 里面 (通常是 main)
    // 遍历所有非 NPU 的函数 (防止递归处理)
    SmallVector<func::FuncOp, 4> funcsToProcess;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (!func->hasAttr("npu.target")) {
        funcsToProcess.push_back(func);
      }
    }

    for (auto func : funcsToProcess) {
      // 1. 扫描并按照 Cluster ID 分组
      // 使用 MapVector 保持插入顺序，这对保持 Op 的拓扑序很重要！
      llvm::MapVector<int, std::vector<Operation *>> clusters;

      func.walk([&](Operation *op) {
        if (auto attr = op->getAttrOfType<IntegerAttr>("npu.cluster_id")) {
          int id = attr.getInt();
          clusters[id].push_back(op);
        }
      });

      // 2. 对每个 Cluster 执行 Outline
      for (auto &[id, ops] : clusters) {
        outlineCluster(module, func, id, ops);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuOutlinePass() {
  return std::make_unique<NpuOutlinePass>();
}

static PassRegistration<NpuOutlinePass> pass;