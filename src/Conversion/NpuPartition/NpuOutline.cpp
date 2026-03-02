//=============================================================================
// /src/Conversion/NpuPartition/NpuOutline.cpp
// this file implements the NPU outlining pass that extracts
// NPU clusters into independent functions.
//=============================================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

struct NpuOutlinePass
    : public PassWrapper<NpuOutlinePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuOutlinePass)

  llvm::StringRef getArgument() const override { return "npu-outline"; }
  llvm::StringRef getDescription() const override {
    return "Outline scf.execute_region blocks into independent NPU kernels.";
  }

  std::string getUniqueKernelName(ModuleOp module, StringRef prefix) {
    int id = 0;
    while (module.lookupSymbol(prefix.str() + "_" + std::to_string(id))) {
      id++;
    }
    return prefix.str() + "_" + std::to_string(id);
  }

  void outlineRegion(ModuleOp module, scf::ExecuteRegionOp executeOp) {
    OpBuilder builder(module.getContext());
    Region &region = executeOp.getRegion();

    // === 1. 分析所有捕获的值 ===
    llvm::SetVector<Value> allCaptures;
    mlir::getUsedValuesDefinedAbove(region, allCaptures);

    // === 2. 区分：哪些作为参数传递，哪些在内部克隆 ===
    SmallVector<Value> dynamicArgs;      // 真正的函数参数
    SmallVector<Value> constantsToClone; // 需要在内部克隆的常数

    for (Value v : allCaptures) {
      Operation *defOp = v.getDefiningOp();
      // 检查是否为常数操作 (如 arith.constant)
      if (defOp && defOp->hasTrait<OpTrait::ConstantLike>()) {
        constantsToClone.push_back(v);
      } else {
        dynamicArgs.push_back(v);
      }
    }

    // === 3. 创建 Kernel 函数（仅使用动态参数） ===
    SmallVector<Type> inputTypes;
    for (Value v : dynamicArgs)
      inputTypes.push_back(v.getType());

    ResultRange results = executeOp.getResults();
    SmallVector<Type> outputTypes(
        results.getTypes().begin(), results.getTypes().end());

    std::string kernelName = getUniqueKernelName(module, "npu_kernel");
    auto funcType = builder.getFunctionType(inputTypes, outputTypes);

    builder.setInsertionPointToEnd(module.getBody());
    auto kernelFunc =
        builder.create<func::FuncOp>(executeOp.getLoc(), kernelName, funcType);
    kernelFunc->setAttr("npu.target", builder.getStringAttr("npu"));
    kernelFunc.setVisibility(SymbolTable::Visibility::Private);

    // === 4. 填充函数体 ===
    Block *entryBlock = kernelFunc.addEntryBlock();
    IRMapping mapping;

    builder.setInsertionPointToStart(entryBlock);

    // 4.1 处理动态参数映射
    for (auto [idx, argVal] : llvm::enumerate(dynamicArgs)) {
      mapping.map(argVal, entryBlock->getArgument(idx));
    }

    // 4.2 【关键修改】在函数内部克隆常数操作
    for (Value constVal : constantsToClone) {
      Operation *constOp = constVal.getDefiningOp();
      // 克隆常数操作到 Kernel 开头，并建立映射
      Operation *clonedOp = builder.clone(*constOp, mapping);
      // 注意：这里假设常数 Op 只有一个结果
      mapping.map(constVal, clonedOp->getResult(0));
    }

    // 4.3 Clone 主体逻辑 (保持不变)
    Block &srcBlock = region.front();
    for (Operation &op : srcBlock.without_terminator()) {
      builder.clone(op, mapping);
    }

    // 4.4 处理 Return (保持不变)
    auto yieldOp = cast<scf::YieldOp>(srcBlock.getTerminator());
    SmallVector<Value> returnOperands;
    for (Value val : yieldOp.getOperands()) {
      returnOperands.push_back(mapping.lookup(val));
    }
    builder.create<func::ReturnOp>(yieldOp.getLoc(), returnOperands);

    // === 5. 更新调用处（仅传递动态参数） ===
    builder.setInsertionPoint(executeOp);
    auto callOp = builder.create<func::CallOp>(executeOp.getLoc(), kernelFunc,
        dynamicArgs // 这里只传动态值
    );

    executeOp.replaceAllUsesWith(callOp.getResults());
    executeOp.erase();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<scf::ExecuteRegionOp> regionsToOutline;

    // 收集
    module.walk([&](func::FuncOp func) {
      if (func->hasAttr("npu.target"))
        return;
      func.walk(
          [&](scf::ExecuteRegionOp op) { regionsToOutline.push_back(op); });
    });

    // 执行
    for (auto op : regionsToOutline) {
      outlineRegion(module, op);
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuOutlinePass() {
  return std::make_unique<NpuOutlinePass>();
}