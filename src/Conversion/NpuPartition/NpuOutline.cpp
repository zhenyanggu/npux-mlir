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
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Transforms/RegionUtils.h"

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

    // === 1. 使用官方工具分析 Captures (修复点) ===
    // getUsedValuesDefinedAbove 会自动找到所有在 Region 内使用但定义在 Region 外的值
    llvm::SetVector<Value> captures;
    mlir::getUsedValuesDefinedAbove(region, captures);

    // === 2. 创建 Kernel 函数 ===
    SmallVector<Type> inputTypes;
    for (Value v : captures)
      inputTypes.push_back(v.getType());

    ResultRange results = executeOp.getResults();
    SmallVector<Type> outputTypes(results.getTypes().begin(), results.getTypes().end());

    std::string kernelName = getUniqueKernelName(module, "npu_kernel");
    auto funcType = builder.getFunctionType(inputTypes, outputTypes);

    builder.setInsertionPointToEnd(module.getBody());
    auto kernelFunc = builder.create<func::FuncOp>(executeOp.getLoc(), kernelName, funcType);
    kernelFunc->setAttr("npu.target", builder.getStringAttr("npu"));
    kernelFunc.setVisibility(SymbolTable::Visibility::Private);

    // === 3. 填充函数体 ===
    Block *entryBlock = kernelFunc.addEntryBlock();
    IRMapping mapping;

    // 3.1 映射 Captures -> BlockArgs
    for (auto [idx, captureVal] : llvm::enumerate(captures)) {
      mapping.map(captureVal, entryBlock->getArgument(idx));
    }

    builder.setInsertionPointToStart(entryBlock);

    // 3.2 Clone Op (排除 Terminator)
    Block &srcBlock = region.front();
    for (Operation &op : srcBlock.without_terminator()) {
      builder.clone(op, mapping);
    }

    // 3.3 处理 Terminator (scf.yield -> func.return)
    auto yieldOp = cast<scf::YieldOp>(srcBlock.getTerminator());
    SmallVector<Value> returnOperands;
    for (Value val : yieldOp.getOperands()) {
      // 必须从 mapping 中查找。如果是内部值，肯定在 map 里；如果是 capture，也在 map 里。
      Value mappedVal = mapping.lookup(val);
      
      // 【安全检查】如果 mappedVal 为空，说明前面的 clone 或 capture 分析有误
      // 这通常意味着 val 是一个未被捕获的外部值，或者 clone 遗漏了。
      if (!mappedVal) {
        // Fallback: 如果它是一个常量或未被捕获的外部值（理论上不应发生），尝试直接使用
        // 但为了调试，我们这里做一个断言，因为这往往是 Crash 的根源
        llvm::errs() << "Error: Value not mapped for return: " << val << "\n";
        assert(mappedVal && "Yield operand not found in mapping! Use-Def chain broken.");
      }
      returnOperands.push_back(mappedVal);
    }
    builder.create<func::ReturnOp>(yieldOp.getLoc(), returnOperands);

    // === 4. 替换调用 ===
    builder.setInsertionPoint(executeOp);
    auto callOp = builder.create<func::CallOp>(
        executeOp.getLoc(),
        kernelFunc,
        captures.getArrayRef()
    );

    // 替换 execute_region 的结果用途
    executeOp.replaceAllUsesWith(callOp.getResults());

    // === 5. 安全删除 ===
    // 此时 executeOp 应该是无用的。
    // 如果 executeOp 内部还有 Op 被外部引用（理论上不可能，除非 IR 非法），erase 会崩溃。
    // Drop all references explicitly just in case (though erase does this).
    executeOp.erase();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<scf::ExecuteRegionOp> regionsToOutline;

    // 收集
    module.walk([&](func::FuncOp func) {
      if (func->hasAttr("npu.target")) return;
      func.walk([&](scf::ExecuteRegionOp op) {
        regionsToOutline.push_back(op);
      });
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