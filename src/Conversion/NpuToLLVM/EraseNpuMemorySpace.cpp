//===========================================
// src/Conversion/NpuToLLVM/EraseNpuMemorySpace.cpp
//
// Description:
// This pass erases memory space '1' (NPU DRAM) from all memrefs,
// converting them to the default memory space '0'. 
// This is useful for Unified Memory Architectures where CPU and NPU
// share the same physical DRAM, allowing standard LLVM lowering 
// to process them without address space mismatch errors.
//===========================================

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

#include "src/Dialect/Npux/NpuxOps.hpp" 
#include "src/Pass/Passes.hpp"

// 如果你的函数签名 (入参) 也包含 space 1，可能需要取消下面头文件的注释
// #include "mlir/Dialect/Func/Transforms/FuncConversions.h"

using namespace mlir;
using namespace npux;

namespace {

// ==============================================================
// 1. 定义 TypeConverter：指导 MLIR 如何全局替换类型
// ==============================================================
class MemorySpaceConverter : public TypeConverter {
public:
  MemorySpaceConverter() {
    // 默认规则：保留所有其他不相关的类型
    addConversion([](Type type) { return type; });

    // 核心规则：将 MemRef 的 Space 1 改为 Space 0 (nullptr)
    addConversion([](MemRefType type) -> Type {
      if (type.getMemorySpaceAsInt() == 1) {
        return MemRefType::Builder(type).setMemorySpace(nullptr);
      }
      return type;
    });

    // 适配规则：兼容 UnrankedMemRef 的情况
    addConversion([](UnrankedMemRefType type) -> Type {
      if (type.getMemorySpaceAsInt() == 1) {
        return UnrankedMemRefType::get(type.getElementType(), nullptr);
      }
      return type;
    });
  }
};

// ==============================================================
// 2. 泛型 Pattern：拦截并重建所有受类型变化影响的 Operation
// ==============================================================
struct EraseSpaceGenericPattern : public ConversionPattern {
  EraseSpaceGenericPattern(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    
    // 我们不在这里处理 func.func，避免破坏函数签名机制
    if (isa<func::FuncOp>(op)) {
      return failure();
    }

    // 检查：操作数的类型或返回值的类型是否发生了变化？
    SmallVector<Type> newResultTypes;
    bool hasTypeChange = false;
    for (Type type : op->getResultTypes()) {
      Type convertedType = typeConverter->convertType(type);
      newResultTypes.push_back(convertedType);
      if (convertedType != type) hasTypeChange = true;
    }

    bool hasOperandChange = false;
    for (auto [orig, updated] : llvm::zip(op->getOperands(), operands)) {
      if (orig != updated) hasOperandChange = true;
    }

    // 如果该 Op 与 Space 1 毫无关系，则忽略（返回 failure 让框架跳过）
    if (!hasTypeChange && !hasOperandChange) {
      return failure();
    }

    // ==============================================================
    // Logic: Replace Op (重新生成算子，赋予其干净的 Space 0 类型)
    // ==============================================================
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResultTypes, op->getAttrs(), op->getSuccessors());
    
    // 将原 Op 中包含的 Regions 完整迁移到新 Op 中 (如 scf.for/scf.if 等)
    for (Region &region : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(region, *newRegion, newRegion->begin());
    }

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    
    return success();
  }
};

// ==============================================================
// 3. 定义 Pass 本体
// ==============================================================
struct EraseNpuMemorySpacePass : public PassWrapper<EraseNpuMemorySpacePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EraseNpuMemorySpacePass)

  StringRef getArgument() const override { return "npu-erase-memory-space"; }
  
  StringRef getDescription() const override { 
    return "Erase memory space 1 from memrefs, converting them to default unified space (0)"; 
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    func::FuncOp func = getOperation();

    MemorySpaceConverter typeConverter;
    RewritePatternSet patterns(context);

    // 注册我们编写的泛型拦截 Pattern
    patterns.add<EraseSpaceGenericPattern>(typeConverter, context);

    // *注意*：如果你的 `func.func` 的入参（arguments）也有 memref<..., 1>，
    // 需要加上这一行来重写函数签名。如果没有，则不需要。
    // mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);

    // 配置 Conversion Target (合法性校验目标)
    ConversionTarget target(*context);
    
    // 动态合法性判断：只要该 Operation 里面的所有类型，
    // 都能被 TypeConverter 原封不动地通过（即不包含 Space 1），就是合法的。
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (isa<func::FuncOp>(op)) return true; // 放行函数定义本身
      return typeConverter.isLegal(op);
    });

    // 使用 Partial Conversion 应用 Pattern
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createEraseNpuMemorySpacePass() {
  return std::make_unique<EraseNpuMemorySpacePass>();
}