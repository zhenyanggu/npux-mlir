//==============================================
// src/Conversion/NpuToLLVM/NpuMemoryPlacement.cpp
// NPU Memory Placement Pass
// Defines the boundary between Host and NPU
//==============================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

const int HOST_SPACE = 0;
const int SRAM_SPACE = 1;        // Internal
const int SHARED_DRAM_SPACE = 2; // Interface

static MemRefType setMemorySpace(MemRefType type, int space) {
    return MemRefType::get(type.getShape(), 
                           type.getElementType(),
                           type.getLayout(),
                           OpBuilder(type.getContext()).getI64IntegerAttr(space));
}

static void propagateMemorySpace(Value value, int space) {
    for (Operation *user : value.getUsers()) {
        if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
            auto oldType = mlir::cast<MemRefType>(subview.getType());
            if (oldType.getMemorySpaceAsInt() == space) continue;
            auto newType = setMemorySpace(oldType, space);
            subview.getResult().setType(newType);
            propagateMemorySpace(subview.getResult(), space);
        }
        else if (auto castOp = dyn_cast<memref::CastOp>(user)) {
            auto oldType = mlir::cast<MemRefType>(castOp.getType());
            if (oldType.getMemorySpaceAsInt() == space) continue;
            auto newType = setMemorySpace(oldType, space);
            castOp.getResult().setType(newType);
            propagateMemorySpace(castOp.getResult(), space);
        }
    }
}

struct NpuMemoryPlacementPass : public PassWrapper<NpuMemoryPlacementPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuMemoryPlacementPass)
    StringRef getArgument() const override { return "npu-memory-placement"; }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        SymbolTable symbolTable(module);

        SmallVector<func::FuncOp> npuKernels;
        module.walk([&](func::FuncOp func) {
            if (func->hasAttr("npu.target")) npuKernels.push_back(func);
        });

        for (func::FuncOp func : npuKernels) {
            OpBuilder b(func.getContext());
            
            // -------------------------------------------------------------
            // 0. 【关键修复】预先分析哪些 Alloc 被 Return 了
            // -------------------------------------------------------------
            DenseSet<Value> returnedValues;
            func.walk([&](func::ReturnOp retOp) {
                for (Value operand : retOp.getOperands()) {
                    returnedValues.insert(operand);
                }
            });

            // -------------------------------------------------------------
            // 1. 处理 Kernel 内部 Alloc
            // -------------------------------------------------------------
            func.walk([&](memref::AllocOp allocOp) {
                auto oldType = allocOp.getType();
                if (oldType.getMemorySpaceAsInt() == HOST_SPACE) {
                    // 【关键逻辑】
                    // 如果是被 Return 的 -> Space 2 (Shared)
                    // 如果只是内部用 -> Space 1 (SRAM)
                    int targetSpace = returnedValues.count(allocOp.getResult()) ? SHARED_DRAM_SPACE : SRAM_SPACE;
                    
                    auto newType = setMemorySpace(oldType, targetSpace);
                    allocOp.getResult().setType(newType);
                    propagateMemorySpace(allocOp.getResult(), targetSpace);
                }
            });

            // -------------------------------------------------------------
            // 2. 处理 Kernel 接口 (Signature)
            // -------------------------------------------------------------
            FunctionType oldFuncType = func.getFunctionType();
            std::vector<Type> newArgTypes;
            std::vector<Type> newResultTypes;

            // Args -> Space 2
            for (Type t : oldFuncType.getInputs()) {
                if (auto memrefT = mlir::dyn_cast<MemRefType>(t))
                    newArgTypes.push_back(setMemorySpace(memrefT, SHARED_DRAM_SPACE));
                else
                    newArgTypes.push_back(t);
            }
            // Results -> Space 2
            for (Type t : oldFuncType.getResults()) {
                if (auto memrefT = mlir::dyn_cast<MemRefType>(t))
                    newResultTypes.push_back(setMemorySpace(memrefT, SHARED_DRAM_SPACE));
                else
                    newResultTypes.push_back(t);
            }
            func.setType(FunctionType::get(func.getContext(), newArgTypes, newResultTypes));

            // 更新 Block Arguments (Args)
            Block &entryBlock = func.front();
            for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
                BlockArgument arg = entryBlock.getArgument(i);
                if (arg.getType() != newArgTypes[i]) {
                    arg.setType(newArgTypes[i]);
                    propagateMemorySpace(arg, SHARED_DRAM_SPACE);
                }
            }

            // -------------------------------------------------------------
            // 3. 更新 Call Sites
            // -------------------------------------------------------------
            if (auto uses = symbolTable.getSymbolUses(func, module)) {
                for (auto use : *uses) {
                    if (auto callOp = dyn_cast<func::CallOp>(use.getUser())) {
                        b.setInsertionPoint(callOp);
                        
                        // A. 输入参数处理 (保持你原本的 Copy 逻辑，虽然慢但安全)
                        for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
                            Value hostVal = callOp.getOperand(i);
                            // 获取目标类型 (已经是 Space 2 了)
                            Type targetType = newArgTypes[i]; 
                            
                            if (auto memrefTarget = mlir::dyn_cast<MemRefType>(targetType)) {
                                auto currentType = mlir::cast<MemRefType>(hostVal.getType());
                                // 只有当输入不是 Space 2 时才拷贝
                                if (currentType.getMemorySpaceAsInt() != SHARED_DRAM_SPACE) {
                                    Value sharedAlloc = b.create<memref::AllocOp>(callOp.getLoc(), memrefTarget);
                                    b.create<memref::CopyOp>(callOp.getLoc(), hostVal, sharedAlloc);
                                    callOp.setOperand(i, sharedAlloc);
                                }
                            }
                        }

                        // B. 结果类型更新 + 兼容性处理 (修复点)
                        for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
                            Value oldResult = callOp.getResult(i);
                            Type newResType = newResultTypes[i]; // Space 2
                            
                            // 1. 先更新 Call 的结果类型，让 verify 通过
                            oldResult.setType(newResType);

                            // 2. 处理下游兼容性
                            // 下游的 Op (比如 main 函数里的 return 或其他计算) 可能还在期待 Space 0
                            // 我们需要欺骗它们，说这个 Space 2 的结果就是 Space 0
                            if (!oldResult.use_empty()) {
                                b.setInsertionPointAfter(callOp);
                                
                                auto memrefT = mlir::cast<MemRefType>(newResType);
                                auto hostSpaceType = setMemorySpace(memrefT, HOST_SPACE);

                                // 【修复】使用 UnrealizedConversionCast 而不是 MemorySpaceCast
                                // 这相当于告诉编译器：在此处建立一个桥梁，后续的 Pass (NpuAlloc) 会消除它
                                Value castedBack = b.create<UnrealizedConversionCastOp>(
                                    callOp.getLoc(), hostSpaceType, oldResult).getResult(0);
                                
                                // 安全替换：除了 castedBack 自己的定义，其他所有用到 oldResult 的地方都换掉
                                oldResult.replaceAllUsesExcept(castedBack, castedBack.getDefiningOp());
                            }
                        }
                    }
                }
            }
        }
    }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuMemoryPlacementPass() {
    return std::make_unique<NpuMemoryPlacementPass>();
}

static PassRegistration<NpuMemoryPlacementPass> pass;