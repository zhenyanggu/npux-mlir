//==============================================================
// src/Conversion/NpuBufferization/SignatureRewrite.cpp
// this file implements the NPU kernel signature rewrite pass
//==============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Pass/Passes.hpp"

using namespace mlir;

using namespace mlir::bufferization; // 【新增】方便使用 bufferization::ToTensorOp

namespace {

// --- Cleanup Pattern 1: 消除 Cast(To_Tensor(x)) -> x ---
struct RemoveUselessCasts : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, PatternRewriter &rewriter) const override {
    auto input = op.getOperand(0);
    if (auto toTensorOp = input.getDefiningOp<bufferization::ToTensorOp>()) {
        Value originalMemRef = toTensorOp.getOperation()->getOperand(0);
        
        if (op.getResult(0).getType() == originalMemRef.getType()) {
            rewriter.replaceOp(op, originalMemRef);
            return success();
        }
        if (mlir::dyn_cast<MemRefType>(op.getResult(0).getType())) {
            rewriter.replaceOpWithNewOp<memref::CastOp>(op, op.getResult(0).getType(), originalMemRef);
            return success();
        }
    }
    return failure();
  }
};

// --- Cleanup Pattern 2: 【新增】消除 To_Buffer(Cast(x)) -> x ---
// 专门解决你现在的报错: failed to legalize operation 'bufferization.to_buffer'
struct RemoveUselessToBuffer : public OpRewritePattern<bufferization::ToBufferOp> {
  using OpRewritePattern<bufferization::ToBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::ToBufferOp op, PatternRewriter &rewriter) const override {
    // 获取 to_buffer 的输入 (是一个 Tensor)
    Value input = op.getOperation()->getOperand(0);
    
    // 检查这个 Tensor 是否是由 unrealized_conversion_cast 产生的
    if (auto castOp = input.getDefiningOp<UnrealizedConversionCastOp>()) {
        // 获取 cast 的输入 (是一个 MemRef)
        Value originalMemRef = castOp.getOperand(0);
        
        // 如果输入不是 MemRef，跳过
        if (!mlir::dyn_cast<MemRefType>(originalMemRef.getType())) return failure();

        // 情况 A: 类型完全一致 -> 直接替换
        if (op.getType() == originalMemRef.getType()) {
            rewriter.replaceOp(op, originalMemRef);
            return success();
        }
        
        // 情况 B: 类型不一致 (Layout 不同) -> 插入 memref.cast
        if (mlir::dyn_cast<MemRefType>(op.getType())) {
            rewriter.replaceOpWithNewOp<memref::CastOp>(op, op.getType(), originalMemRef);
            return success();
        }
    }
    return failure();
  }
};

// --- 签名归一化逻辑 ---
static Type normalizeMemRefType(Type type) {
    if (auto memrefType = mlir::dyn_cast<MemRefType>(type)) {
        if (memrefType.getLayout().isIdentity())
            return type;
        return MemRefType::get(memrefType.getShape(), 
                               memrefType.getElementType(),
                               MemRefLayoutAttrInterface(), 
                               memrefType.getMemorySpace());
    }
    return type;
}

struct NpuSignatureRewritePass : public PassWrapper<NpuSignatureRewritePass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuSignatureRewritePass)

    StringRef getArgument() const override { return "npu-signature-rewrite"; }
    StringRef getDescription() const override { return "Normalize NPU signatures and cleanup bufferization artifacts"; }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        SymbolTable symbolTable(module);

        // --- 步骤 A: 修改签名 & 插入兼容性 Cast (保持不变) ---
        SmallVector<func::FuncOp> npuKernels;
        module.walk([&](func::FuncOp func) {
            if (func->hasAttr("npu.target")) npuKernels.push_back(func);
        });

        for (func::FuncOp func : npuKernels) {
            OpBuilder b(func.getContext());
            // 1. 计算新类型
            FunctionType oldFuncType = func.getFunctionType();
            std::vector<Type> newArgTypes;
            std::vector<Type> newResultTypes;
            for (Type t : oldFuncType.getInputs()) newArgTypes.push_back(normalizeMemRefType(t));
            for (Type t : oldFuncType.getResults()) newResultTypes.push_back(normalizeMemRefType(t));
            auto newFuncType = FunctionType::get(func.getContext(), newArgTypes, newResultTypes);

            if (newFuncType == oldFuncType) continue;

            // 2. 修复 Call Sites
            if (auto uses = symbolTable.getSymbolUses(func, module)) {
                for (auto use : *uses) {
                    if (auto callOp = dyn_cast<func::CallOp>(use.getUser())) {
                        b.setInsertionPoint(callOp);
                        for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
                            if (callOp.getOperand(i).getType() != newArgTypes[i]) {
                                Value cast = b.create<memref::CastOp>(callOp.getLoc(), newArgTypes[i], callOp.getOperand(i));
                                callOp.setOperand(i, cast);
                            }
                        }
                        for (unsigned i = 0; i < callOp.getNumResults(); ++i) callOp.getResult(i).setType(newResultTypes[i]);
                        b.setInsertionPointAfter(callOp);
                        for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
                            if (callOp.getResult(i).getType() != oldFuncType.getResult(i)) {
                                Value cast = b.create<memref::CastOp>(callOp.getLoc(), oldFuncType.getResult(i), callOp.getResult(i));
                                callOp.getResult(i).replaceAllUsesExcept(cast, cast.getDefiningOp());
                            }
                        }
                    }
                }
            }

            // 3. 修复函数内部
            Block &entryBlock = func.front();
            for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
                BlockArgument arg = entryBlock.getArgument(i);
                if (arg.getType() != newArgTypes[i]) {
                    Type oldType = arg.getType();
                    arg.setType(newArgTypes[i]);
                    b.setInsertionPointToStart(&entryBlock);
                    Value cast = b.create<memref::CastOp>(func.getLoc(), oldType, arg);
                    arg.replaceAllUsesExcept(cast, cast.getDefiningOp());
                }
            }
            func.walk([&](func::ReturnOp retOp) {
                for (unsigned i = 0; i < retOp.getNumOperands(); ++i) {
                    if (retOp.getOperand(i).getType() != newResultTypes[i]) {
                        b.setInsertionPoint(retOp);
                        Value cast = b.create<memref::CastOp>(retOp.getLoc(), newResultTypes[i], retOp.getOperand(i));
                        retOp.setOperand(i, cast);
                    }
                }
            });
            func.setType(newFuncType);
        }

        // --- 步骤 B: 执行 Cleanup Pattern ---
        RewritePatternSet patterns(&getContext());
        // 1. 消除 cast(to_tensor)
        patterns.add<RemoveUselessCasts>(&getContext());
        // 2. 【新增】消除 to_buffer(cast)
        patterns.add<RemoveUselessToBuffer>(&getContext());
        
        if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuSignatureRewritePass() {
  return std::make_unique<NpuSignatureRewritePass>();
}

static PassRegistration<NpuSignatureRewritePass> pass;