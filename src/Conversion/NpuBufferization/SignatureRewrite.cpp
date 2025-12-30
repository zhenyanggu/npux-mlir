//==============================================================
// src/Conversion/NpuBufferization/SignatureRewrite.cpp
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
using namespace mlir::bufferization;

namespace {

// --- Cleanup Pattern 1: 消除 Cast(To_Tensor(x)) -> x ---
struct RemoveUselessCasts : public OpRewritePattern<UnrealizedConversionCastOp> {
    using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, PatternRewriter &rewriter) const override {
        auto input = op.getOperand(0);
        
        // 检查输入是否是 ToTensorOp
        if (auto toTensorOp = input.getDefiningOp<bufferization::ToTensorOp>()) {
            // 【修复】使用 getOperand(0) 替代 getMemref()
            Value originalMemRef = toTensorOp->getOperand(0); 
            Type targetType = op.getResult(0).getType();

            // 1. 类型完全一致 -> 直接替换
            if (targetType == originalMemRef.getType()) {
                rewriter.replaceOp(op, originalMemRef);
                return success();
            }
            
            // 2. 都是 MemRef 但类型不同 (如 Layout/Strides 不同) -> 插入 memref.cast
            if (mlir::dyn_cast<MemRefType>(targetType) && mlir::dyn_cast<MemRefType>(originalMemRef.getType())) {
                rewriter.replaceOpWithNewOp<memref::CastOp>(op, targetType, originalMemRef);
                return success();
            }
        }
        return failure();
    }
};

// --- Cleanup Pattern 2: 消除 To_Buffer(Cast(x)) -> x ---
struct RemoveUselessToBuffer : public OpRewritePattern<bufferization::ToBufferOp> {
    using OpRewritePattern<bufferization::ToBufferOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(bufferization::ToBufferOp op, PatternRewriter &rewriter) const override {
        // 【修复】使用 getOperand(0) 替代 getBufferTensor() / getTensor()
        // 这样无论 API 怎么变，只要它是单操作数 Op 就能跑
        Value input = op->getOperand(0); 
        
        // 1. 穿透 UnrealizedConversionCast 或 ToTensor 找到最原始的 MemRef
        Value originalMemRef;
        if (auto castOp = input.getDefiningOp<UnrealizedConversionCastOp>()) {
            originalMemRef = castOp.getOperand(0);
        } else if (auto toTensorOp = input.getDefiningOp<bufferization::ToTensorOp>()) {
            // 【修复】这里也用 getOperand(0)
            originalMemRef = toTensorOp->getOperand(0);
        } else {
            return failure();
        }

        // 2. 比较原始 MemRef 和 to_buffer 的结果类型
        Type originalType = originalMemRef.getType();
        Type targetType = op.getType();

        if (!mlir::dyn_cast<MemRefType>(originalType)) return failure();

        // 情况 A: 类型完全一样 -> 直接替换
        if (originalType == targetType) {
            rewriter.replaceOp(op, originalMemRef);
            return success();
        }

        // 情况 B: 类型不一样 (Identity vs Dynamic Strides) -> 插入 Cast
        if (mlir::dyn_cast<MemRefType>(targetType)) {
            rewriter.replaceOpWithNewOp<memref::CastOp>(op, targetType, originalMemRef);
            return success();
        }
        
        return failure();
    }
};

// --- Cleanup Pattern 3: 消除 AllocTensor Copy ---
struct RemoveAllocTensorCopy : public OpRewritePattern<bufferization::AllocTensorOp> {
    using OpRewritePattern<bufferization::AllocTensorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(bufferization::AllocTensorOp op, PatternRewriter &rewriter) const override {
        // 获取 copy 操作数
        Value sourceTensor = op.getCopy();
        if (!sourceTensor) return failure();

        // 检查源 Tensor 是不是从某个 MemRef 转换来的
        bool isFromCast = false;
        if (sourceTensor.getDefiningOp<UnrealizedConversionCastOp>() || 
            sourceTensor.getDefiningOp<bufferization::ToTensorOp>()) {
            isFromCast = true;
        }

        if (isFromCast) {
            rewriter.replaceOp(op, sourceTensor);
            return success();
        }

        return failure();
    }
};

// --- Cleanup Pattern 4: 合并连续的 Cast ---
struct FoldMemRefCasts : public OpRewritePattern<memref::CastOp> {
    using OpRewritePattern<memref::CastOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(memref::CastOp op, PatternRewriter &rewriter) const override {
        Value input = op.getSource();
        if (auto prevCast = input.getDefiningOp<memref::CastOp>()) {
            rewriter.replaceOpWithNewOp<memref::CastOp>(op, op.getType(), prevCast.getSource());
            return success();
        }
        return failure();
    }
};

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
            FunctionType oldFuncType = func.getFunctionType();
            std::vector<Type> newArgTypes;
            std::vector<Type> newResultTypes;
            for (Type t : oldFuncType.getInputs()) newArgTypes.push_back(normalizeMemRefType(t));
            for (Type t : oldFuncType.getResults()) newResultTypes.push_back(normalizeMemRefType(t));
            auto newFuncType = FunctionType::get(func.getContext(), newArgTypes, newResultTypes);

            if (newFuncType == oldFuncType) continue;

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
        patterns.add<RemoveAllocTensorCopy>(&getContext());
        patterns.add<RemoveUselessToBuffer>(&getContext());
        patterns.add<RemoveUselessCasts>(&getContext());
        patterns.add<FoldMemRefCasts>(&getContext());

        if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuSignatureRewritePass() {
  return std::make_unique<NpuSignatureRewritePass>();
}