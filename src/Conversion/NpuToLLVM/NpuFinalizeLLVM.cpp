//==================================================================
// src/Conversion/NpuToLLVM/NpuFinalizeLLVM.cpp
//==================================================================

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "src/Pass/Passes.hpp" 

using namespace mlir;

namespace {

struct NpuMemRefToStructPattern : public OpRewritePattern<UnrealizedConversionCastOp> {
    const LLVMTypeConverter &converter;

    NpuMemRefToStructPattern(LLVMTypeConverter &converter, MLIRContext *context)
        : OpRewritePattern<UnrealizedConversionCastOp>(context), converter(converter) {}

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, PatternRewriter &rewriter) const override {
        // 1. 识别 Cast B: MemRef -> Struct
        Value inputMemRef = op.getInputs()[0];
        Type outputType = op.getResultTypes()[0];

        auto memRefType = llvm::dyn_cast<MemRefType>(inputMemRef.getType());
        auto structType = llvm::dyn_cast<LLVM::LLVMStructType>(outputType);
        
        if (!memRefType || !structType) {
            return failure();
        }

        // 2. 向上追溯 Cast A: Ptr -> MemRef
        auto prevCast = inputMemRef.getDefiningOp<UnrealizedConversionCastOp>();
        if (!prevCast) return failure();

        Value rawPtr = prevCast.getInputs()[0];
        if (!llvm::isa<LLVM::LLVMPointerType>(rawPtr.getType())) {
            return failure();
        }

        Location loc = op.getLoc();

        // ==============================================================
        // 3. 【核心修复】处理地址空间转换 (AddrSpaceCast)
        // ==============================================================
        // rawPtr 是 !llvm.ptr (Space 0)
        // structType 的第一个字段期望 !llvm.ptr<2> (Space 2)
        // 我们必须显式转换它，否则后续的 extractvalue 会类型错乱崩溃
        
        Value ptrToInsert = rawPtr;
        Type targetPtrType = structType.getBody()[0]; // 获取 Struct 第一个字段的类型

        if (rawPtr.getType() != targetPtrType) {
            // 生成 addrspacecast: ptr -> ptr<2>
            ptrToInsert = rewriter.create<LLVM::AddrSpaceCastOp>(
                loc, targetPtrType, rawPtr);
        }

        // ==============================================================
        // 4. 构建 LLVM MemRef Descriptor
        // ==============================================================
        Value desc = rewriter.create<LLVM::UndefOp>(loc, outputType);

        // Field 0: Allocated Pointer (使用转换后的指针)
        desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, ptrToInsert, ArrayRef<int64_t>{0});
        // Field 1: Aligned Pointer
        desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, ptrToInsert, ArrayRef<int64_t>{1});

        // Field 2: Offset
        Value zero = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
        desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, zero, ArrayRef<int64_t>{2});

        // Field 3 & 4: Sizes and Strides
        SmallVector<int64_t> strides;
        int64_t offset;
        if (failed(memRefType.getStridesAndOffset(strides, offset))) {
            return failure();
        }

        ArrayRef<int64_t> shape = memRefType.getShape();
        for (unsigned i = 0; i < shape.size(); ++i) {
            // Size
            Value s = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(shape[i]));
            desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, s, ArrayRef<int64_t>{3, i});
            
            // Stride
            Value st = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(strides[i]));
            desc = rewriter.create<LLVM::InsertValueOp>(loc, desc, st, ArrayRef<int64_t>{4, i});
        }

        rewriter.replaceOp(op, desc);
        return success();
    }
};

struct NpuStructToMemRefToPtrPattern : public OpRewritePattern<UnrealizedConversionCastOp> {
    
    using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, PatternRewriter &rewriter) const override {
        // 1. 识别 Cast C: MemRef -> Ptr (我们想要的结果)
        Value inputMemRef = op.getInputs()[0];
        Type outputType = op.getResultTypes()[0];

        // 输出必须是指针
        if (!llvm::isa<LLVM::LLVMPointerType>(outputType)) {
            return failure();
        }
        // 输入必须是 MemRef
        if (!llvm::dyn_cast<MemRefType>(inputMemRef.getType())) {
            return failure();
        }

        // 2. 向上追溯 Cast B: Struct -> MemRef
        auto prevCast = inputMemRef.getDefiningOp<UnrealizedConversionCastOp>();
        if (!prevCast) return failure();

        Value rawStruct = prevCast.getInputs()[0];
        Type structType = rawStruct.getType();

        // 确保源头是 Struct
        if (!llvm::isa<LLVM::LLVMStructType>(structType)) {
            return failure();
        }

        // ==============================================================
        // 3. 直接从 Struct 提取指针
        // ==============================================================
        Location loc = op.getLoc();

        // MemRef Descriptor 的第 1 个字段通常是 Aligned Pointer
        // 结构: {AllocatedPtr, AlignedPtr, Offset, Sizes, Strides}
        Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(
            loc, rawStruct, ArrayRef<int64_t>{1});

        // 4. 处理地址空间 (Address Space)
        // Struct 里的指针可能是 ptr<2>，但我们需要的是 ptr<0> (或者反之)
        Value finalPtr = alignedPtr;
        if (alignedPtr.getType() != outputType) {
            finalPtr = rewriter.create<LLVM::AddrSpaceCastOp>(
                loc, outputType, alignedPtr);
        }

        // 5. 替换 Cast C
        rewriter.replaceOp(op, finalPtr);
        
        return success();
    }
};

struct NpuFinalizeLLVMPass : public PassWrapper<NpuFinalizeLLVMPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuFinalizeLLVMPass)
    StringRef getArgument() const override { return "npu-finalize-llvm"; }
    
    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext *context = &getContext();
        
        LLVMTypeConverter converter(context);
        RewritePatternSet patterns(context);
        
        // 1. Ptr -> MemRef -> Struct
        patterns.add<NpuMemRefToStructPattern>(converter, context);
        
        // 2. 【新增】Struct -> MemRef -> Ptr
        patterns.add<NpuStructToMemRefToPtrPattern>(context);

        if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuFinalizeLLVMPass() {
    return std::make_unique<NpuFinalizeLLVMPass>();
}