//==================================================================
// src/Conversion/NpuToLLVM/NpuInstructionLowering.cpp
// this file is for lowering high level NPU ops to the calls of NPU API
//==================================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/PassManager.h" 
#include "mlir/Transforms/Passes.h"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

const int HOST_SPACE = 0;
const int NPU_SRAM_SPACE = 1;
const int SHARED_DRAM_SPACE = 2;

const int SFU_OP_GELU = 1;
const int SFU_OP_ADD = 3;

// --- 辅助函数 ---

Value createI32(PatternRewriter &rewriter, Location loc, int32_t val) {
  return rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(val));
}
Value createI16(PatternRewriter &rewriter, Location loc, int16_t val) {
  return rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI16Type(), rewriter.getI16IntegerAttr(val));
}
Value createI8(PatternRewriter &rewriter, Location loc, int8_t val) {
  return rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(val));
}
Value createBool(PatternRewriter &rewriter, Location loc, bool val) {
  return rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI1Type(), rewriter.getBoolAttr(val));
}

static func::FuncOp getOrInsertFunc(OpBuilder &builder, ModuleOp module, 
                                    StringRef funcName, TypeRange operandTypes,
                                    TypeRange resultTypes = {}) {
  // 1. 先查找是否已存在
  if (auto func = module.lookupSymbol<func::FuncOp>(funcName))
    return func;
  
  // 2. 保存插入点 (RAII)
  OpBuilder::InsertionGuard guard(builder);
  
  // 3. 设置插入点到 Module 开始
  builder.setInsertionPointToStart(module.getBody());
  
  // 4. 创建函数
  auto funcOp = builder.create<func::FuncOp>(module.getLoc(), funcName, 
      FunctionType::get(builder.getContext(), operandTypes, resultTypes));
  
  funcOp.setPrivate();
  return funcOp;
}

// 获取 SRAM Offset
Value getSramOffset(
    PatternRewriter &rewriter, Location loc, Value memrefValue) {
  Operation *defOp = memrefValue.getDefiningOp();
  if (auto allocOp = dyn_cast_or_null<memref::AllocOp>(defOp)) {
    if (auto attr = allocOp->getAttrOfType<IntegerAttr>("npu.sram_offset")) {
      return createI32(rewriter, loc, (int32_t)attr.getInt());
    }
  }
  // 处理 SubView: 如果有 SubView，通常它指向的 BaseAlloc 带有 offset
  // 简化处理：假设 subview 的 offset 在 lower 阶段已经被处理，这里递归找 Base
  if (auto subview = dyn_cast_or_null<memref::SubViewOp>(defOp)) {
    return getSramOffset(rewriter, loc, subview.getSource());
  }
  return createI32(rewriter, loc, 0);
}

// 【关键新增】提取布局信息的结构体
struct LayoutInfo {
  int64_t stride;    // 行步长 (Row Stride)
  int64_t colNum;    // 原始宽度 (用于 API 的 input)
  int64_t rowNum;    // 原始高度
  int64_t colNumReg; // 寄存器值 (Width - 1)
  int64_t rowNumReg; // 寄存器值 (Height - 1)
};

// 【关键新增】从 MemRef 类型计算 Stride 和 Shape
// 能够正确处理 subview 产生的 strided layout
LogicalResult getLayoutInfo(MemRefType type, LayoutInfo &info) {
  ArrayRef<int64_t> shape = type.getShape();
  int rank = type.getRank();

  if (rank < 2)
    return failure(); // 暂时只支持 >= 2D

  // 1. 获取 Shape 信息
  info.rowNum = shape[rank - 2];
  info.colNum = shape[rank - 1];
  info.rowNumReg = info.rowNum - 1; // N-1
  info.colNumReg = info.colNum - 1; // N-1

  // 2. 获取 Stride 信息
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(type.getStridesAndOffset(strides, offset))) {
    return failure();
  }

  // 对于 2D 图片搬运，Stride 通常是倒数第二维的 stride
  // 例如 shape [1, 16, 64, 64], strides [65536, 4096, 64, 1]
  // Row Stride = 64
  info.stride = strides[rank - 2];

  return success();
}

// 辅助：获取元素大小 (bytes)
int64_t getElementSize(Type type) {
  if (type.isF32() || type.isInteger(32))
    return 4;
  if (type.isInteger(16) || type.isF16())
    return 2;
  if (type.isInteger(8))
    return 1;
  return 1; // Default
}

// 辅助：生成指针计算代码 (把 MemRef 变成 void*)
Value getFlatPointer(PatternRewriter &rewriter, Location loc, Value memrefVal) {
    // 1. 拆解 MemRef
    auto metadata = rewriter.create<memref::ExtractStridedMetadataOp>(loc, memrefVal);
    Value baseBuffer = metadata.getBaseBuffer();
    Value offsetIndex = metadata.getOffset();
    
    // 2. Base MemRef -> LLVM Ptr
    Value basePtr = rewriter.create<UnrealizedConversionCastOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext()), baseBuffer).getResult(0);

    // 3. 计算字节偏移 (index -> i64)
    Type elemType = mlir::cast<MemRefType>(memrefVal.getType()).getElementType();
    Value vElemBytes = rewriter.create<arith::ConstantIndexOp>(loc, getElementSize(elemType));
    Value byteOffsetIndex = rewriter.create<arith::MulIOp>(loc, offsetIndex, vElemBytes);
    Value byteOffsetI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), byteOffsetIndex);

    // 4. GEP 计算最终指针
    Value finalPtr = rewriter.create<LLVM::GEPOp>(
        loc, 
        LLVM::LLVMPointerType::get(rewriter.getContext()), 
        rewriter.getI8Type(), 
        basePtr, 
        ValueRange{byteOffsetI64}
    );
    return finalPtr;
}

// ==============================================================================
// Pattern: Lowering Alloc (Space 2) -> npu_mem_alloc
// ==============================================================================
struct LowerNpuAllocPattern : public OpRewritePattern<memref::AllocOp> {
    using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(memref::AllocOp op, PatternRewriter &rewriter) const override {
        if (op.getType().getMemorySpaceAsInt() != SHARED_DRAM_SPACE) return failure();

        Location loc = op.getLoc();
        MemRefType type = op.getType();
        
        int64_t totalSize = 1;
        for(int64_t dim : type.getShape()) totalSize *= dim;
        totalSize *= getElementSize(type.getElementType());

        // Call npu_mem_alloc
        ModuleOp module = op->getParentOfType<ModuleOp>();
        auto allocFunc = getOrInsertFunc(rewriter, module, "npu_mem_alloc", 
            {rewriter.getI64Type()}, 
            {LLVM::LLVMPointerType::get(rewriter.getContext())} // Ret: !llvm.ptr
        );
        
        Value vSize = rewriter.create<arith::ConstantIntOp>(loc, totalSize, 64);
        Value vPtr = rewriter.create<func::CallOp>(loc, allocFunc, ValueRange{vSize}).getResult(0);

        // Cast Ptr -> MemRef (保持 IR 合法性，直到 ToLLVM 阶段)
        Value memrefCast = rewriter.create<UnrealizedConversionCastOp>(
            loc, type, vPtr).getResult(0);

        rewriter.replaceOp(op, memrefCast);
        return success();
    }
};

struct LowerDmaCopyPattern : public OpRewritePattern<memref::CopyOp> {
    using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(memref::CopyOp op, PatternRewriter &rewriter) const override {
        Value src = op.getSource();
        Value dst = op.getTarget();
        
        auto srcType = mlir::cast<MemRefType>(src.getType());
        auto dstType = mlir::cast<MemRefType>(dst.getType());
        int srcSpace = srcType.getMemorySpaceAsInt();
        int dstSpace = dstType.getMemorySpaceAsInt();

        Location loc = op.getLoc();
        ModuleOp module = op->getParentOfType<ModuleOp>();

        // Case A: MVIN (Host/DRAM -> SRAM)
        if (srcSpace == SHARED_DRAM_SPACE && dstSpace == NPU_SRAM_SPACE) {
            LayoutInfo srcInfo, dstInfo;
            if (failed(getLayoutInfo(srcType, srcInfo)) || failed(getLayoutInfo(dstType, dstInfo))) return failure();

            // 【关键】计算 Host 指针，消化掉 SubView
            Value vHostPtrRaw = getFlatPointer(rewriter, loc, src);
            Value vSramAddr = getSramOffset(rewriter, loc, dst);

            SmallVector<Type> types = {
                LLVM::LLVMPointerType::get(rewriter.getContext()), // void*
                rewriter.getI32Type(), rewriter.getI16Type(), rewriter.getI16Type(),
                rewriter.getI16Type(), rewriter.getI16Type(), rewriter.getI8Type(),
                rewriter.getI8Type(), rewriter.getI8Type(), rewriter.getI1Type(),
                rewriter.getI32Type(), rewriter.getI16Type(), rewriter.getI16Type()
            };
            auto funcOp = getOrInsertFunc(rewriter, module, "npu_dma_mvin", types);

            rewriter.create<func::CallOp>(loc, funcOp, ValueRange{
                vHostPtrRaw, vSramAddr, 
                createI16(rewriter, loc, dstInfo.colNumReg), createI16(rewriter, loc, dstInfo.rowNumReg), 
                createI16(rewriter, loc, dstInfo.stride), createI16(rewriter, loc, srcInfo.stride),
                createI8(rewriter, loc, 1), createI8(rewriter, loc, 0), createI8(rewriter, loc, 0), 
                createBool(rewriter, loc, false), createI32(rewriter, loc, 0), 
                createI16(rewriter, loc, 1), createI16(rewriter, loc, 0)
            });
            rewriter.eraseOp(op);
            return success();
        }

        // Case B: MVOUT (SRAM -> Host/DRAM)
        if (srcSpace == NPU_SRAM_SPACE && dstSpace == SHARED_DRAM_SPACE) {
            LayoutInfo srcInfo, dstInfo;
            if (failed(getLayoutInfo(srcType, srcInfo)) || failed(getLayoutInfo(dstType, dstInfo))) return failure();

            // 【关键】计算 Host 指针 (MVOUT 的 Target 是 Host)
            Value vHostPtrRaw = getFlatPointer(rewriter, loc, dst);
            Value vSramAddr = getSramOffset(rewriter, loc, src);

            SmallVector<Type> types = {
                LLVM::LLVMPointerType::get(rewriter.getContext()), // void*
                rewriter.getI32Type(), rewriter.getI16Type(), rewriter.getI16Type(),
                rewriter.getI16Type(), rewriter.getI16Type(), rewriter.getI8Type(),
                rewriter.getI8Type(), rewriter.getI8Type(), rewriter.getI1Type(),
                rewriter.getI32Type(), rewriter.getI16Type(), rewriter.getI16Type()
            };
            auto funcOp = getOrInsertFunc(rewriter, module, "npu_dma_mvout", types);

            rewriter.create<func::CallOp>(loc, funcOp, ValueRange{
                vHostPtrRaw, vSramAddr, 
                createI16(rewriter, loc, srcInfo.colNumReg), createI16(rewriter, loc, srcInfo.rowNumReg), 
                createI16(rewriter, loc, srcInfo.stride), createI16(rewriter, loc, dstInfo.stride),
                createI8(rewriter, loc, 1), createI8(rewriter, loc, 0), createI8(rewriter, loc, 0), 
                createBool(rewriter, loc, false), createI32(rewriter, loc, 0), 
                createI16(rewriter, loc, 1), createI16(rewriter, loc, 0)
            });
            rewriter.eraseOp(op);
            return success();
        }
        return failure();
    }
};

struct LowerSfuPattern : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
        if (!op->hasAttr("npu.target")) return failure();
        for (Value operand : op.getOperands()) {
            auto type = mlir::cast<MemRefType>(operand.getType());
            if (type.getMemorySpaceAsInt() != NPU_SRAM_SPACE) return failure();
        }

        auto libCall = op.getLibraryCallAttr();
        if (!libCall) return failure();
        StringRef opName = libCall.getValue();
        int opType = (opName == "npu_gelu") ? SFU_OP_GELU : SFU_OP_ADD;

        Location loc = op.getLoc();
        Value input = op.getInputs()[0];
        Value output = op.getOutputs()[0];
        LayoutInfo inInfo;
        if (failed(getLayoutInfo(mlir::cast<MemRefType>(input.getType()), inInfo))) return failure();

        Value vSramIn  = getSramOffset(rewriter, loc, input);
        Value vSramOut = getSramOffset(rewriter, loc, output);

        ModuleOp module = op->getParentOfType<ModuleOp>();
        SmallVector<Type> sfuTypes = {
            rewriter.getI8Type(), rewriter.getI8Type(), rewriter.getI1Type(),
            rewriter.getI32Type(), rewriter.getI16Type(), rewriter.getI16Type(),
            rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getI16Type(), 
            rewriter.getI16Type(), rewriter.getI16Type(), rewriter.getI16Type(), rewriter.getI16Type()
        };
        func::FuncOp sfuFunc = getOrInsertFunc(rewriter, module, "npu_sfu_run", sfuTypes);
        
        rewriter.create<func::CallOp>(loc, sfuFunc, ValueRange{
            createI8(rewriter, loc, (int8_t)opType), createI8(rewriter, loc, 2), createBool(rewriter, loc, false),
            vSramIn, createI16(rewriter, loc, inInfo.colNumReg), createI16(rewriter, loc, inInfo.rowNumReg), vSramOut,
            createI32(rewriter, loc, 0), createI16(rewriter, loc, 0), createI16(rewriter, loc, 1),
            createI16(rewriter, loc, 0), createI16(rewriter, loc, 1), createI16(rewriter, loc, 0)
        });

        rewriter.eraseOp(op);
        return success();
    }
};

struct EliminateHostCopyPattern : public OpRewritePattern<memref::CopyOp> {
    using OpRewritePattern<memref::CopyOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(memref::CopyOp op, PatternRewriter &rewriter) const override {
        Value src = op.getSource();
        Value dst = op.getTarget();
        auto srcType = mlir::cast<MemRefType>(src.getType());
        auto dstType = mlir::cast<MemRefType>(dst.getType());
        int srcSpace = srcType.getMemorySpaceAsInt();
        int dstSpace = dstType.getMemorySpaceAsInt();

        if (srcSpace == HOST_SPACE && dstSpace == SHARED_DRAM_SPACE) {
            auto srcAllocOp = src.getDefiningOp<memref::AllocOp>();
            auto dstAllocOp = dst.getDefiningOp<memref::AllocOp>();
            if (!srcAllocOp || !dstAllocOp) return failure();

            rewriter.setInsertionPoint(srcAllocOp);
            MemRefType newType = MemRefType::get(srcType.getShape(), srcType.getElementType(), 
                                                 srcType.getLayout(), rewriter.getI64IntegerAttr(SHARED_DRAM_SPACE));
            Value newAlloc = rewriter.create<memref::AllocOp>(srcAllocOp.getLoc(), newType);
            rewriter.replaceOp(srcAllocOp, newAlloc);
            rewriter.replaceOp(dstAllocOp, newAlloc);
            rewriter.eraseOp(op);
            return success();
        } 
        else if (srcSpace == SHARED_DRAM_SPACE && dstSpace == HOST_SPACE) {
            auto dstAllocOp = dst.getDefiningOp<memref::AllocOp>();
            if (!dstAllocOp) return failure();
            rewriter.setInsertionPoint(op);
            Value castedValue = rewriter.create<memref::MemorySpaceCastOp>(op.getLoc(), dstType, src);
            rewriter.replaceOp(dstAllocOp, castedValue);
            rewriter.eraseOp(op);
            return success();
        }
        return failure();
    }
};

struct LowerUnrealizedCastPattern : public OpRewritePattern<UnrealizedConversionCastOp> {
   using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;
   LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, PatternRewriter &rewriter) const override {
       // 处理 Space2 -> Space0 的回退 Cast，保持 IR 干净
       Value input = op.getInputs()[0];
       auto srcType = llvm::dyn_cast<MemRefType>(input.getType());
       auto dstType = llvm::dyn_cast<MemRefType>(op.getResultTypes()[0]);
       if (srcType && dstType && 
           srcType.getMemorySpaceAsInt() == SHARED_DRAM_SPACE && 
           dstType.getMemorySpaceAsInt() == HOST_SPACE) {
           rewriter.replaceOpWithNewOp<memref::MemorySpaceCastOp>(op, dstType, input);
           return success();
       }
       return failure();
   }
};

struct NpuInstructionLoweringPass : public PassWrapper<NpuInstructionLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuInstructionLoweringPass)
    StringRef getArgument() const override { return "npu-instruction-lowering"; }
    
    void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // 1. 插入 npu_init (保持不变)
    OpBuilder builder(module.getBodyRegion());
    auto initFunc = getOrInsertFunc(builder, module, "npu_init", {}, builder.getI32Type());
    if (auto mainFunc = module.lookupSymbol<func::FuncOp>("main_graph")) {
        builder.setInsertionPointToStart(&mainFunc.front());
        builder.create<func::CallOp>(mainFunc.getLoc(), initFunc, ValueRange{});
    }

    // 2. 运行你的 NPU Patterns (Alloc->Call, MVIN/MVOUT Ptr化)
    // 这一步会生成 extract_strided_metadata(subview(...))
    {
        RewritePatternSet patterns(context);
        patterns.add<
            EliminateHostCopyPattern,
            LowerDmaCopyPattern,
            LowerSfuPattern,
            LowerUnrealizedCastPattern,
            LowerNpuAllocPattern
        >(context);

        if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
            signalPassFailure();
            return; // 失败直接返回
        }
    }

    // ================================================================
    // 【关键新增步骤】 消灭 memref.subview
    // ================================================================
    // 我们创建一个临时的 PassManager 来运行 MemRef 的清理 Pass
    PassManager pm(context);
    
    // 1. 将 subview 展开为算术运算 (arith ops)
    // 这会将 extract_metadata(subview(x)) 变成 extract_metadata(x) + offset_calc
    // 从而移除 subview op 本身。
    pm.addPass(memref::createExpandStridedMetadataPass());

    // 2. 规范化 (Canonicalize)
    // 这一步非常重要，它会执行 Dead Code Elimination (DCE)。
    // 因为展开后，subview 变成了无用的代码，Canonicalize 会把它删掉。
    pm.addPass(createCanonicalizerPass());

    if (failed(pm.run(module))) {
        signalPassFailure();
    }
}
};

} // namespace

std::unique_ptr<Pass> npux::createNpuInstructionLoweringPass() {
  return std::make_unique<NpuInstructionLoweringPass>();
}
