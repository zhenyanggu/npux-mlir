//==================================================================
// src/Conversion/NpuToLLVM/NpuInstructionLowering.cpp
// this file is for lowering high level NPU ops to the calls of NPU API
//==================================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

// 【关键定义】引入第三种空间
const int HOST_SPACE = 0;        // 普通 CPU 内存
const int NPU_SRAM_SPACE = 1;    // NPU 内部 SRAM (Space 1)
const int SHARED_DRAM_SPACE = 2; // CPU/NPU 共享 DRAM (Space 2)

// SRAM 地址硬编码 (Ping-Pong 策略)
const uint32_t SRAM_BANK_0 = 0x0000;
const uint32_t SRAM_BANK_1 = 0x4000;

const int SFU_OP_GELU = 1;
const int SFU_OP_ADD  = 3;

// ... (辅助函数 createI32 等保持不变，略) ...
Value createI32(PatternRewriter &rewriter, Location loc, int32_t val) {
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(val));
}
Value createI16(PatternRewriter &rewriter, Location loc, int16_t val) {
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getI16Type(), rewriter.getI16IntegerAttr(val));
}
Value createI8(PatternRewriter &rewriter, Location loc, int8_t val) {
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(val));
}
Value createBool(PatternRewriter &rewriter, Location loc, bool val) {
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getBoolAttr(val));
}

// ... (getOrInsertFunc 保持不变，记得 setPrivate) ...
static func::FuncOp getOrInsertFunc(PatternRewriter &rewriter, ModuleOp module, 
                                    StringRef funcName, TypeRange operandTypes) {
    if (auto func = module.lookupSymbol<func::FuncOp>(funcName))
        return func;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto funcOp = rewriter.create<func::FuncOp>(module.getLoc(), funcName, 
        FunctionType::get(rewriter.getContext(), operandTypes, {}));
    funcOp.setPrivate();
    return funcOp;
}

struct EliminateHostCopyPattern : public OpRewritePattern<memref::CopyOp> {
    using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(memref::CopyOp op, PatternRewriter &rewriter) const override {
        Value src = op.getSource();
        Value dst = op.getTarget();
        
        auto srcType = mlir::cast<MemRefType>(src.getType());
        auto dstType = mlir::cast<MemRefType>(dst.getType());

        // 【修改点】只匹配 Space 0 -> Space 2
        if (srcType.getMemorySpaceAsInt() != HOST_SPACE || 
            dstType.getMemorySpaceAsInt() != SHARED_DRAM_SPACE)
            return failure();

        // 追溯 Source 是否是 Alloc
        auto srcAllocOp = src.getDefiningOp<memref::AllocOp>();
        if (!srcAllocOp) return failure();

        // 追溯 Target 是否是 Alloc (由 Placement 插入的)
        auto dstAllocOp = dst.getDefiningOp<memref::AllocOp>();
        if (!dstAllocOp) return failure();

        // -------------------------------------------------------------
        // 改造：提升 Source Alloc 到 Space 2
        // -------------------------------------------------------------
        rewriter.setInsertionPoint(srcAllocOp);
        
        // 创建新的 Space 2 Alloc
        MemRefType newType = MemRefType::get(srcType.getShape(), srcType.getElementType(), 
                                             srcType.getLayout(), 
                                             rewriter.getI64IntegerAttr(SHARED_DRAM_SPACE));
        Value newAlloc = rewriter.create<memref::AllocOp>(srcAllocOp.getLoc(), newType);

        // 1. 让原本使用 Space 0 的操作（比如 CPU 填充数据的代码）现在直接写到 Space 2
        rewriter.replaceOp(srcAllocOp, newAlloc);

        // 2. 让原本使用 Space 2 的操作（比如 Kernel Call）直接用这个新内存
        rewriter.replaceOp(dstAllocOp, newAlloc);

        // 3. 既然源头和尽头是同一块内存，Copy 就不需要了
        rewriter.eraseOp(op); 

        return success();
    }
};

// ==============================================================================
// Pattern 2: Lowering Linalg.Generic -> MVIN + SFU + MVOUT
// ==============================================================================
struct LowerGeluToCommand : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
        if (!op->hasAttr("npu.target")) return failure();
        auto libCall = op.getLibraryCallAttr();
        if (!libCall) return failure();
        StringRef opName = libCall.getValue();

        int opType = 0;
        if (opName == "npu_gelu") opType = SFU_OP_GELU;
        else if (opName == "npu_add") opType = SFU_OP_ADD;
        else return failure();

        Location loc = op.getLoc();
        Value input = op.getInputs()[0];
        Value output = op.getOutputs()[0];
        auto inType = mlir::cast<MemRefType>(input.getType());
        
        ArrayRef<int64_t> shape = inType.getShape();
        int64_t rowNum = shape[shape.size() - 2]; 
        int64_t colNum = shape[shape.size() - 1]; 
        
        int64_t dramStride = 224; 
        int64_t sramStride = colNum;

        Value vSramIn  = createI32(rewriter, loc, SRAM_BANK_0);
        Value vSramOut = createI32(rewriter, loc, SRAM_BANK_1);
        Value vCol     = createI16(rewriter, loc, (int16_t)colNum);
        Value vRow     = createI16(rewriter, loc, (int16_t)rowNum);
        Value vSStride = createI16(rewriter, loc, (int16_t)sramStride);
        Value vDStride = createI16(rewriter, loc, (int16_t)dramStride);
        Value vPrec    = createI8(rewriter, loc, 2);
        Value vType    = createI8(rewriter, loc, 0);
        Value vDest    = createI8(rewriter, loc, 0);
        Value vFalse   = createBool(rewriter, loc, false);
        Value vZero32  = createI32(rewriter, loc, 0);
        Value vZero16  = createI16(rewriter, loc, 0);

        ModuleOp module = op->getParentOfType<ModuleOp>();

        // ---------------------------------------------------------
        // Step A: MVIN
        // ---------------------------------------------------------
        // MVIN 的输入现在可能是 Space 2 (Shared DRAM)，也可能是 Space 0 (Host Ptr)。
        // 只要是 MemRef 就行。
        SmallVector<Type> mvinTypes = {
            input.getType(), // Host/Shared MemRef
            rewriter.getI32Type(), rewriter.getI16Type(), rewriter.getI16Type(),
            rewriter.getI16Type(), rewriter.getI16Type(), rewriter.getI8Type(),
            rewriter.getI8Type(), rewriter.getI8Type(), rewriter.getI1Type(),
            rewriter.getI32Type(), rewriter.getI16Type(), rewriter.getI16Type()
        };
        
        func::FuncOp mvinFunc = getOrInsertFunc(rewriter, module, "npu_dma_mvin", mvinTypes);
        rewriter.create<func::CallOp>(loc, mvinFunc, ValueRange{
            input, vSramIn, vCol, vRow, vSStride, vDStride, vPrec, vType, vDest,
            vFalse, vZero32, vZero16, vZero16
        });

        // ---------------------------------------------------------
        // Step B: SFU
        // ---------------------------------------------------------
        SmallVector<Type> sfuTypes = {
            rewriter.getI8Type(), rewriter.getI8Type(), rewriter.getI1Type(),
            rewriter.getI32Type(), rewriter.getI16Type(), rewriter.getI16Type(),
            rewriter.getI32Type(),
            rewriter.getI32Type(), rewriter.getI16Type(), rewriter.getI16Type(),
            rewriter.getI16Type(), rewriter.getI16Type(), rewriter.getI16Type()
        };
        func::FuncOp sfuFunc = getOrInsertFunc(rewriter, module, "npu_sfu_run", sfuTypes);
        
        Value vOpType = createI8(rewriter, loc, (int8_t)opType);
        Value vColMinus1 = createI16(rewriter, loc, (int16_t)(colNum - 1));
        Value vRowMinus1 = createI16(rewriter, loc, (int16_t)(rowNum - 1));

        rewriter.create<func::CallOp>(loc, sfuFunc, ValueRange{
            vOpType, vPrec, vFalse,
            vSramIn, vColMinus1, vRowMinus1,
            vSramOut,
            vZero32, vZero16, vZero16, vZero16, vZero16, vZero16
        });

        // ---------------------------------------------------------
        // Step C: MVOUT
        // ---------------------------------------------------------
        // 输出通常也是 Space 2 或 Space 0
        SmallVector<Type> mvoutTypes = {
            output.getType(), 
            rewriter.getI32Type(), rewriter.getI16Type(), rewriter.getI16Type(),
            rewriter.getI16Type(), rewriter.getI16Type(), rewriter.getI8Type(),
            rewriter.getI8Type(), rewriter.getI8Type(), rewriter.getI1Type(),
            rewriter.getI32Type(), rewriter.getI16Type(), rewriter.getI16Type()
        };
        func::FuncOp mvoutFunc = getOrInsertFunc(rewriter, module, "npu_dma_mvout", mvoutTypes); 

        rewriter.create<func::CallOp>(loc, mvoutFunc, ValueRange{
            output, vSramOut, vCol, vRow, vSStride, vDStride, vPrec, vType, vDest,
            vFalse, vZero32, vZero16, vZero16
        });

        rewriter.eraseOp(op);
        return success();
    }
};

struct NpuInstructionLoweringPass : public PassWrapper<NpuInstructionLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuInstructionLoweringPass)
    StringRef getArgument() const override { return "npu-instruction-lowering"; }
    
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        // 【注意】使用新的 Pattern
        patterns.add<EliminateHostCopyPattern, LowerGeluToCommand>(&getContext());
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuInstructionLoweringPass() {
    return std::make_unique<NpuInstructionLoweringPass>();
}

static PassRegistration<NpuInstructionLoweringPass> pass;