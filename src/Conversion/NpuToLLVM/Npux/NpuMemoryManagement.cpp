//=================================================
// src/Conversion/NpuToLLVM/NpuMemoryManagement.cpp
// This file implements NPU memory management and
// data movement insertion patterns.
//=================================================

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Conversion/NpuToLLVM/NpuxConversionHelper.hpp"
#include "src/Dialect/Npux/NpuxOps.hpp"

using namespace mlir;
using namespace npux;

namespace {

// Helper: Check if the operation is inside a function marked with
// npu.target="npu"
bool isInNpuKernel(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return false;
  if (auto attr = funcOp->getAttrOfType<StringAttr>("npu.target")) {
    return attr.getValue() == "npu";
  }
  return false;
}

// =========================================================
// Pattern 1: Convert ACC -> SPM Quantization Generic to NPU Move
// Matches: linalg.generic { npu.pp_stage = "quant_acc2spm" }
// =========================================================
class ConvertAccToSpmPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {

    // 1. 检查 Label 是否匹配
    auto stageAttr = op->getAttrOfType<StringAttr>("npu.pp_stage");
    if (!stageAttr || stageAttr.getValue() != "quant_acc2spm") {
      return failure();
    }

    // 2. 获取操作数
    // Generic: ins(ACC_i32), outs(SPM_i8)
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
      return failure();

    Value src = op.getInputs()[0];
    Value dst = op.getOutputs()[0]; // Generic 的输出在 outputs/inits 列表里

    auto srcType = cast<MemRefType>(src.getType());
    // 简单校验空间 (ACC=3, SPM=2)
    if (srcType.getMemorySpaceAsInt() != 3)
      return failure();

    Location loc = op.getLoc();

    // 3. 计算维度 (逻辑同原 Copy Pattern)
    auto shape = srcType.getShape();
    int64_t rank = shape.size();

    int64_t rows = (rank >= 2) ? shape[rank - 2] : 1;
    int64_t cols = (rank >= 1) ? shape[rank - 1] : 1;

    // Handle NCHWc32 or other packed layouts if necessary
    if (rank == 5) { // e.g. [N, C_blk, H, W, C_packed]
      rows = shape[2];
      cols = shape[3] * shape[4];
    }

    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows, 16);
    Value vStride = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);

    // 4. 生成 NPU 指令 (MvAccToSpm)
    // 硬件指令通常自带量化/截断功能
    rewriter.create<npux::MvAccToSpmOp>(
        loc, src, dst, vCol, vRow, vStride, vStride);

    // 5. 删除原 Generic
    rewriter.eraseOp(op);
    return success();
  }
};

// =========================================================
// Pattern 2: Convert memref.copy to NPU DMA Ops
// Handles:
// 1. DRAM (Space 0) -> SPM (Space 2) => npux.dma_mvin (dest=0)
// 2. DRAM (Space 0) -> ACC (Space 3) => npux.dma_mvin (dest=1)
// 3. SPM (Space 2) -> DRAM (Space 0) => npux.dma_mvout
// Note: ACC->SPM is now handled by ConvertAccToSpmPattern above.
// =========================================================
class ConvertMemrefCopyToNpuxPattern : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      memref::CopyOp op, PatternRewriter &rewriter) const override {

    Value src = op.getSource();
    Value dst = op.getTarget();

    auto srcType = cast<MemRefType>(src.getType());
    auto dstType = cast<MemRefType>(dst.getType());

    int srcSpace = srcType.getMemorySpaceAsInt();
    int dstSpace = dstType.getMemorySpaceAsInt();

    Location loc = op.getLoc();

    // ---------------------------------------------------------
    // Case 2: DMA Operations (Mvin / Mvout)
    // ---------------------------------------------------------
    bool isMvin = (srcSpace == 0 && (dstSpace == 2 || dstSpace == 3));
    bool isMvout = (srcSpace == 2 && dstSpace == 0);

    if (!isMvin && !isMvout) {
      return failure();
    }

    // 1. Calculate Physical Dimensions
    auto shape = srcType.getShape();
    int64_t rank = shape.size();

    int64_t physicalRows = 1;
    int64_t physicalCols = 1;

    if (rank == 5) {
      physicalRows = shape[2] - 1;
      physicalCols = shape[3] * shape[4] - 1;
    } else if (rank == 6) {
      physicalRows = shape[0] - 1;
      physicalCols = shape[1] * shape[2] * shape[3] * shape[4] * shape[5] - 1;
    } else if (rank >= 2) {
      physicalRows = shape[rank - 2] - 1;
      physicalCols = shape[rank - 1] - 1;
    } else if (rank == 1) {
      physicalRows = 1;
      physicalCols = shape[0];
    } else {
      return failure(); 
    }

    // 2. Hardware Registers
    int64_t dmaRegCol = std::max<int64_t>(0, physicalCols);
    int64_t dmaRegRow = std::max<int64_t>(0, physicalRows);

    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, dmaRegCol, 16);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, dmaRegRow, 16);

    // 3. Strides
    int64_t strideVal = physicalCols;
    Value vSramStride = rewriter.create<arith::ConstantIntOp>(loc, strideVal, 16);
    Value vDramStride = rewriter.create<arith::ConstantIntOp>(loc, strideVal, 16);

    // 4. Precision Logic (Modified Here)
    // ---------------------------------------------------------
    Type elemType = srcType.getElementType();
    int64_t precisionVal = 0;

    if (elemType.isInteger(8)) {
      precisionVal = 0;
    } else if (elemType.isInteger(32)) {
      precisionVal = 1;
    } else {
      // 如果不是 i8 也不是 i32，报 warning 并默认设为 0（或者根据需求 return failure）
      op->emitWarning("Unsupported precision for DMA copy, defaulting to i8 (0).");
      precisionVal = 0;
    }
    // ---------------------------------------------------------

    Value vZero8 = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);
    Value vZero16 = rewriter.create<arith::ConstantIntOp>(loc, 0, 16);
    Value vZero32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value vFalse = rewriter.create<arith::ConstantIntOp>(loc, 0, 1);
    
    Value vPrecision = rewriter.create<arith::ConstantIntOp>(loc, precisionVal, 8);
    Value vInputType;

    // 保留你原本对 rank 6 的 vInputType 逻辑
    if (rank == 6) {
      vInputType = rewriter.create<arith::ConstantIntOp>(loc, 1, 8);
    } else {
      vInputType = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);
    }

    if (isMvin) {
      int64_t destFlag = (dstSpace == 3) ? 1 : 0;
      Value vDest = rewriter.create<arith::ConstantIntOp>(loc, destFlag, 8);
      Value vIsBias = rewriter.create<arith::ConstantIntOp>(loc, 0, 1);

      rewriter.create<DmaMvinOp>(loc, src, dst, vCol, vRow, vSramStride,
          vDramStride, vPrecision, vInputType, vDest, vIsBias, vFalse, vZero32,
          vZero16, vZero16);
    } else {
      rewriter.create<DmaMvoutOp>(loc, dst, src, vCol, vRow, vSramStride,
          vDramStride, vPrecision, vInputType, vZero8, vFalse, vZero32, vZero16,
          vZero16);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// =========================================================
// Pattern 3 & 4: Alloc/Dealloc (保持不变)
// =========================================================
class ConvertHostAllocToNpuxPattern : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      memref::AllocOp op, PatternRewriter &rewriter) const override {

    if (!isInNpuKernel(op))
      return failure();
    int space = op.getType().getMemorySpaceAsInt();

    if (space == 0) {
      rewriter.replaceOpWithNewOp<npux::AllocOp>(
          op, op.getType(), op.getDynamicSizes());
      return success();
    }
    if (space == 2) {
      rewriter.replaceOpWithNewOp<npux::SramAllocOp>(
          op, op.getType(), op.getDynamicSizes());
      return success();
    }
    if (space == 3) {
      rewriter.replaceOpWithNewOp<npux::AccAllocOp>(
          op, op.getType(), op.getDynamicSizes());
      return success();
    }
    return success();
  }
};

class ConvertDeallocToNpuxPattern : public OpRewritePattern<memref::DeallocOp> {
public:
  using OpRewritePattern<memref::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      memref::DeallocOp op, PatternRewriter &rewriter) const override {

    if (!isInNpuKernel(op))
      return failure();
    Value memref = op.getMemref();
    auto type = cast<MemRefType>(memref.getType());
    int space = type.getMemorySpaceAsInt();

    if (space == 2) {
      rewriter.create<npux::SramFreeOp>(op.getLoc(), memref);
      rewriter.eraseOp(op);
      return success();
    }
    if (space == 0) {
      rewriter.create<npux::FreeOp>(op.getLoc(), memref);
      rewriter.eraseOp(op);
      return success();
    }
    if (space == 3) {
      rewriter.create<npux::AccFreeOp>(op.getLoc(), memref);
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

} // namespace

// =========================================================
// Registration
// =========================================================

void npux::populateSramDataMovementPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertMemrefCopyToNpuxPattern>(patterns.getContext());
  patterns.add<ConvertAccToSpmPattern>(patterns.getContext());
}

void npux::populateHostAllocToNpuxPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertHostAllocToNpuxPattern>(patterns.getContext());
  patterns.add<ConvertDeallocToNpuxPattern>(patterns.getContext());
}