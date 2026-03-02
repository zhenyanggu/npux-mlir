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

std::pair<int64_t, int64_t> getFlattened2DShape(ArrayRef<int64_t> shape) {
  int64_t rank = shape.size();

  if (rank == 0)
    return {1, 1};
  if (rank == 1)
    return {1, shape[0]};

  // [NEW] Rank 2 处理：保留矩阵结构
  if (rank == 2) {
    return {shape[0], shape[1]}; // Row, Col
  }

  // Rank >= 3 处理：低两维合并
  int64_t col = shape[rank - 1] * shape[rank - 2];
  int64_t row = 1;
  for (int i = 0; i < rank - 2; ++i) {
    row *= shape[i];
  }

  return {row, col};
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
    // 1. Check Library Call Name
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr || libCallAttr.getValue() != "mv_acc_to_spm") {
      return failure();
    }

    // 2. Get Operands
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    Value src = op.getInputs()[0];
    Value dst = op.getOutputs()[0];

    auto srcType = cast<MemRefType>(src.getType());
    auto dstType = cast<MemRefType>(dst.getType());

    if (srcType.getMemorySpaceAsInt() != 3 ||
        dstType.getMemorySpaceAsInt() != 2)
      return failure();

    Location loc = op.getLoc();

    // 3. Create Npux Op (No shape args needed now)
    rewriter.create<npux::MvAccToSpmOp>(loc, src, dst);

    // 4. Erase original
    rewriter.eraseOp(op);
    return success();
  }
};


class ConvertMemrefCopyToNpuxPattern
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr) return failure();
    
    StringRef libName = libCallAttr.getValue();


    if (libName != "npu_dma_mvin" && libName != "npu_dma_mvout") {
      return failure();
    }

    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1) {
      return failure();
    }

    Value src = op.getInputs()[0];
    Value dst = op.getOutputs()[0];

    auto srcType = cast<MemRefType>(src.getType());
    auto dstType = cast<MemRefType>(dst.getType());

    int srcSpace = srcType.getMemorySpaceAsInt();
    int dstSpace = dstType.getMemorySpaceAsInt();
    bool isMvin = (srcSpace == 0 && (dstSpace == 2 || dstSpace == 3));
    bool isMvout = (srcSpace == 2 && dstSpace == 0);
    
    if (!isMvin && !isMvout)
      return failure();

    // --- 核心修复：根据传输方向确定 DRAM 和 SRAM 的类型 ---
    MemRefType dramType = isMvin ? srcType : dstType;
    MemRefType sramType = isMvin ? dstType : srcType;
    Location loc = op.getLoc();

    // 1. 获取物理形状 (通常从逻辑形状一致的 dramType 获取)
    auto shape = dramType.getShape();
    auto [rows, cols] = getFlattened2DShape(shape);

    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols-1, 16);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows-1, 16);

    // 2. 获取 DRAM Strides
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    if (failed(dramType.getStridesAndOffset(strides, offset))) {
      return failure();
    }

    int64_t rank = dramType.getRank();
    int64_t dramStrideVal = 0;

    if (rank >= 3) {
      dramStrideVal = strides[rank - 3];
    } else if (rank == 2) {
      dramStrideVal = strides[0];
    } else {
      dramStrideVal = 0;
    }

    Value vDramStride = rewriter.create<arith::ConstantIntOp>(loc, dramStrideVal, 16);

    // 3. 获取 SRAM Strides
    // 通常 SRAM 是连续的，stride 等于 cols。但如果 SRAM 也有 layout，应从 sramType 获取
    // 这里暂时保持和 cols 一致，或者通过 sramType 计算
    Value vSramStride = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);

    // 4. Precision Logic (从数据源获取类型)
    // 无论是 mvin 还是 mvout，元素类型通常是一致的
    Type elemType = dramType.getElementType();
    int64_t precisionVal = elemType.isInteger(32) ? 1 : 0;

    Value vPrecision =
        rewriter.create<arith::ConstantIntOp>(loc, precisionVal, 8);
    Value vInputType =
        rewriter.create<arith::ConstantIntOp>(loc, 0, 8); // Default

    // Common Constants
    Value vZero32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value vZero16 = rewriter.create<arith::ConstantIntOp>(loc, 0, 16);
    Value vZero8 = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);
    Value vFalse = rewriter.create<arith::ConstantIntOp>(loc, 0, 1);

    if (isMvin) {
      // === DMA MVIN ===
      // Dest: 0 for SPM (Space 2), 1 for ACC (Space 3)
      int64_t destFlag = (dstSpace == 3) ? 1 : 0;
      Value vDest = rewriter.create<arith::ConstantIntOp>(loc, destFlag, 8);

      // Is Bias: Always 0 (Hardcoded as requested)
      Value vIsBias = vFalse;

      // Quant: Disabled
      Value vIsQuant = vFalse;

      rewriter.create<DmaMvinOp>(loc, src, dst, vCol, vRow, vSramStride,
          vDramStride, vPrecision, vInputType, vDest, vIsBias, vIsQuant,
          vZero32, vZero16, vZero16);
    } else {
      // === DMA MVOUT ===
      // Dest: usually 0 for DRAM
      rewriter.create<DmaMvoutOp>(loc, dst, src, vCol, vRow, vSramStride,
          vDramStride, vPrecision, vInputType,
          vZero8, // dest (ignored)
          vFalse, // is_bias (ignored)
          vZero32, vZero16, vZero16);
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