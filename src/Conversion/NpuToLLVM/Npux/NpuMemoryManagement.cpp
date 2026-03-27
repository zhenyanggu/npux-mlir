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
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <tuple>

using namespace mlir;
using namespace npux;

namespace {
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
    if (!libCallAttr)
      return failure();

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
    bool isMvin = (srcSpace == 1 && (dstSpace == 2 || dstSpace == 3));
    bool isMvout = (srcSpace == 2 && dstSpace == 1);

    if (!isMvin && !isMvout)
      return failure();

    // --- 核心修复：根据传输方向确定 DRAM 和 SRAM 的类型 ---
    MemRefType dramType = isMvin ? srcType : dstType;
    MemRefType sramType = isMvin ? dstType : srcType;
    Location loc = op.getLoc();

    auto shape = dramType.getShape();
    int64_t rank = shape.size();
    if (llvm::any_of(shape, [](int64_t d) { return d < 0; })) {
      op->emitError()
          << "Dynamic memref shape is unsupported for DMA parameter lowering: "
          << dramType;
      return failure();
    }

    // ========================================================================
    // 1. 直接从上游 Pass 读取 col_dim_idx 属性，告别启发式猜测
    // ========================================================================
    int64_t splitIdx = 0; // 默认 0 表示完全连续的 1D DMA
    if (auto colDimAttr = op->getAttrOfType<IntegerAttr>("npu.dma_col_dim")) {
      splitIdx = colDimAttr.getInt();
    }
    
    // 防御性保护，防止属性异常
    if (splitIdx < 0 || splitIdx > rank) {
      splitIdx = 0;
    }

    int64_t rows = 1;
    int64_t cols = 1;

    // splitIdx 左边的维度累乘为 Row
    for (int i = 0; i < splitIdx; ++i) {
      rows *= shape[i];
    }
    // splitIdx 及右边的维度累乘为 Col
    for (int i = splitIdx; i < rank; ++i) {
      cols *= shape[i];
    }

    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols - 1, 32);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows - 1, 32);

    // ========================================================================
    // 2. 获取 DRAM Strides (复用你原本借助 MLIR 布局的稳健推导)
    // ========================================================================
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    if (failed(dramType.getStridesAndOffset(strides, offset))) {
      return failure();
    }

    int64_t dramStrideVal = cols;
    // 使用 splitIdx - 1 精准获取跨越 Row 的步长
    if (splitIdx > 0 && (splitIdx - 1) < (int64_t)strides.size()) {
      dramStrideVal = strides[splitIdx - 1];
    }

    Value vDramStride =
        rewriter.create<arith::ConstantIntOp>(loc, dramStrideVal, 32);

    // ========================================================================
    // 3. 获取 SRAM Stride
    // ========================================================================
    int64_t sramOffset;
    SmallVector<int64_t, 4> sramStrides;
    if (failed(sramType.getStridesAndOffset(sramStrides, sramOffset))) {
      return failure();
    }
    
    int64_t sramStrideVal = sramStrides[splitIdx - 1];
    
    
    Value vSramStride =
        rewriter.create<arith::ConstantIntOp>(loc, sramStrideVal, 16);

    // ========================================================================
    // 4. Precision & Hardware Logic (完全保持原样)
    // ========================================================================
    Value vPrecision = rewriter.create<arith::ConstantIntOp>(loc, 1, 8);
    Value vInputType = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);

    auto getIntAttrOr = [&](StringRef name, int64_t defaultVal) -> int64_t {
      if (auto attr = op->getAttrOfType<IntegerAttr>(name))
        return attr.getInt();
      return defaultVal;
    };

    bool mvinIsQuant = false;
    if (auto attr = op->getAttrOfType<BoolAttr>("npu.is_quant")) {
      mvinIsQuant = attr.getValue();
    }
    int64_t mvinQuantZero = getIntAttrOr("npu.quant_zero", 0);
    int64_t mvinQuantScale = getIntAttrOr("npu.quant_scale", 0);
    int64_t mvinQuantShift = getIntAttrOr("npu.quant_shift", 0);

    Value vZero32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value vZero16 = rewriter.create<arith::ConstantIntOp>(loc, 0, 16);
    Value vZero8 = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);
    Value vFalse = rewriter.create<arith::ConstantIntOp>(loc, 0, 1);
    Value vMvinIsQuant =
        rewriter.create<arith::ConstantIntOp>(loc, mvinIsQuant ? 1 : 0, 1);
    Value vMvinQuantZero =
        rewriter.create<arith::ConstantIntOp>(loc, mvinQuantZero, 32);
    Value vMvinQuantScale =
        rewriter.create<arith::ConstantIntOp>(loc, mvinQuantScale, 16);
    Value vMvinQuantShift =
        rewriter.create<arith::ConstantIntOp>(loc, mvinQuantShift, 16);

    if (isMvin) {
      int64_t destFlag = (dstSpace == 3) ? 1 : 0;
      Value vDest = rewriter.create<arith::ConstantIntOp>(loc, destFlag, 8);
      Value vIsBias = vFalse;

      rewriter.create<DmaMvinOp>(loc, src, dst, vCol, vRow, vSramStride,
          vDramStride, vPrecision, vInputType, vDest, vIsBias, vMvinIsQuant,
          vMvinQuantZero, vMvinQuantScale, vMvinQuantShift);
    } else {
      rewriter.create<DmaMvoutOp>(loc, dst, src, vCol, vRow, vSramStride,
          vDramStride, vPrecision, vInputType,
          vZero8, vFalse, vZero32, vZero16, vZero16);
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

    int space = op.getType().getMemorySpaceAsInt();

    if (space == 1) {
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

    Value memref = op.getMemref();
    auto type = cast<MemRefType>(memref.getType());
    int space = type.getMemorySpaceAsInt();
    if (space == 1) {
      rewriter.create<npux::FreeOp>(op.getLoc(), memref);
      rewriter.eraseOp(op);
      return success();
    }
    if (space == 2) {
      rewriter.create<npux::SramFreeOp>(op.getLoc(), memref);
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
