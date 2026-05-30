//=================================================
// src/Conversion/NpuToLLVM/NpuMemoryManagement.cpp
// This file implements NPU memory management and
// data movement insertion patterns.
//=================================================

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Conversion/NpuToLLVM/NpuxConversionHelper.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include "src/Dialect/Npux/NpuxOps.hpp"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <tuple>

using namespace mlir;
using namespace npux;

namespace {
// =========================================================
// Helper: lower a memref-form DMA op to the final npux DMA op.
// =========================================================
static LogicalResult lowerDmaToNpux(Operation *op, PatternRewriter &rewriter,
    Value src, Value dst, bool isMvin) {
  auto srcType = dyn_cast<MemRefType>(src.getType());
  auto dstType = dyn_cast<MemRefType>(dst.getType());
  if (!srcType || !dstType)
    return failure();

  int srcSpace = srcType.getMemorySpaceAsInt();
  int dstSpace = dstType.getMemorySpaceAsInt();
  if (isMvin) {
    if (!(srcSpace == 1 && (dstSpace == 2 || dstSpace == 3)))
      return failure();
  } else {
    if (!(srcSpace == 2 && dstSpace == 1))
      return failure();
  }

  MemRefType dramType = isMvin ? srcType : dstType;
  MemRefType sramType = isMvin ? dstType : srcType;
  Location loc = op->getLoc();

  auto shape = dramType.getShape();
  int64_t rank = shape.size();
  if (llvm::any_of(shape, [](int64_t d) { return d < 0; })) {
    op->emitError()
        << "Dynamic memref shape is unsupported for DMA parameter lowering: "
        << dramType;
    return failure();
  }

  int64_t splitIdx = 0;
  if (auto colDimAttr = op->getAttrOfType<IntegerAttr>("npu.dma_col_dim")) {
    splitIdx = colDimAttr.getInt();
  } else if (auto colDimAttr = op->getAttrOfType<IntegerAttr>("dma_col_dim")) {
    splitIdx = colDimAttr.getInt();
  }
  if (splitIdx < 0 || splitIdx > rank)
    splitIdx = 0;

  int64_t rows = 1;
  int64_t cols = 1;
  for (int i = 0; i < splitIdx; ++i)
    rows *= shape[i];
  for (int i = splitIdx; i < rank; ++i)
    cols *= shape[i];

  Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols - 1, 32);
  Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows - 1, 32);

  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(dramType.getStridesAndOffset(strides, offset)))
    return failure();

  int64_t dramStrideVal = cols;
  if (splitIdx > 0 && (splitIdx - 1) < (int64_t)strides.size())
    dramStrideVal = strides[splitIdx - 1];
  Value vDramStride =
      rewriter.create<arith::ConstantIntOp>(loc, dramStrideVal, 32);

  int64_t sramOffset;
  SmallVector<int64_t, 4> sramStrides;
  if (failed(sramType.getStridesAndOffset(sramStrides, sramOffset)))
    return failure();

  int64_t sramStrideVal = cols;
  if (splitIdx > 0 && (splitIdx - 1) < (int64_t)sramStrides.size())
    sramStrideVal = sramStrides[splitIdx - 1];
  Value vSramStride =
      rewriter.create<arith::ConstantIntOp>(loc, sramStrideVal, 16);

  Value vPrecision = rewriter.create<arith::ConstantIntOp>(loc, 1, 8);
  Value vInputType = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);

  auto getIntAttrOr = [&](StringRef name, int64_t defaultVal) -> int64_t {
    if (auto attr = op->getAttrOfType<IntegerAttr>(name))
      return attr.getInt();
    return defaultVal;
  };
  auto getBoolAttrOr = [&](StringRef name, bool defaultVal) -> bool {
    if (auto attr = op->getAttrOfType<BoolAttr>(name))
      return attr.getValue();
    return defaultVal;
  };

  bool mvinIsQuant =
      getBoolAttrOr("npu.is_quant", getBoolAttrOr("is_quant", false));
  int64_t mvinQuantZero =
      getIntAttrOr("npu.quant_zero", getIntAttrOr("quant_zero", 0));
  int64_t mvinQuantScale =
      getIntAttrOr("npu.quant_scale", getIntAttrOr("quant_scale", 0));
  int64_t mvinQuantShift =
      getIntAttrOr("npu.quant_shift", getIntAttrOr("quant_shift", 0));

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
        vDramStride, vPrecision, vInputType, vZero8, vFalse, vZero32, vZero16,
        vZero16);
  }

  rewriter.eraseOp(op);
  return success();
}

class ConvertNpucoreAccToSpmPattern
    : public OpRewritePattern<npucore::MvAccToSpmOp> {
public:
  using OpRewritePattern<npucore::MvAccToSpmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::MvAccToSpmOp op, PatternRewriter &rewriter) const override {
    if (op.hasTensorSemantics())
      return failure();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    Value src = op.getInputs()[0];
    Value dst = op.getOutputs()[0];

    auto srcType = dyn_cast<MemRefType>(src.getType());
    auto dstType = dyn_cast<MemRefType>(dst.getType());
    if (!srcType || !dstType)
      return failure();
    if (srcType.getMemorySpaceAsInt() != 3 || dstType.getMemorySpaceAsInt() != 2)
      return failure();

    rewriter.replaceOpWithNewOp<npux::MvAccToSpmOp>(op, src, dst);
    return success();
  }
};

class ConvertNpucoreDmaMvinPattern
    : public OpRewritePattern<npucore::DmaMvinOp> {
public:
  using OpRewritePattern<npucore::DmaMvinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::DmaMvinOp op, PatternRewriter &rewriter) const override {
    if (op.hasTensorSemantics())
      return failure();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();
    return lowerDmaToNpux(
        op, rewriter, op.getInputs().front(), op.getOutputs().front(), true);
  }
};

class ConvertNpucoreDmaMvoutPattern
    : public OpRewritePattern<npucore::DmaMvoutOp> {
public:
  using OpRewritePattern<npucore::DmaMvoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::DmaMvoutOp op, PatternRewriter &rewriter) const override {
    if (op.hasTensorSemantics())
      return failure();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();
    return lowerDmaToNpux(
        op, rewriter, op.getInputs().front(), op.getOutputs().front(), false);
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
  patterns.add<ConvertNpucoreAccToSpmPattern>(patterns.getContext());
  patterns.add<ConvertNpucoreDmaMvinPattern>(patterns.getContext());
  patterns.add<ConvertNpucoreDmaMvoutPattern>(patterns.getContext());
}

void npux::populateHostAllocToNpuxPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertHostAllocToNpuxPattern>(patterns.getContext());
  patterns.add<ConvertDeallocToNpuxPattern>(patterns.getContext());
}
