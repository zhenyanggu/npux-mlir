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

std::pair<int64_t, int64_t> getFlattened2DShape(
    ArrayRef<int64_t> shape, Operation *op = nullptr) {
  int64_t rank = shape.size();

  // 处理低维度情况
  if (rank == 0)
    return {1, 1};
  if (rank == 1)
    return {1, shape[0]};
  if (rank == 2)
    return {shape[0], shape[1]};

  int64_t row = 1;
  int64_t col = 1;
  const int64_t COL_LIMIT = 262144; // 现在已经更改为 32bit寄存器限制，但是spm只有512k，所以限制

  // 定义分割点索引：从该索引开始（含）往后的所有维度都乘入 col
  int64_t splitIdx = rank - 1;

  // 分情况讨论逻辑
  if (rank == 3 || rank == 4) {
    // 3、4维：Col 为最里面一维
    splitIdx = rank - 1;
  } else if (rank == 5) {
    // 5维：Col 为最里面两位相乘
    splitIdx = rank - 2;
  } else if (rank >= 6) {
    // 6维及以上：Col 为最里面五位相乘, 这是处理卷积的权重
    splitIdx = rank - 5;
  }

  // 安全边界检查：防止 splitIdx 计算越界
  if (splitIdx < 0)
    splitIdx = 0;

  // 计算 Col 乘积
  for (int i = splitIdx; i < rank; ++i) {
    col *= shape[i];
  }

  // 计算 Row 乘积
  for (int i = 0; i < splitIdx; ++i) {
    row *= shape[i];
  }

  if (col > COL_LIMIT) {
    if (op) {
      // 如果有 Op 上下文，直接在 IR 位置报 warning
      op->emitWarning() << "Hardware Constraint: col value (" << col
                        << ") exceeds 16-bit register limit (" << COL_LIMIT
                        << ") at Rank " << rank;
    } else {
      // 否则使用 llvm::errs 打印到控制台
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "[NPU Warning] Col value (" << col
         << ") exceeds 16-bit register limit (" << COL_LIMIT << ") for shape [";
      for (size_t i = 0; i < shape.size(); ++i) {
        os << shape[i] << (i == shape.size() - 1 ? "" : ", ");
      }
      os << "]\n";
      llvm::errs() << os.str();
    }
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
    auto [rows, cols] = getFlattened2DShape(shape,op);

    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols - 1, 32);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows - 1, 32);

    // 2. 获取 DRAM Strides
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    if (failed(dramType.getStridesAndOffset(strides, offset))) {
      return failure();
    }

    int64_t rank = dramType.getRank();
    int64_t splitIdx = rank - 1;

    // 与 getFlattened2DShape 的分块规则保持一致
    if (rank == 5) {
      splitIdx = rank - 2;
    } else if (rank >= 6) {
      splitIdx = rank - 5;
    }

    if (splitIdx < 0)
      splitIdx = 0;

    int64_t dramStrideVal = cols;
    // 对于有 row 维度的情况，优先使用真实 memref stride，兼容非连续 layout。
    // 对于 splitIdx == 0（row=1）则退回 cols。
    if (splitIdx > 0 && (splitIdx - 1) < (int64_t)strides.size()) {
      dramStrideVal = strides[splitIdx - 1];
    }

    Value vDramStride =
      rewriter.create<arith::ConstantIntOp>(loc, dramStrideVal, 32);

    // 3. 获取 SRAM Strides
    // 通常 SRAM 是连续的，stride 等于 cols。但如果 SRAM 也有 layout，应从
    // sramType 获取 这里暂时保持和 cols 一致，或者通过 sramType 计算
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