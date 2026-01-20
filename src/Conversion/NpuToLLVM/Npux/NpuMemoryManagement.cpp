//=================================================
// src/Conversion/NpuToLLVM/NpuMemoryManagement.cpp
// this file implements npu sram data movement insertion pass
//=================================================
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Conversion/NpuToLLVM/NpuxConversionHelper.hpp"
#include "src/Dialect/Npux/NpuxOps.hpp"

using namespace mlir;
using namespace npux;

namespace {
bool isInNpuKernel(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return false;
  // 检查函数是否有 npu.target = "npu" 属性
  // 你也可以根据实际情况只检查是否有 "npu.target" 属性存在
  if (auto attr = funcOp->getAttrOfType<StringAttr>("npu.target")) {
    return attr.getValue() == "npu";
  }
  return false;
}

// =========================================================
// Pattern 1: 将 memref.copy 转换为 NPU DMA (Mvin / Mvout)
// =========================================================
class ConvertMemrefCopyToNpuDmaPattern
    : public OpRewritePattern<memref::CopyOp> {
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

    bool isMvin = (srcSpace == 0 && dstSpace == 2);
    bool isMvout = (srcSpace == 2 && dstSpace == 0);

    if (!isMvin && !isMvout) {
      return failure();
    }

    Location loc = op.getLoc();

    // 1. 获取 Shape 信息
    auto shape = srcType.getShape();
    int64_t rank = shape.size();

    // ================== 修复开始 ==================
    // 处理 Rank 1 (Vector) 和 Rank >= 2 (Tensor) 的情况
    int64_t rows, cols;

    if (rank == 0) {
      return failure();
    } else if (rank == 1) {
      // 1D Tensor
      rows = 1;
      cols = shape[0];
    } else if (rank == 2) {
      // [FIX] 新增：专门处理 Rank 2 (例如 3x32 的 Bias/Scale 表)
      // 对于 Rank 2，通常不需要跳过末尾的 packing 维度
      rows = shape[0];
      cols = shape[1];
    } else {
      // Rank >= 3 (Tensor)
      // 这里的逻辑假设最后一位是 packing (例如 32)，所以取 rank-3 和 rank-2
      // 例子: 1x3x1280x960x32 -> 取 1280 (H) 和 960 (W)
      rows = shape[rank - 3];
      cols = shape[rank - 2];
    }

    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows, 16);

    // 2. Stride 计算
    // 你的 Runtime 需要 dram_stride 和 sram_stride。
    // 如果是连续内存，Stride = Width (Cols)。
    // 如果是 Subview，我们需要从 Type 中提取 Stride，这里简化处理，假设
    // Row-Major 且 Dense。 实际项目中应使用 memref::getStridesAndOffset。
    Value vSramStride = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);

    // 对于 DRAM Stride，如果是 subview，它可能不等于 cols。
    // 这里为了跑通先 Mock 为 64 (和你之前一样)，或者等于 cols。
    // 如果需要精确控制，可以检查 srcType.getLayout()。
    Value vDramStride = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);

    // 3. 其他配置 (默认值)
    Value vZero32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value vFalse = rewriter.create<arith::ConstantIntOp>(loc, 0, 1);
    Value vZero8 = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);
    Value vZero16 = rewriter.create<arith::ConstantIntOp>(loc, 0, 16);
    Value vPrecision =
        rewriter.create<arith::ConstantIntOp>(loc, 1, 8); // 假设 1=int8/fp32

    if (isMvin) {
      // DRAM -> SRAM
      // 参数顺序参考 Npux.td: host_ptr, sram_memref, ...
      rewriter.create<DmaMvinOp>(
          loc, src, dst, vCol, vRow, vSramStride, vDramStride, vPrecision,
          vZero8, vZero8,                   // precision, input_type, dest
          vFalse, vZero32, vZero16, vZero16 // quant params
      );
    } else {
      // SRAM -> DRAM
      // 参数顺序参考 Npux.td: host_ptr, sram_memref, ...
      // 注意: copy(src, dst) -> copy(SRAM, DRAM) -> mvout(DRAM, SRAM)
      // 所以 mvout 的第一个参数是 dst (DRAM)，第二个是 src (SRAM)
      rewriter.create<DmaMvoutOp>(
          loc, dst, src, vCol, vRow, vSramStride, vDramStride, vPrecision,
          vZero8, vZero8,                   // precision, output_type, source
          vFalse, vZero32, vZero16, vZero16 // quant params
      );
    }

    rewriter.eraseOp(op);
    return success();
  }
};


class ConvertHostAllocToNpuxPattern : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      memref::AllocOp op, PatternRewriter &rewriter) const override {

    if (!isInNpuKernel(op)) {
      return failure();
    }
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

    // 1. 同样先检查 Scope，防止误伤 host 侧代码
    if (!isInNpuKernel(op)) {
      return failure();
    }

    if (space == 2) {
      rewriter.create<npux::SramFreeOp>(op.getLoc(), memref);
      rewriter.eraseOp(op);
      return success();
    }

    // 3. 如果是 DRAM (Space 0)，转换为 npux.free
    if (space == 0) {
      rewriter.create<npux::FreeOp>(op.getLoc(), memref);
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

} // namespace

void npux::populateSramDataMovementPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertMemrefCopyToNpuDmaPattern>(patterns.getContext());
}

void npux::populateHostAllocToNpuxPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertHostAllocToNpuxPattern>(patterns.getContext());
  patterns.add<ConvertDeallocToNpuxPattern>(patterns.getContext());
}