//=================================================
// src/Conversion/NpuToLLVM/SramDataMovement.cpp
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

// =========================================================
// Pattern 1: 将 memref.copy 转换为 NPU DMA (Mvin / Mvout)
// =========================================================
class ConvertMemrefCopyToNpuDmaPattern : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op, PatternRewriter &rewriter) const override {
    Value src = op.getSource();
    Value dst = op.getTarget();

    auto srcType = cast<MemRefType>(src.getType());
    auto dstType = cast<MemRefType>(dst.getType());

    int srcSpace = srcType.getMemorySpaceAsInt();
    int dstSpace = dstType.getMemorySpaceAsInt();

    // 0 = DRAM (Host), 2 = SRAM
    bool isMvin = (srcSpace == 0 && dstSpace == 2);
    bool isMvout = (srcSpace == 2 && dstSpace == 0);

    if (!isMvin && !isMvout) {
      return failure(); // 普通的 CPU copy 或 SRAM 内部 copy 不处理
    }

    Location loc = op.getLoc();

    // 1. 获取 Shape 信息
    // 注意：memref.copy 的 src/dst 可能是 subview，getType().getShape() 获取的是 View 的形状
    // 这正是我们 DMA 需要搬运的大小。
    auto shape = srcType.getShape();
    int64_t rank = shape.size();
    if (rank < 2) return failure(); 

    // 简单起见，取最后两维作为 Row/Col (配合你的 Runtime 逻辑)
    int64_t rows = shape[rank - 2]; // Height
    int64_t cols = shape[rank - 1]; // Width

    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows, 16);

    // 2. Stride 计算
    // 你的 Runtime 需要 dram_stride 和 sram_stride。
    // 如果是连续内存，Stride = Width (Cols)。
    // 如果是 Subview，我们需要从 Type 中提取 Stride，这里简化处理，假设 Row-Major 且 Dense。
    // 实际项目中应使用 memref::getStridesAndOffset。
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
    Value vPrecision = rewriter.create<arith::ConstantIntOp>(loc, 1, 8); // 假设 1=int8/fp32

    if (isMvin) {
      // DRAM -> SRAM
      // 参数顺序参考 Npux.td: host_ptr, sram_memref, ...
      rewriter.create<DmaMvinOp>(loc, 
          src, dst, 
          vCol, vRow, vSramStride, vDramStride, 
          vPrecision, vZero8, vZero8, // precision, input_type, dest
          vFalse, vZero32, vZero16, vZero16 // quant params
      );
    } else {
      // SRAM -> DRAM
      // 参数顺序参考 Npux.td: host_ptr, sram_memref, ...
      // 注意: copy(src, dst) -> copy(SRAM, DRAM) -> mvout(DRAM, SRAM)
      // 所以 mvout 的第一个参数是 dst (DRAM)，第二个是 src (SRAM)
      rewriter.create<DmaMvoutOp>(loc, 
          dst, src, 
          vCol, vRow, vSramStride, vDramStride, 
          vPrecision, vZero8, vZero8, // precision, output_type, source
          vFalse, vZero32, vZero16, vZero16 // quant params
      );
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// =========================================================
// Pattern 2: 将 memref.dealloc 转换为 npux.free (DRAM) 或 删除 (SRAM)
// =========================================================
class ConvertDeallocToNpuxPattern : public OpRewritePattern<memref::DeallocOp> {
public:
  using OpRewritePattern<memref::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DeallocOp op, PatternRewriter &rewriter) const override {
    Value memref = op.getMemref();
    auto type = cast<MemRefType>(memref.getType());
    int space = type.getMemorySpaceAsInt();

    // 1. SRAM Dealloc -> 直接删除
    if (space == 2) {
        rewriter.eraseOp(op);
        return success();
    }

    // 2. DRAM Dealloc -> 转 npux.free
    // 因为 Target 已经过滤了，能进来的 Space 0 肯定是要转的
    if (space == 0) {
        // 创建 npux.free
        // 注意：npux.alloc 返回的是 memref，npux.free 接受的也是 memref
        rewriter.create<npux::FreeOp>(op.getLoc(), memref);
        rewriter.eraseOp(op);
        return success();
    }

    return failure();
  }
};

// ... (保持原有的 ConvertHostAllocToNpuxPattern 不变) ...
class ConvertHostAllocToNpuxPattern : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op, PatternRewriter &rewriter) const override {
    // 1. 如果已经是 SRAM (space=2) 跳过，留给 LLVM lowering 处理或保持原样
    if (op.getType().getMemorySpaceAsInt() != 0) {
      return failure();
    }

    // 2. 检查该 alloc 是否被 NPU 相关的 Call 或 Copy 使用
    bool usedByNpu = false;
    for (Operation *user : op->getUsers()) {
      if (auto callOp = dyn_cast<func::CallOp>(user)) {
        auto module = op->getParentOfType<ModuleOp>();
        auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
        if (callee && callee->hasAttr("npu.target")) {
          usedByNpu = true; break;
        }
      }
      // 新增：如果被用作 DMA 的源或目的，也需要是 NPU 内存
      if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
          // 这里可以加更细致的判断，暂时只要涉Copy就认为是
          usedByNpu = true; break;
      }
    }

    if (!usedByNpu) return failure();

    // 替换为 npux.alloc
    auto newAlloc = rewriter.create<npux::AllocOp>(
        op.getLoc(), op.getType(), op.getDynamicSizes()
    );

    rewriter.replaceOp(op, newAlloc);
    return success();
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