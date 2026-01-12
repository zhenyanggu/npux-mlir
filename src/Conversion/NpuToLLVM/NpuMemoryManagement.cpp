//=================================================
// src/Conversion/NpuToLLVM/SramDataMovement.cpp
// this file implements npu sram data movement insertion pass
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

class SramDataMovementPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    // 1. 检查是否是 NPU 目标
    if (!op->hasAttr("npu.target"))
      return failure();

    if (op.getInputs().empty())
      return failure();
    auto inType = cast<MemRefType>(op.getInputs()[0].getType());
    if (inType.getMemorySpaceAsInt() == 2) {
      return failure(); // 已经在 SRAM，那是 SFU Pattern 的活
    }

    Location loc = op.getLoc();

    // ==========================================
    // 3. 准备工作：创建 SRAM Buffer
    // ==========================================
    Value dramInput = op.getInputs()[0];
    Value dramOutput = op.getOutputs()[0]; // 这里实际上是 DPS 的 Init Tensor

    // 获取 Shape 以创建 Alloc
    auto shape = inType.getShape();
    auto sramType =
        MemRefType::get(shape, inType.getElementType(), {}, 2); // Space 2

    // 插入 Alloc
    Value sramIn = rewriter.create<memref::AllocOp>(loc, sramType);
    Value sramOut = rewriter.create<memref::AllocOp>(loc, sramType);

    // ==========================================
    // 4. 插入 DMA In (DRAM -> SRAM)
    // ==========================================
    // 需要构建 Shape/Stride 参数，这里复用你之前的逻辑
    int64_t rows = shape[shape.size() - 2];
    int64_t cols = shape[shape.size() - 1];
    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows, 16);

    // 这里的 Stride 需要认真处理，通常 Input (DRAM) 是 strided，SRAM 是
    // continuous 为了跑通，先 Mock 一下，实际需根据 op 属性计算
    Value vSramStride = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vDramStride =
        rewriter.create<arith::ConstantIntOp>(loc, 64, 16); // Mock

    // 其他配置 (类型、量化等) 暂时填默认值，因为具体的 SFU Pattern 会再解析一遍
    // 这里 DMA 主要是为了搬运 bit，只要长度对就行
    Value vZero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value vFalse = rewriter.create<arith::ConstantIntOp>(loc, 0, 1);
    Value vZero8 = rewriter.create<arith::ConstantIntOp>(loc, 0, 8);
    Value vZero16 = rewriter.create<arith::ConstantIntOp>(loc, 0, 16);

    rewriter.create<DmaMvinOp>(loc, dramInput, sramIn, vCol, vRow, vSramStride,
        vDramStride, vZero8, vZero8, vZero8, vFalse, vZero, vZero16, vZero16);

    // ==========================================
    // 5. 克隆计算节点 (DRAM Linalg -> SRAM Linalg)
    // ==========================================
    // 这是一个关键技巧：我们不直接在这里转 SFU，而是生成一个新的
    // linalg.generic， 但是把它的 operands 换成了 sramIn 和 sramOut。
    // 这样，下一个 Pattern (LinalgSfuToNpuxPattern) 就会匹配到它。

    // 使用 clone 进行复制
    Operation *newOp = rewriter.clone(*op.getOperation());
    auto newLinalgOp = dyn_cast<linalg::GenericOp>(newOp);

    // 替换操作数：Input -> sramIn, Output(Init) -> sramOut
    // 注意：Linalg 的操作数替换 API 比较琐碎，这里用最通用的 setOperands
    // 假设是单输入单输出：[Input, OutputBuffer]
    newOp->setOperand(0, sramIn);
    newOp->setOperand(
        1, sramOut); // 对于 GenericOp，Output 也是 operand (inputs + outputs)

    // ==========================================
    // 6. 插入 DMA Out (SRAM -> DRAM)
    // ==========================================
    rewriter.create<DmaMvoutOp>(loc, dramOutput, sramOut, vCol, vRow,
        vSramStride, vDramStride, vZero8, vZero8, vZero8, vFalse, vZero,
        vZero16, vZero16);

    // ==========================================
    // 7. 资源释放
    // ==========================================
    rewriter.create<memref::DeallocOp>(loc, sramIn);
    rewriter.create<memref::DeallocOp>(loc, sramOut);

    rewriter.eraseOp(op);

    return success();
  }
};

class ConvertHostAllocToNpuxPattern : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      memref::AllocOp op, PatternRewriter &rewriter) const override {
    // 1. 如果已经是 SRAM (space=2) 或者已经是 NPU managed (space=1)，跳过
    if (op.getType().getMemorySpaceAsInt() != 0) {
      return failure();
    }

    // 2. 检查该 alloc 的用途 (Users)
    // 我们只替换那些被传递给 NPU Kernel 的 alloc
    bool usedByNpuKernel = false;
    for (Operation *user : op->getUsers()) {
      // 情况 A: 直接作为参数传给 CallOp
      if (auto callOp = dyn_cast<func::CallOp>(user)) {
        // 检查被调用的函数是否有 npu.target 属性
        auto module = op->getParentOfType<ModuleOp>();
        auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
        if (callee && callee->hasAttr("npu.target")) {
          usedByNpuKernel = true;
          break;
        }
      }
      // 情况 B: 这里可以扩展，例如通过 view/reshape 传给 Kernel
      // 目前简单起见只处理直接传递
    }

    if (!usedByNpuKernel) {
      return failure();
    }

    // 3. 直接使用原 Op 的类型 (Space 0)
    auto memRefType = op.getType(); 

    // 4. 创建 npux.alloc
    // 注意：因为类型完全一样，我们不需要任何 Cast！
    auto newAlloc = rewriter.create<npux::AllocOp>(
        op.getLoc(), 
        memRefType,             // 使用原始类型 (Space 0)
        op.getDynamicSizes()
    );

    // 5. 直接替换
    rewriter.replaceOp(op, newAlloc);
    return success();
  }
};

void npux::populateSramDataMovementPatterns(RewritePatternSet &patterns) {
  patterns.add<SramDataMovementPattern>(patterns.getContext());
}
void npux::populateHostAllocToNpuxPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertHostAllocToNpuxPattern>(patterns.getContext());
}