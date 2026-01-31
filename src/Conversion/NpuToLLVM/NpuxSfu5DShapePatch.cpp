

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "src/Dialect/Npux/NpuxOps.hpp" 
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace npux;

namespace {

// 定义 Pass 类
struct NpuxSfu5DShapePatchPass : public PassWrapper<NpuxSfu5DShapePatchPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuxSfu5DShapePatchPass)

  llvm::StringRef getArgument() const override { return "npux-sfu-reshape"; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(&getContext());

    // 遍历函数中的所有操作
    func.walk([&](SfuRunOp sfuOp) {
      // 1. 获取输入 MemRef 类型
      Value input = sfuOp.getInputSramMemref();
      auto memRefType = dyn_cast<MemRefType>(input.getType());

      if (!memRefType) return;

      // 2. 检查是否为 5 维 (NCHWc32)
      if (memRefType.getRank() != 5) return;

      ArrayRef<int64_t> shape = memRefType.getShape();
      // shape 索引: 0:N, 1:C, 2:H, 3:W, 4:c
      // 确保维度是静态的
      for (auto dim : shape) {
        if (dim == ShapedType::kDynamic) return; // 暂不处理动态形状
      }

      // 3. 计算新的参数值
      // "col改成后三维相乘-1" -> dim[2] * dim[3] * dim[4] - 1
      int64_t dim2 = shape[2];
      int64_t dim3 = shape[3];
      int64_t dim4 = shape[4];

      int64_t newColVal = (dim2 * dim3 * dim4) - 1;
      
      // "row改成第四维-1" -> dim[3] - 1 (索引从0开始，第四维是index 3)
      int64_t newRowVal = shape[1] - 1;

      // "stride改成col" -> 使用 newColVal
      int64_t newStrideVal = newColVal;

      // 准备常量 Value (i16)
      // 注意：builder 的插入点需要设置在合适的位置，这里为了简单，
      // 我们在 sfuOp 之前插入常量，如果常量已存在 MLIR 会自动折叠(CSE)但不保证位置，
      // 最稳妥的是在 sfuOp 之前插入。
      builder.setInsertionPoint(sfuOp);
      
      auto cCol = builder.create<arith::ConstantIntOp>(sfuOp.getLoc(), newColVal, 16);
      auto cRow = builder.create<arith::ConstantIntOp>(sfuOp.getLoc(), newRowVal, 16);
      auto cStride = builder.create<arith::ConstantIntOp>(sfuOp.getLoc(), newStrideVal, 16);

      // 4. 更新 SFU_RUN 自身的参数 (如果有参数要改也改一下)
      // 保持一致性，更新 SFU 的 input_col_num 和 input_row_num
      sfuOp.getInputColNumMutable().assign(cCol);
      sfuOp.getInputRowNumMutable().assign(cRow);

      // 5. 更新输入 DMA (dma_mvin)
      // 查找定义 input 的 Op
      for (Operation *user : input.getUsers()) {
        if (auto mvinOp = dyn_cast<DmaMvinOp>(user)) {
          // 额外的安全检查：确保 input 是作为 dma_mvin 的 dst_memref 使用的
          if (mvinOp.getDstMemref() != input) continue;

          builder.setInsertionPoint(mvinOp);
          
          auto mvinCol = builder.create<arith::ConstantIntOp>(mvinOp.getLoc(), newColVal, 16);
          auto mvinRow = builder.create<arith::ConstantIntOp>(mvinOp.getLoc(), newRowVal, 16);
          auto mvinStride = builder.create<arith::ConstantIntOp>(mvinOp.getLoc(), newStrideVal, 16);

          mvinOp.getColNumMutable().assign(mvinCol);
          mvinOp.getRowNumMutable().assign(mvinRow);
          
          // 更新 stride
          mvinOp.getSramStrideMutable().assign(mvinStride);
          mvinOp.getDramStrideMutable().assign(mvinStride);
        }
      }

      // 6. 更新输出 DMA (dma_mvout)
      // 查找使用 output 的 Op
      Value output = sfuOp.getOutputSramMemref();
      for (Operation *user : output.getUsers()) {
        if (auto mvoutOp = dyn_cast<DmaMvoutOp>(user)) {
          builder.setInsertionPoint(mvoutOp);
          
          auto mvoutCol = builder.create<arith::ConstantIntOp>(mvoutOp.getLoc(), newColVal, 16);
          auto mvoutRow = builder.create<arith::ConstantIntOp>(mvoutOp.getLoc(), newRowVal, 16);
          auto mvoutStride = builder.create<arith::ConstantIntOp>(mvoutOp.getLoc(), newStrideVal, 16);

          mvoutOp.getColNumMutable().assign(mvoutCol);
          mvoutOp.getRowNumMutable().assign(mvoutRow);
          // 同理更新 stride
          mvoutOp.getSramStrideMutable().assign(mvoutStride);
          mvoutOp.getDramStrideMutable().assign(mvoutStride);
        }
      }
    });
  }
};

} // namespace

// 注册 Pass 的函数
std::unique_ptr<Pass> npux::createNpuxSfu5DShapePatchPass() {
  return std::make_unique<NpuxSfu5DShapePatchPass>();
}