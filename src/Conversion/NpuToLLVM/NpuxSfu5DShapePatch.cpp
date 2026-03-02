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

// 辅助函数：计算 NPU DMA 需要的 2D 维度 (Row, Col)
// 逻辑：
// Row = N * C - 1
// Col = H * W * (InnerC if packed) - 1
struct Npu2DShape {
  int64_t rowVal;
  int64_t colVal;
  int64_t strideVal;
  bool isValid;
};

Npu2DShape calculateNpu2DShape(MemRefType type) {
  if (!type || !type.hasStaticShape()) return {0, 0, 0, false};
  
  ArrayRef<int64_t> shape = type.getShape();
  int rank = type.getRank();
  
  int64_t n = 1, c = 1, h = 1, w = 1, c_inner = 1;

  // 针对 NCHW (Rank 4)
  if (rank == 4) {
    n = shape[0];
    c = shape[1];
    h = shape[2];
    w = shape[3];
  } 
  // 针对 NCHWc32 (Rank 5: N, C_outer, H, W, C_inner)
  else if (rank == 5) {
    n = shape[0];
    c = shape[1]; // C_outer
    h = shape[2];
    w = shape[3];
    c_inner = shape[4]; // 32
  } else {
    // 其他 Rank 暂不支持自动计算
    return {0, 0, 0, false};
  }

  // 核心逻辑：
  // Row = N * C (如果是 Packed，C 就是 C_outer)
  int64_t totalRow = n * c;
  
  // Col = H * W * C_inner (如果是 Packed，要把 C_inner 算进 Col)
  // 如果是 NCHW，C_inner 是 1
  int64_t totalCol = h * w * c_inner;

  // 寄存器值需要 -1
  int64_t rowReg = totalRow - 1;
  int64_t colReg = totalCol - 1;
  
  // 根据你的描述：stride 使用 col 的值
  int64_t strideReg = colReg; 

  return {rowReg, colReg, strideReg, true};
}

// 辅助函数：更新 DMA Op 的 Shape 参数
void updateDmaOp(Operation* op, Value memref, const Npu2DShape& shapeCfg, OpBuilder& builder) {
  builder.setInsertionPoint(op);
  auto cCol = builder.create<arith::ConstantIntOp>(op->getLoc(), shapeCfg.colVal, 16);
  auto cRow = builder.create<arith::ConstantIntOp>(op->getLoc(), shapeCfg.rowVal, 16);
  auto cStride = builder.create<arith::ConstantIntOp>(op->getLoc(), shapeCfg.strideVal, 16);

  if (auto mvinOp = dyn_cast<DmaMvinOp>(op)) {
    // 确保我们修改的是对应 memref 的 DMA
    if (mvinOp.getDstMemref() == memref) {
      mvinOp.getColNumMutable().assign(cCol);
      mvinOp.getRowNumMutable().assign(cRow);
      mvinOp.getSramStrideMutable().assign(cStride);
      mvinOp.getDramStrideMutable().assign(cStride);
    }
  } else if (auto mvoutOp = dyn_cast<DmaMvoutOp>(op)) {
    if (mvoutOp.getSramMemref() == memref) {
      mvoutOp.getColNumMutable().assign(cCol);
      mvoutOp.getRowNumMutable().assign(cRow);
      mvoutOp.getSramStrideMutable().assign(cStride);
      mvoutOp.getDramStrideMutable().assign(cStride);
    }
  }
}

// 定义 Pass 类
struct NpuxSfu5DShapePatchPass : public PassWrapper<NpuxSfu5DShapePatchPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuxSfu5DShapePatchPass)

  llvm::StringRef getArgument() const override { return "npux-sfu-reshape"; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(&getContext());

    // 遍历函数中的所有操作
    func.walk([&](Operation *op) {
      
      // ==========================================================
      // Case 1: SFU Run (Elementwise)
      // ==========================================================
      if (auto sfuOp = dyn_cast<SfuRunOp>(op)) {
        Value input = sfuOp.getInputSramMemref();
        Value output = sfuOp.getOutputSramMemref();
        auto memRefType = dyn_cast<MemRefType>(input.getType());
        
        Npu2DShape cfg = calculateNpu2DShape(memRefType);
        if (!cfg.isValid) return;

        // 1. 更新 SFU Op 自身参数 (input_col/row)
        builder.setInsertionPoint(sfuOp);
        auto cCol = builder.create<arith::ConstantIntOp>(sfuOp.getLoc(), cfg.colVal, 16);
        auto cRow = builder.create<arith::ConstantIntOp>(sfuOp.getLoc(), cfg.rowVal, 16);
        sfuOp.getInputColNumMutable().assign(cCol);
        sfuOp.getInputRowNumMutable().assign(cRow);

        // 2. 更新关联的 Input DMA (Mvin)
        for (Operation *user : input.getUsers()) {
            updateDmaOp(user, input, cfg, builder);
        }

        // 3. 更新关联的 Output DMA (Mvout)
        // SFU 输出维度通常与输入一致
        for (Operation *user : output.getUsers()) {
            updateDmaOp(user, output, cfg, builder);
        }
      } 
      // ==========================================================
      // Case 2: Resample Run (Resize)
      // ==========================================================
      else if (auto resampleOp = dyn_cast<ResampleOp>(op)) {
        // Resample 输入和输出维度不一样，需要分别计算
        
        // --- Input Side ---
        Value input = resampleOp.getInputSram();
        Npu2DShape inCfg = calculateNpu2DShape(dyn_cast<MemRefType>(input.getType()));
        
        if (inCfg.isValid) {
            builder.setInsertionPoint(resampleOp);
            auto cCol = builder.create<arith::ConstantIntOp>(resampleOp.getLoc(), inCfg.colVal, 16);
            auto cRow = builder.create<arith::ConstantIntOp>(resampleOp.getLoc(), inCfg.rowVal, 16);
            
            // 更新 Op 自身参数
            resampleOp.getInputColNumMutable().assign(cCol);
            resampleOp.getInputRowNumMutable().assign(cRow);

            // 更新 Input DMA
            for (Operation *user : input.getUsers()) {
                updateDmaOp(user, input, inCfg, builder);
            }
        }

        // --- Output Side ---
        Value output = resampleOp.getOutputSram();
        Npu2DShape outCfg = calculateNpu2DShape(dyn_cast<MemRefType>(output.getType()));
        
        if (outCfg.isValid) {
            // 更新 Output DMA
            for (Operation *user : output.getUsers()) {
                updateDmaOp(user, output, outCfg, builder);
            }
        }
      }
      // ==========================================================
      // Case 3: Layout Conversion (NCHW <-> NCHWc32)
      // ==========================================================
      // 对于 Layout Op，我们不修改 Op 本身的 n,c,h,w 参数（因为 CAPI 可能需要逻辑形状来计算地址转换），
      // 我们只修改 负责搬运数据进出的 DMA 配置，欺骗 DMA 以为在搬运 2D 数据。
      else if (isa<LayoutNchwToNchwc32Op>(op) || isa<LayoutNchwc32ToNchwOp>(op)) {
        Value input, output;
        
        if (auto packOp = dyn_cast<LayoutNchwToNchwc32Op>(op)) {
            input = packOp.getInputSram();
            output = packOp.getOutputSram();
        } else if (auto unpackOp = dyn_cast<LayoutNchwc32ToNchwOp>(op)) {
            input = unpackOp.getInputSram();
            output = unpackOp.getOutputSram();
        }

        // 更新 Input DMA
        Npu2DShape inCfg = calculateNpu2DShape(dyn_cast<MemRefType>(input.getType()));
        if (inCfg.isValid) {
            for (Operation *user : input.getUsers()) {
                updateDmaOp(user, input, inCfg, builder);
            }
        }

        // 更新 Output DMA
        Npu2DShape outCfg = calculateNpu2DShape(dyn_cast<MemRefType>(output.getType()));
        if (outCfg.isValid) {
            for (Operation *user : output.getUsers()) {
                updateDmaOp(user, output, outCfg, builder);
            }
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