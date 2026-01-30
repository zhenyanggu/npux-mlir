//=============================================================================
// src/Conversion/NpuToLLVM/WeightHoisting.cpp
//
// Implements the "Weight Hoisting" optimization:
// 1. Identifies weight buffers loaded inside loops via dma_mvin.
// 2. Hoists the allocation and loading to the outer scope (Global SRAM).
// 3. Replaces inner allocations with 'npux.subview' (SRAM Slicing).
// 4. Calculates precise DMA parameters based on hardware constraints.
//=============================================================================

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Dialect/Npux/NpuxOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace npux;

namespace {

//=============================================================================
// Helper Structure & Function
//=============================================================================

struct MvinParams {
  Value col;
  Value row;
  Value strideSram;
  Value strideDram;
};

// 计算参数并创建 arith.constant Op
MvinParams createFullMvinParams(OpBuilder &builder, Location loc,
                                Value rootDramBuffer, ArrayAttr dseTilingAttr) {

  auto weightType = cast<MemRefType>(rootDramBuffer.getType());
  auto shape = weightType.getShape();
  
  // 安全检查：确保 shape 合法
  if (shape.size() < 4) {
      llvm::errs() << "Error: Weight shape dim < 4\n";
      Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 16);
      return {zero, zero, zero, zero}; 
  }
  
  int64_t kh = shape[shape.size() - 4];
  int64_t kw = shape[shape.size() - 3];

  // 这里不再查找属性，而是检查传入的属性是否为空
  if (!dseTilingAttr || dseTilingAttr.size() < 4) {
    llvm::errs() << "Error: npu.dse_tiling missing or invalid on ComputeOp\n";
    Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 16);
    return {zero, zero, zero, zero};
  }

  // 下面的计算逻辑保持不变
  int64_t tileIC = cast<IntegerAttr>(dseTilingAttr[2]).getInt();
  int64_t tileOC = cast<IntegerAttr>(dseTilingAttr[3]).getInt();

  int64_t logicalRows = tileOC / 32;
  int64_t icOuter = tileIC / 32;
  int64_t logicalWidth = icOuter * kh * kw * 32 * 32;

  int64_t hwRowNum = logicalRows - 1;
  int64_t hwColNum = logicalWidth - 1;
  int64_t hwStride = logicalWidth;

  Value rowVal = builder.create<arith::ConstantIntOp>(loc, hwRowNum, 16);
  Value colVal = builder.create<arith::ConstantIntOp>(loc, hwColNum, 16);
  Value strideVal = builder.create<arith::ConstantIntOp>(loc, hwStride, 16);

  return {colVal, rowVal, strideVal, strideVal};
}

//=============================================================================
// The Optimization Pass
//=============================================================================

class WeightHoistingPass
    : public PassWrapper<WeightHoistingPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WeightHoistingPass)

  StringRef getArgument() const override { return "npu-weight-hoisting"; }
  StringRef getDescription() const override {
    return "Hoist weight DMA MVIN out of loops";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());

    DenseMap<Value, Value> hoistedWeights;
    llvm::SmallVector<Operation *> opsToErase;

    func.walk([&](npux::ComputeRunOp computeOp) {
      // 1. [FIX] 直接从当前 ComputeOp 获取 Tiling 属性
      // 因为 IR 里是: npux.compute_run ... {npu.dse_tiling = [...]}
      ArrayAttr dseTilingAttr = computeOp->getAttrOfType<ArrayAttr>("npu.dse_tiling");

      // Input B corresponds to Weight
      Value weightBuffer = computeOp.getInputB();

      auto *sramAllocOp = weightBuffer.getDefiningOp();
      if (!sramAllocOp || !isa<npux::SramAllocOp>(sramAllocOp))
        return;

      npux::DmaMvinOp mvinOp;
      for (Operation *user : weightBuffer.getUsers()) {
        if (auto op = dyn_cast<npux::DmaMvinOp>(user)) {
          if (op.getSramMemref() == weightBuffer) {
            mvinOp = op;
            break;
          }
        }
      }
      if (!mvinOp) return;

      Value hostPtr = mvinOp.getHostPtr();
      auto memrefSubview = hostPtr.getDefiningOp<memref::SubViewOp>();
      if (!memrefSubview) return;

      Value rootDramBuffer = memrefSubview.getSource();
      Value fullSramBuffer;

      // =========================================================
      // 1. Hoist Logic (Global Alloc/Mvin/Free)
      // =========================================================
      if (hoistedWeights.count(rootDramBuffer)) {
        fullSramBuffer = hoistedWeights[rootDramBuffer];
      } else {
        OpBuilder::InsertionGuard guard(rewriter);
        
        // A. Insert Alloc at Start
        if (auto defOp = rootDramBuffer.getDefiningOp()) {
          rewriter.setInsertionPointAfter(defOp);
        } else {
          rewriter.setInsertionPointToStart(rootDramBuffer.getParentBlock());
        }

        // [FIX] 删除原本在这里查找 tilingAttr 的代码
        // 因为属性不在 Buffer 定义上，也不在 Function 上，而是在 ComputeOp 上

        auto dramType = cast<MemRefType>(rootDramBuffer.getType());
        auto sramType = MemRefType::get(
            dramType.getShape(),
            dramType.getElementType(),
            dramType.getLayout(), 
            rewriter.getI32IntegerAttr(2)
        );

        auto newAlloc = rewriter.create<npux::SramAllocOp>(
            rootDramBuffer.getLoc(), sramType, ValueRange{});

        // [FIX] 将从 computeOp 获取到的 dseTilingAttr 传进去
        MvinParams params = createFullMvinParams(rewriter, mvinOp.getLoc(), 
                                                 rootDramBuffer, dseTilingAttr);

        rewriter.create<npux::DmaMvinOp>(
            mvinOp.getLoc(),
            rootDramBuffer, 
            newAlloc,       
            params.col, 
            params.row, 
            params.strideSram, 
            params.strideDram,
            mvinOp.getPrecision(), 
            mvinOp.getInputType(), 
            mvinOp.getDest(),
            mvinOp.getIsQuant(), 
            mvinOp.getQuantZero(),
            mvinOp.getQuantScale(),
            mvinOp.getQuantShift()
        );

        // B. Insert Free at End
        Block *parentBlock = newAlloc->getBlock();
        rewriter.setInsertionPoint(parentBlock->getTerminator());
        rewriter.create<npux::SramFreeOp>(rootDramBuffer.getLoc(), newAlloc);

        fullSramBuffer = newAlloc;
        hoistedWeights[rootDramBuffer] = fullSramBuffer;
      }

      // =========================================================
      // 2. Create Subview (Inside Loop)
      // =========================================================
      rewriter.setInsertionPoint(mvinOp);

      SmallVector<Value> newOffsets;
      auto mixedOffsets = memrefSubview.getMixedOffsets();
      Location loc = memrefSubview.getLoc();

      for (auto offset : mixedOffsets) {
        if (auto dynamicOffset = llvm::dyn_cast_if_present<Value>(offset)) {
           newOffsets.push_back(dynamicOffset);
        } else {
           auto attr = llvm::cast<Attribute>(offset);
           int64_t intVal = llvm::cast<IntegerAttr>(attr).getInt();
           Value cst = rewriter.create<arith::ConstantIndexOp>(loc, intVal);
           newOffsets.push_back(cst);
        }
      }
      
      auto npuSubview = rewriter.create<npux::SubviewOp>(
          loc,
          weightBuffer.getType(), // Slice Type
          fullSramBuffer, 
          newOffsets      
      );

      // =========================================================
      // 3. Selective Replacement (Fix Dominance Error)
      // =========================================================
      // 我们只替换 Compute Op 的使用，不能替换外面的 SramFree Op
      
      for (Operation *user : llvm::make_early_inc_range(weightBuffer.getUsers())) {
          if (isa<npux::SramFreeOp>(user)) {
              // 这是一个旧的 Free 指令，标记删除
              opsToErase.push_back(user);
          } 
          else if (user == mvinOp) {
              // 这是旧的 Mvin，会被删除
          } 
          else {
              // 假设其他都是 Compute 或者 Loop 内合法的 Use
              user->replaceUsesOfWith(weightBuffer, npuSubview);
          }
      }

      // Mark old alloc and mvin for deletion
      bool alreadyMarked = false;
      for(auto* op : opsToErase) if(op == mvinOp) alreadyMarked = true;
      
      if(!alreadyMarked) {
          opsToErase.push_back(mvinOp);
          opsToErase.push_back(sramAllocOp);
      }
    });

    // Cleanup
    for (auto op : opsToErase) {
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createWeightHoistingPass() {
  return std::make_unique<WeightHoistingPass>();
}