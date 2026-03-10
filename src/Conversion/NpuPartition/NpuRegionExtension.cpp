//===============================================
// src/Conversion/NpuPartition/NpuRegionExtension.cpp
// this file extends the NPU execute_region boundary
// by sinking producers  and fusing consumers
// into the region, realizing zero-copy optimization.
//===============================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SmallSet.h"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {
static bool isDynamicFeatureMap(Value v) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp) return true; // Block 参数通常是网络的动态输入

  if (defOp->hasTrait<OpTrait::ConstantLike>()) {
    return false;
  }
  return true; 
}

static bool hasDataDependencyOrSharedInput(scf::ExecuteRegionOp a, scf::ExecuteRegionOp b) {
  // 条件 1: 显式数据依赖 (Region A 的输出直接喂给 Region B)
  for (Value res : a.getResults()) {
    for (Operation *user : res.getUsers()) {
      if (b->isProperAncestor(user)) return true;
    }
  }

  // 收集 A 的外部生产者
  llvm::SmallSet<Operation *, 8> producersOfA;
  a.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      // 只记录产生“动态特征图”的外部生产者
      if (defOp && !a->isProperAncestor(defOp) && isDynamicFeatureMap(operand)) {
        producersOfA.insert(defOp);
      }
    }
  });

  // 条件 2: 共享动态特征图 (真正的 Y 型残差分支)
  bool shared = false;
  b.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (defOp && producersOfA.count(defOp)) {
        shared = true; // 抓到真正的残差分支了！
      }
    }
  });

  return shared;
}

//=============================================================================
// Helper 1: 安全移动 Op 及其依赖
//=============================================================================
static void moveOpAndDepsIntoRegion(Operation *op, Block *targetBlock,
    bool moveToStart, Operation *boundary, PatternRewriter &rewriter) {

  if (moveToStart) {
    rewriter.moveOpBefore(op, targetBlock, targetBlock->begin());
  } else {
    assert(boundary && "Boundary must be provided for consumers");
    rewriter.moveOpBefore(op, boundary);
  }

  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    for (OpOperand &initOperand : dpsOp.getDpsInitsMutable()) {
      Operation *defOp = initOperand.get().getDefiningOp();

      if (defOp && isa<tensor::EmptyOp>(defOp) &&
          defOp->getBlock() != targetBlock) {

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(targetBlock);

        Operation *newEmpty = rewriter.clone(*defOp);

        rewriter.modifyOpInPlace(
            op, [&] { initOperand.set(newEmpty->getResult(0)); });
      }
    }
  }
}

//=============================================================================
// Helper 2: Constant Input Copy
//=============================================================================
static void injectCopyForConstants(Operation *op, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  for (OpOperand &operand : op->getOpOperands()) {
    Value inputVal = operand.get();
    Operation *defOp = inputVal.getDefiningOp();

    if (defOp && defOp->getName().getStringRef() == "onnx.Constant") {
      auto tensorType = cast<RankedTensorType>(inputVal.getType());
      Location loc = op->getLoc();

      Value emptyBuffer = rewriter.create<tensor::EmptyOp>(
          loc, tensorType.getShape(), tensorType.getElementType());
      auto copyOp = rewriter.create<linalg::CopyOp>(loc, inputVal, emptyBuffer);

      rewriter.modifyOpInPlace(op, [&] {
        op->setOperand(operand.getOperandNumber(), copyOp.getResult(0));
      });
    }
  }
}

//=============================================================================
// Helper 3: Output Isolation
//=============================================================================
static void injectInternalOutputCopy(
    scf::YieldOp yieldOp, unsigned resultIdx, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(yieldOp);

  Value originalVal = yieldOp.getOperand(resultIdx);

  if (auto defOp = originalVal.getDefiningOp()) {
    if (isa<linalg::CopyOp>(defOp))
      return;
  }

  auto tensorType = cast<RankedTensorType>(originalVal.getType());
  Location loc = yieldOp.getLoc();

  Value emptyBuffer = rewriter.create<tensor::EmptyOp>(
      loc, tensorType.getShape(), tensorType.getElementType());

  auto copyOp = rewriter.create<linalg::CopyOp>(loc, originalVal, emptyBuffer);

  rewriter.modifyOpInPlace(
      yieldOp, [&] { yieldOp.setOperand(resultIdx, copyOp.getResult(0)); });
}

//=============================================================================
// Pattern 1: Extend Region Boundary (修复 Dominance 崩溃版)
//=============================================================================
struct ExtendRegionBoundary : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern<scf::ExecuteRegionOp>::OpRewritePattern;

  static constexpr StringRef kProcessedAttr = "npu.region_extended";

  LogicalResult matchAndRewrite(
      scf::ExecuteRegionOp execOp, PatternRewriter &rewriter) const override {

    Block *regionBlock = &execOp.getRegion().front();
    bool changed = false;

    if (execOp->hasAttr(kProcessedAttr)) {
      return failure();
    }

    //=========================================================================
    // Part 1. Input Sinking (生产者下沉)
    //=========================================================================
    for (Operation &op : llvm::make_early_inc_range(*regionBlock)) {
      if (isa<linalg::CopyOp>(op))
        continue;

      for (Value operand : op.getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp || defOp->getBlock() != execOp->getBlock())
          continue;

        if (isa<scf::ExecuteRegionOp>(defOp))
            continue;

        if (defOp->getName().getStringRef() == "onnx.Constant") {
          injectCopyForConstants(&op, rewriter);
          changed = true;
        } else {
          bool allUsersInRegion = llvm::all_of(defOp->getUsers(), [&](Operation *user) {
            return execOp->isProperAncestor(user);
          });

          if (!allUsersInRegion) {
            continue; 
          }

          moveOpAndDepsIntoRegion(defOp, regionBlock, true, nullptr, rewriter);
          changed = true;
        }
      }
    }

    //=========================================================================
    // Part 2. Output Fusion (消费者融合)
    //=========================================================================
    Operation *terminator = regionBlock->getTerminator();
    auto yieldOp = cast<scf::YieldOp>(terminator);

    for (OpResult result : execOp.getResults()) {
      unsigned resultIdx = result.getResultNumber();

      bool canFuse = true;
      Operation *user = nullptr;

      if (!result.hasOneUse()) {
        canFuse = false;
      } else {
        user = *result.getUsers().begin();
        if (user->getBlock() != execOp->getBlock())
          canFuse = false;
        if (user->getNumResults() != 1)
          canFuse = false;
        if (user->hasTrait<OpTrait::IsTerminator>())
          canFuse = false;
        if (isa<scf::ExecuteRegionOp>(user))
          canFuse = false;

        // [NEW FIX 3] Dominance 安全检查！
        // 检查 user 的所有输入：如果在同一个 Block 内，必须排在 execOp 前面
        if (canFuse) {
          for (Value operand : user->getOperands()) {
            Operation *operandDefOp = operand.getDefiningOp();
            if (operandDefOp && operandDefOp->getBlock() == execOp->getBlock()) {
              // 如果操作数在当前 Region 的后面才定义，严禁融合！
              if (!operandDefOp->isBeforeInBlock(execOp)) {
                canFuse = false;
                break;
              }
            }
          }
        }
      }

      // [Branch A] 无法融合 -> 隔离 (Isolation)
      if (!canFuse) {
        Value yieldOperand = yieldOp.getOperand(resultIdx);
        if (auto defOp = yieldOperand.getDefiningOp()) {
          if (isa<linalg::CopyOp>(defOp))
            continue; 
        }

        injectInternalOutputCopy(yieldOp, resultIdx, rewriter);
        changed = true; 
        continue;
      }

      // [Branch B] 执行融合 (Fusion)
      Value internalYieldVal = yieldOp.getOperand(resultIdx);
      Type newResultType = user->getResult(0).getType();

      moveOpAndDepsIntoRegion(
          user, regionBlock, /*moveToStart=*/false, yieldOp, rewriter);

      rewriter.modifyOpInPlace(
          user, [&] { user->replaceUsesOfWith(result, internalYieldVal); });

      rewriter.modifyOpInPlace(execOp, [&] { result.setType(newResultType); });

      rewriter.replaceAllUsesWith(user->getResult(0), result);

      rewriter.modifyOpInPlace(
          yieldOp, [&] { yieldOp.setOperand(resultIdx, user->getResult(0)); });

      changed = true;
    }

    if (changed) {
      rewriter.modifyOpInPlace(execOp,
          [&] { execOp->setAttr(kProcessedAttr, rewriter.getUnitAttr()); });
      return success();
    }

    return failure();
  }
};
struct MergeSiblingRegions : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern<scf::ExecuteRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      scf::ExecuteRegionOp regionA, PatternRewriter &rewriter) const override {

    // 1. 在同一个 Block 内向下寻找候选的 Region B
    scf::ExecuteRegionOp regionB = nullptr;
    for (Operation *next = regionA->getNextNode(); next != nullptr; next = next->getNextNode()) {
      if (auto candidate = dyn_cast<scf::ExecuteRegionOp>(next)) {
        if (hasDataDependencyOrSharedInput(regionA, candidate)) {
          regionB = candidate;
          break;
        }
      }
      // 安全守卫: 如果 A 和 B 之间有副作用的 Op (如内存写入、控制流)，则放弃融合
      // 这里假设你用 isMemoryEffectFree 来判断，如果没有副作用接口，可以简化判定
      if (!isMemoryEffectFree(next) && !next->hasTrait<OpTrait::IsTerminator>()) {
        break; 
      }
    }

    if (!regionB) return failure();

    // 2. 准备新的 Result Types (A 和 B 结果的并集)
    SmallVector<Type> fusedResultTypes;
    fusedResultTypes.append(regionA.getResultTypes().begin(), regionA.getResultTypes().end());
    fusedResultTypes.append(regionB.getResultTypes().begin(), regionB.getResultTypes().end());

    // 3. 创建合并后的新 ExecuteRegionOp
    Location loc = regionA.getLoc();
    auto fusedRegion = rewriter.create<scf::ExecuteRegionOp>(loc, fusedResultTypes);
    Block *fusedBlock = rewriter.createBlock(&fusedRegion.getRegion());

    // 4. 获取旧的 Yield 信息
    auto yieldA = cast<scf::YieldOp>(regionA.getRegion().front().getTerminator());
    auto yieldB = cast<scf::YieldOp>(regionB.getRegion().front().getTerminator());
    SmallVector<Value> yieldOperandsA(yieldA.getOperands());
    SmallVector<Value> yieldOperandsB(yieldB.getOperands());

    // 5. 将 A 和 B 的 Operations 移动到新的 Block 中
    for (Operation &op : llvm::make_early_inc_range(regionA.getRegion().front().without_terminator())) {
      rewriter.moveOpBefore(&op, fusedBlock, fusedBlock->end());
    }
    for (Operation &op : llvm::make_early_inc_range(regionB.getRegion().front().without_terminator())) {
      rewriter.moveOpBefore(&op, fusedBlock, fusedBlock->end());
    }

    // 6. 核心逻辑: 重定向依赖 (Rewiring)
    // 处理 Region A 的结果: 如果被 B 使用(现在已经移动到了内部)，使用内部 Value；如果被外部使用，使用新的 Result
    for (auto [idx, resA] : llvm::enumerate(regionA.getResults())) {
      Value internalVal = yieldOperandsA[idx];
      Value newExternalRes = fusedRegion.getResult(idx);

      for (OpOperand &use : llvm::make_early_inc_range(resA.getUses())) {
        if (fusedRegion->isProperAncestor(use.getOwner())) {
          // B 内部的操作，直接连接到 A 的内部输出
          rewriter.modifyOpInPlace(use.getOwner(), [&]() { use.set(internalVal); });
        } else {
          // 外部的其他操作，连接到新的 External Result
          rewriter.modifyOpInPlace(use.getOwner(), [&]() { use.set(newExternalRes); });
        }
      }
    }

    // 处理 Region B 的结果 (全部映射到新的 External Result)
    unsigned offset = regionA.getNumResults();
    for (auto [idx, resB] : llvm::enumerate(regionB.getResults())) {
      Value newExternalRes = fusedRegion.getResult(offset + idx);
      rewriter.replaceAllUsesWith(resB, newExternalRes);
    }

    // 7. 创建新的融合 YieldOp
    rewriter.setInsertionPointToEnd(fusedBlock);
    SmallVector<Value> fusedYieldOperands;
    fusedYieldOperands.append(yieldOperandsA.begin(), yieldOperandsA.end());
    fusedYieldOperands.append(yieldOperandsB.begin(), yieldOperandsB.end());
    rewriter.create<scf::YieldOp>(loc, fusedYieldOperands);

    // 8. 擦除旧的 Region (它们的 Yield 也会被一并擦除)
    rewriter.eraseOp(regionA);
    rewriter.eraseOp(regionB);

    return success();
  }
};
struct NpuRegionExtensionPass
    : public PassWrapper<NpuRegionExtensionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuRegionExtensionPass)
  llvm::StringRef getArgument() const override {
    return "npu-region-extension";
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ExtendRegionBoundary>(context);
    patterns.add<MergeSiblingRegions>(context);
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true).enableFolding(true);

    if (failed(applyPatternsGreedily(
            getOperation().getBody(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuRegionExtensionPass() {
  return std::make_unique<NpuRegionExtensionPass>();
}