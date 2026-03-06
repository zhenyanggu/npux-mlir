//=============================================================================
// src/Conversion/NpuPartition/NpuInsertDma.cpp
// This file inserts explicit mvin/mvout operations
// around NPU-executable linalg.generic operations.
//=============================================================================
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

//=============================================================================
// Helper: Create Identity Maps
//=============================================================================
static SmallVector<AffineMap> getIdentityMaps(MLIRContext *context, int rank) {
  return {AffineMap::getMultiDimIdentityMap(rank, context),
      AffineMap::getMultiDimIdentityMap(rank, context)};
}

//=============================================================================
// Helper: Create Medium Tensor for NPU Compute Op
// 专门用来为 NPU 算子提前准备 NPU 内部的 medium tensor (默认 memory_space = 2)
//=============================================================================
static Value createNpuMediumTensor(PatternRewriter &rewriter, Location loc,
    Value oldOutput, int64_t memorySpace = 2) {
  auto tensorType = cast<RankedTensorType>(oldOutput.getType());
  auto memSpaceAttr = rewriter.getI64IntegerAttr(memorySpace);

  // 分配一个新的 NPU Tensor，作为 linalg(op) 的直接输出目标
  return rewriter.create<bufferization::AllocTensorOp>(
      loc, tensorType, ValueRange{}, Value{}, memSpaceAttr);
}

//=============================================================================
// Helper: Collect dynamic sizes for a tensor value
//=============================================================================
static SmallVector<Value> getDynamicTensorSizes(
    PatternRewriter &rewriter, Location loc, Value tensor) {
  SmallVector<Value> dynamicSizes;
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType)
    return dynamicSizes;

  for (auto [idx, dim] : llvm::enumerate(tensorType.getShape())) {
    if (ShapedType::isDynamic(dim)) {
      dynamicSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, tensor, idx));
    }
  }

  return dynamicSizes;
}

//=============================================================================
// Helper: Stage a DMA input into internal DRAM
//
// Runtime 要求 DMA 访问的 DRAM 指针必须来自 npu_mem_alloc。这里显式创建一个
// memory_space=0 的内部 staging tensor，并通过普通拷贝把外部/未知来源的数据
// 先搬进去，再让后续 npu_dma_mvin 从这个 staging tensor 读取。
//=============================================================================
static Value createInternalDramStagingTensor(
    PatternRewriter &rewriter, Location loc, Value input) {
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<Value> dynamicSizes = getDynamicTensorSizes(rewriter, loc, input);

  Value stagingTensor = rewriter.create<bufferization::AllocTensorOp>(
      loc, inputType, dynamicSizes);

  SmallVector<utils::IteratorType> iteratorTypes(
      inputType.getRank(), utils::IteratorType::parallel);
  AffineMap indexingMap = rewriter.getMultiDimIdentityMap(inputType.getRank());
  SmallVector<AffineMap> maps(2, indexingMap);

  auto copyOp = rewriter.create<linalg::GenericOp>(loc, inputType,
      /*inputs=*/ValueRange{input},
      /*outputs=*/ValueRange{stagingTensor},
      /*indexingMaps=*/maps,
      /*iteratorTypes=*/iteratorTypes,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        b.create<linalg::YieldOp>(nestedLoc, args[0]);
      });

  return copyOp.getResult(0);
}

//=============================================================================
// Helper: Create DMA Generic Op (mvin or mvout)
//=============================================================================
static Value createDmaOp(PatternRewriter &rewriter, Location loc, Value input,
    StringRef dmaName, int64_t memorySpace, StringRef dmaType = "",
    Value dest = nullptr) {
  auto inputType = cast<RankedTensorType>(input.getType());

  // 1. 确定目标 Tensor (Destination)
  Value finalDest = dest;
  if (!finalDest) {
    // 如果没有传入 dest，则按原逻辑分配新内存 (常用于 MVIN)
    auto memSpaceAttr = rewriter.getI64IntegerAttr(memorySpace);
    finalDest = rewriter.create<bufferization::AllocTensorOp>(
        loc, inputType, ValueRange{}, Value{}, memSpaceAttr);
  }

  // 获取输出 Tensor 的类型（可能与输入类型略有不同，如 MemorySpace 不同）
  auto outType = cast<RankedTensorType>(finalDest.getType());

  // 2. 准备并行迭代类型和恒等映射
  SmallVector<utils::IteratorType> iteratorTypes(
      outType.getRank(), utils::IteratorType::parallel);

  // 这种写法能兼容输入和输出 Rank 一致的情况
  AffineMap indexingMaps = rewriter.getMultiDimIdentityMap(outType.getRank());
  // 确保有两个 map (一个给 input, 一个给 output)
  SmallVector<AffineMap> maps(2, indexingMaps);

  // 3. 创建 Generic Op
  auto dmaOp = rewriter.create<linalg::GenericOp>(loc, outType,
      /*inputs=*/ValueRange{input},
      /*outputs=*/ValueRange{finalDest},
      /*indexingMaps=*/maps,
      /*iteratorTypes=*/iteratorTypes,
      /*bodyBuilder=*/
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        b.create<linalg::YieldOp>(nestedLoc, args[0]);
      });

  // 4. 设置属性
  dmaOp->setAttr("library_call", rewriter.getStringAttr(dmaName));
  dmaOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
  if (!dmaType.empty()) {
    dmaOp->setAttr("npu.dma_type", rewriter.getStringAttr(dmaType));
  }

  return dmaOp.getResult(0);
}

//=============================================================================
// Pattern: NpuConvInsertDmaPattern
// Specialized logic for Conv ops with loop_stage awareness
//=============================================================================
struct NpuConvInsertDmaPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted"))
      return failure();

    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr || !libCallAttr.getValue().contains("conv"))
      return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    Location loc = op.getLoc();
    StringRef stage = "single";
    if (auto stageAttr = op->getAttrOfType<StringAttr>("npu.loop_stage")) {
      stage = stageAttr.getValue();
    }

    // ==============================================================
    // 1. Handle Inputs (MVIN) - 保持不变
    // ==============================================================
    SmallVector<Value> newInputs;
    int operandIdx = 0;
    for (Value operand : op.getInputs()) {
      Value processedInput = operand;
      if (operandIdx == 0) {
        Value stagedInput = createInternalDramStagingTensor(rewriter, loc, operand);
        processedInput =
            createDmaOp(rewriter, loc, stagedInput, "npu_dma_mvin", 2, "input");
      } else if (operandIdx == 1) {
        Value stagedWeight = createInternalDramStagingTensor(rewriter, loc, operand);
        processedInput =
            createDmaOp(rewriter, loc, stagedWeight, "npu_dma_mvin", 2, "weight");
      }
      newInputs.push_back(processedInput);
      operandIdx++;
    }

    // ==============================================================
    // 2. Handle Outputs (保持不变，Conv 输出仍然写入 Acc)
    // ==============================================================
    SmallVector<Value> newOutputs;
    newOutputs.push_back(op.getOutputs()[0]);

    // ==============================================================
    // 3. Create New Conv Op - 保持不变
    // ==============================================================
    auto newOp = cast<linalg::GenericOp>(rewriter.clone(*op.getOperation()));
    newOp.getInputsMutable().assign(newInputs);
    newOp.getOutputsMutable().assign(newOutputs);
    newOp->setAttr("npu.dma_inserted", rewriter.getUnitAttr());

    Value newConvResult = newOp.getResult(0);
    rewriter.replaceOp(op, newConvResult);

    // ==============================================================
    // 4. Handle MVOUT (核心修改区：适配新的流水线)
    // ==============================================================
    if (stage == "tail" || stage == "single") {
      // 这里的 users 已经是被 replaceOp 刷新过的，即真正消耗 Conv 结果的 Op
      for (Operation *user : newConvResult.getUsers()) {
        auto genericUser = dyn_cast<linalg::GenericOp>(user);
        if (!genericUser)
          continue;

        auto libCallAttr =
            genericUser->getAttrOfType<StringAttr>("library_call");
        if (libCallAttr && libCallAttr.getValue() == "mv_acc_to_spm") {

          rewriter.setInsertionPoint(genericUser);
          Location uLoc = genericUser.getLoc();

          // 这个 oldOutput 就是你 IR 中的 %extracted_slice_8
          Value oldOutput = genericUser.getOutputs()[0];

          // 步骤 4.1: 创建 Medium Tensor (分配 memory_space=2)
          Value mediumTensor =
              createNpuMediumTensor(rewriter, uLoc, oldOutput, 2);

          // 步骤 4.2: 克隆 mv_acc_to_spm，并将其输出对接到 Medium Tensor
          auto newMvAcc = cast<linalg::GenericOp>(
              rewriter.clone(*genericUser.getOperation()));
          newMvAcc.getInputsMutable().assign(newConvResult);
          newMvAcc.getOutputsMutable().assign(mediumTensor);

          // 步骤 4.3: 为输出插入 MVOUT (Medium Tensor -> MVOUT ->
          // %extracted_slice_8)
          Value mvoutResult = createDmaOp(rewriter, uLoc, newMvAcc.getResult(0),
              "npu_dma_mvout", 0, "", oldOutput);

          // 步骤 4.4: 替换旧的 mv_acc_to_spm
          // 因为 mvoutResult 的类型和原 mv_acc_to_spm 完全一致，
          // 下游的 tensor.insert_slice 会自动无缝接收
          // mvoutResult，不需要我们手动改它！
          rewriter.replaceOp(genericUser, mvoutResult);
          break;
        }
      }
    }

    return success();
  }
};

struct NpuGemmInsertDmaPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {

    // 0. 防止重复插入 DMA
    if (op->hasAttr("npu.dma_inserted"))
      return failure();

    // 1. 检查是否为 Gemm 或 MatMul
    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr)
      return failure();
    StringRef opName = libCallAttr.getValue();
    if (opName != "npu_gemm" && opName != "npu_matmul")
      return failure();

    // 2. 检查 Target 是否为 NPU
    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    Location loc = op.getLoc();
    StringRef stage = "single";
    if (auto stageAttr = op->getAttrOfType<StringAttr>("npu.loop_stage")) {
      stage = stageAttr.getValue();
    }

    // ==============================================================
    // 1. Handle Inputs (MVIN) - 取消 weight/input 的标签区分
    // ==============================================================
    SmallVector<Value> newInputs;
    for (auto it : llvm::enumerate(op.getInputs())) {
      size_t index = it.index();
      Value operand = it.value();

      if (index < 2) {
        // 仅对前两个输入 (通常是 LHS, RHS) 执行 MVIN，放入 Memory Space 2
        Value stagedInput = createInternalDramStagingTensor(rewriter, loc, operand);
        Value processedInput =
            createDmaOp(rewriter, loc, stagedInput, "npu_dma_mvin", 2, "");
        newInputs.push_back(processedInput);
      } else {
        // 超过两个的后续输入（如 Bias 或其他参数）保持原样
        newInputs.push_back(operand);
      }
    }

    // ==============================================================
    // 2. Handle Outputs (保持不变，Gemm 输出仍然写入 Acc)
    // ==============================================================
    SmallVector<Value> newOutputs;
    newOutputs.push_back(op.getOutputs()[0]);

    // ==============================================================
    // 3. Create New Gemm Op
    // ==============================================================
    auto newOp = cast<linalg::GenericOp>(rewriter.clone(*op.getOperation()));
    newOp.getInputsMutable().assign(newInputs);
    newOp.getOutputsMutable().assign(newOutputs);
    newOp->setAttr("npu.dma_inserted", rewriter.getUnitAttr());

    Value newGemmResult = newOp.getResult(0);
    rewriter.replaceOp(op, newGemmResult);

    // ==============================================================
    // 4. Handle MVOUT (适配 mv_acc_to_spm)
    // ==============================================================
    if (stage == "tail" || stage == "single") {
      // 遍历消耗 Gemm 结果的 User
      for (Operation *user : newGemmResult.getUsers()) {
        auto genericUser = dyn_cast<linalg::GenericOp>(user);
        if (!genericUser)
          continue;

        auto userLibCallAttr =
            genericUser->getAttrOfType<StringAttr>("library_call");
        if (userLibCallAttr && userLibCallAttr.getValue() == "mv_acc_to_spm") {

          rewriter.setInsertionPoint(genericUser);
          Location uLoc = genericUser.getLoc();

          // 这个 oldOutput 就是原始 IR 中的 Destination Tensor
          Value oldOutput = genericUser.getOutputs()[0];

          // 步骤 4.1: 创建 Medium Tensor (分配 memory_space=2)
          Value mediumTensor =
              createNpuMediumTensor(rewriter, uLoc, oldOutput, 2);

          // 步骤 4.2: 克隆 mv_acc_to_spm，并将其输出对接到 Medium Tensor
          auto newMvAcc = cast<linalg::GenericOp>(
              rewriter.clone(*genericUser.getOperation()));
          newMvAcc.getInputsMutable().assign(newGemmResult);
          newMvAcc.getOutputsMutable().assign(mediumTensor);

          // 步骤 4.3: 为输出插入 MVOUT (Medium Tensor -> MVOUT -> oldOutput)
          Value mvoutResult = createDmaOp(rewriter, uLoc, newMvAcc.getResult(0),
              "npu_dma_mvout", 0, "", oldOutput);

          // 步骤 4.4: 替换旧的 mv_acc_to_spm
          rewriter.replaceOp(genericUser, mvoutResult);
          break; // 处理完一个就跳出
        }
      }
    }

    return success();
  }
};

//=============================================================================
// Pattern: NpuUnaryInsertDmaPattern
// For unary ops (1 input, 1 output), insert mvin before and mvout after.
//=============================================================================
struct NpuUnaryInsertDmaPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    // 1. 防止重复处理
    if (op->hasAttr("npu.dma_inserted"))
      return failure();

    // 2. 检查是否为 NPU 算子
    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    // 3. 检查算子类型
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
      return failure();

    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr || libCallAttr == "npu_dma_mvin" ||
        libCallAttr == "npu_dma_mvout" || libCallAttr == "mv_acc_to_spm")
      return failure();

    // 如果是卷积，交给专门的 ConvPattern 处理
    if (libCallAttr.getValue().contains("conv"))
      return failure();

    Location loc = op.getLoc();

    // --- 步骤 1: 为输入插入 MVIN ---
    // createDmaOp 内部会自动为 mvin 申请 memory_space=2 的新 tensor
    Value input = op.getInputs()[0];
    Value stagedInput = createInternalDramStagingTensor(rewriter, loc, input);
    Value mvinResult =
      createDmaOp(rewriter, loc, stagedInput, "npu_dma_mvin", 2, "input");

    // --- 步骤 2: 准备 Medium Tensor (分配 memory_space=2 的新内存) ---
    // 无论原输出在哪，NPU 算子的直接输出必须写入 NPU 内部存储
    Value originalOutput = op.getOutputs()[0];
    Value mediumTensor =
        createNpuMediumTensor(rewriter, loc, originalOutput, 2);

    // --- 步骤 3: 克隆计算算子 (对接 Medium Tensor) ---
    auto newOp = cast<linalg::GenericOp>(rewriter.clone(*op.getOperation()));
    newOp.getInputsMutable().assign(mvinResult);
    newOp.getOutputsMutable().assign(mediumTensor); // 算子输出到 Medium Tensor
    newOp->setAttr("npu.dma_inserted", rewriter.getUnitAttr());

    // --- 步骤 4: 为输出插入 MVOUT ---
    // 逻辑：medium tensor(new) -> mvout -> old output tensor
    Value mvoutResult = createDmaOp(rewriter, loc, newOp.getResult(0),
        "npu_dma_mvout", 0, "output", originalOutput);

    // --- 步骤 5: 替换原算子 ---
    rewriter.replaceOp(op, mvoutResult);

    return success();
  }
};

//=============================================================================
// Pass Definition
//=============================================================================
struct NpuInsertDmaPass
    : public PassWrapper<NpuInsertDmaPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuInsertDmaPass)
  llvm::StringRef getArgument() const override { return "npu-insert-dma"; }
  llvm::StringRef getDescription() const override {
    return "Insert explicit mvin/mvout linalg.generic ops around NPU library "
           "calls.";
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 添加针对 Conv 和 Unary 的专用 Pattern
    patterns.add<NpuConvInsertDmaPattern>(context);
    patterns.add<NpuUnaryInsertDmaPattern>(context);
    patterns.add<NpuGemmInsertDmaPattern>(context);

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true).enableFolding(true);

    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuInsertDmaPass() {
  return std::make_unique<NpuInsertDmaPass>();
}