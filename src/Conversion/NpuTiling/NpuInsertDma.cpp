//=============================================================================
// src/Conversion/NpuPartition/NpuInsertDma.cpp
// This file inserts explicit mvin/mvout operations
// around NPU-executable linalg.generic operations.
//=============================================================================
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
// Helper: Change Tensor Encoding
//=============================================================================
static RankedTensorType changeEncoding(RankedTensorType type, int64_t encoding, OpBuilder &b) {
  // 如果 encoding 为 0，通常代表外部主存，我们将其重置为 null Attribute (默认状态)
  Attribute encodingAttr = (encoding == 0) ? Attribute() : b.getI64IntegerAttr(encoding);
  return RankedTensorType::get(type.getShape(), type.getElementType(), encodingAttr);
}

//=============================================================================
// Helper: Create Medium Tensor for NPU Compute Op
// 专门用来为 NPU 算子提前准备 NPU 内部的 medium tensor (默认 encoding = 2)
//=============================================================================
static Value createNpuMediumTensor(PatternRewriter &rewriter, Location loc,
    Value oldOutput, int64_t encoding = 2) {
  auto tensorType = cast<RankedTensorType>(oldOutput.getType());
  auto newType = changeEncoding(tensorType, encoding, rewriter);

  // 分配一个新的 NPU Tensor，作为 linalg(op) 的直接输出目标
  return rewriter.create<bufferization::AllocTensorOp>(
      loc, newType, ValueRange{});
}

//=============================================================================
// Helper: Create DMA Generic Op (mvin or mvout)
//=============================================================================
static Value createDmaOp(PatternRewriter &rewriter, Location loc, Value input,
    StringRef dmaName, int64_t encoding, StringRef dmaType = "",
    Value dest = nullptr) {
  auto inputType = cast<RankedTensorType>(input.getType());

  // 1. 确定目标 Tensor (Destination)
  Value finalDest = dest;
  if (!finalDest) {
    // 如果没有传入 dest，则按原逻辑分配新内存 (通过修改 encoding)
    auto newType = changeEncoding(inputType, encoding, rewriter);
    finalDest = rewriter.create<bufferization::AllocTensorOp>(
        loc, newType, ValueRange{});
  }

  // 获取输出 Tensor 的类型（包含了新的 encoding 信息）
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
// Helper: Create MVIN to ACC(i32) for MATADD
// - Input: typically int8 in DRAM
// - Output: int32 in ACC (for accurate ACC capacity planning)
//=============================================================================
static Value createMataddAccMvinOp(
    PatternRewriter &rewriter, Location loc, Value input) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto accType = RankedTensorType::get(
      inputType.getShape(), rewriter.getI32Type(), rewriter.getI64IntegerAttr(3));
  Value accAlloc =
      rewriter.create<bufferization::AllocTensorOp>(loc, accType, ValueRange{});

  int64_t rank = inputType.getRank();
  SmallVector<utils::IteratorType> iteratorTypes(
      rank, utils::IteratorType::parallel);
  SmallVector<AffineMap> maps = {
      rewriter.getMultiDimIdentityMap(rank),
      rewriter.getMultiDimIdentityMap(rank)};

  auto dmaOp = rewriter.create<linalg::GenericOp>(loc, accType,
      /*inputs=*/ValueRange{input},
      /*outputs=*/ValueRange{accAlloc},
      maps, iteratorTypes,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value in = args[0];
        Value out = in;
        if (!in.getType().isInteger(32)) {
          out = b.create<arith::ExtSIOp>(nestedLoc, b.getI32Type(), in);
        }
        b.create<linalg::YieldOp>(nestedLoc, out);
      });

  dmaOp->setAttr("library_call", rewriter.getStringAttr("npu_dma_mvin"));
  dmaOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
  dmaOp->setAttr("npu.is_quant", rewriter.getBoolAttr(true));
  dmaOp->setAttr("npu.quant_zero", rewriter.getI32IntegerAttr(0));
  dmaOp->setAttr("npu.quant_scale", rewriter.getI16IntegerAttr(1));
  dmaOp->setAttr("npu.quant_shift", rewriter.getI16IntegerAttr(0));

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
        processedInput =
            createDmaOp(rewriter, loc, operand, "npu_dma_mvin", 2, "input");
      } else if (operandIdx == 1) {
        processedInput =
            createDmaOp(rewriter, loc, operand, "npu_dma_mvin", 2, "weight");
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

          // 步骤 4.1: 创建 Medium Tensor (分配 encoding=2)
          Value mediumTensor =
              createNpuMediumTensor(rewriter, uLoc, oldOutput, 2);

          // 步骤 4.2: 克隆 mv_acc_to_spm，并将其输出对接到 Medium Tensor
          auto newMvAcc = cast<linalg::GenericOp>(
              rewriter.clone(*genericUser.getOperation()));
          newMvAcc.getInputsMutable().assign(newConvResult);
          newMvAcc.getOutputsMutable().assign(mediumTensor);

          newMvAcc.getResult(0).setType(mediumTensor.getType());

          // 步骤 4.3: 为输出插入 MVOUT (Medium Tensor -> MVOUT ->
          // %extracted_slice_8)
          Value mvoutResult = createDmaOp(rewriter, uLoc, newMvAcc.getResult(0),
              "npu_dma_mvout", 0, "", oldOutput);

          // 步骤 4.4: 替换旧的 mv_acc_to_spm
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
        // 仅对前两个输入 (通常是 LHS, RHS) 执行 MVIN，赋予 encoding 2
        Value processedInput =
            createDmaOp(rewriter, loc, operand, "npu_dma_mvin", 2, "");
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

          // 步骤 4.1: 创建 Medium Tensor (分配 encoding=2)
          Value mediumTensor =
              createNpuMediumTensor(rewriter, uLoc, oldOutput, 2);

          // 步骤 4.2: 克隆 mv_acc_to_spm，并将其输出对接到 Medium Tensor
          auto newMvAcc = cast<linalg::GenericOp>(
              rewriter.clone(*genericUser.getOperation()));
          newMvAcc.getInputsMutable().assign(newGemmResult);
          newMvAcc.getOutputsMutable().assign(mediumTensor);

          newMvAcc.getResult(0).setType(mediumTensor.getType());

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

struct NpuMataddInsertDmaPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted"))
      return failure();

    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr || libCallAttr.getValue() != "npu_matadd")
      return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return failure();

    Location loc = op.getLoc();

    // 1) Two inputs: DRAM -> ACC(i32)
    SmallVector<Value> newInputs;
    newInputs.push_back(createMataddAccMvinOp(rewriter, loc, op.getInputs()[0]));
    newInputs.push_back(createMataddAccMvinOp(rewriter, loc, op.getInputs()[1]));

    // 2) Output medium tensor in SPM
    Value oldOutput = op.getOutputs()[0];
    Value mediumTensor = createNpuMediumTensor(rewriter, loc, oldOutput, 2);
    auto mediumType = cast<RankedTensorType>(mediumTensor.getType());

    // 3) New matadd generic that consumes ACC(i32) and writes SPM(i8)
    auto newOp = rewriter.create<linalg::GenericOp>(loc,
        TypeRange{mediumType},
        newInputs,
        ValueRange{mediumTensor},
        op.getIndexingMapsArray(),
        op.getIteratorTypesArray(),
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value sum = b.create<arith::AddIOp>(nestedLoc, args[0], args[1]);
          Value out = sum;
          Type outElemType = args[2].getType();
          if (!outElemType.isInteger(32)) {
            out = b.create<arith::TruncIOp>(nestedLoc, outElemType, sum);
          }
          b.create<linalg::YieldOp>(nestedLoc, out);
        });

    newOp->setAttrs(op->getAttrs());
    newOp->setAttr("npu.dma_inserted", rewriter.getUnitAttr());

    // 4) SPM -> DRAM
    Value mvoutResult = createDmaOp(rewriter, loc, newOp.getResult(0),
        "npu_dma_mvout", 0, "", oldOutput);

    rewriter.replaceOp(op, mvoutResult);
    return success();
  }
};

//=============================================================================
// Pattern: NpuUnaryInsertDmaPattern
// For unary ops (1 input, 1 output), insert mvin before and mvout after.
//=============================================================================
struct NpuGeneralInsertDmaPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    // 1. 防止重复处理
    if (op->hasAttr("npu.dma_inserted"))
      return failure();

    // 2. 检查 Target 是否为 NPU
    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    // 3. 排除已经被特定 Pattern 处理的算子和内部 DMA 算子
    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr) return failure();
    
    StringRef opName = libCallAttr.getValue();
    if (opName == "npu_dma_mvin" || opName == "npu_dma_mvout" || opName == "mv_acc_to_spm")
      return failure();

    if (opName.contains("conv") || opName.contains("gemm") ||
        opName.contains("matmul") || opName.contains("matadd"))
      return failure();

    Location loc = op.getLoc();

    // --- 步骤 1: 为输入遍历插入 MVIN ---
    SmallVector<Value> newInputs;
    for (auto [idx, input] : llvm::enumerate(op.getInputs())) {
      // 拦截 Dummy Window：免除搬运，直接分配 encoding=2 的空壳
      if (opName == "npu_maxpool" && idx == 1) {
        auto tensorType = cast<RankedTensorType>(input.getType());
        auto newType = changeEncoding(tensorType, 2, rewriter);
        Value dummySram = rewriter.create<bufferization::AllocTensorOp>(
            loc, newType, ValueRange{});
        newInputs.push_back(dummySram);
        continue;
      }

      // 常规数据：直接对接 mvin，不搞恶心的 Staging 拷贝
      Value mvinResult =
          createDmaOp(rewriter, loc, input, "npu_dma_mvin", 2, "input");
      newInputs.push_back(mvinResult);
    }

    // --- 步骤 2: 准备 Medium Tensor (分配 encoding=2 的新内存) ---
    SmallVector<Value> mediumTensors;
    for (Value originalOutput : op.getOutputs()) {
      Value mediumTensor =
          createNpuMediumTensor(rewriter, loc, originalOutput, 2);
      mediumTensors.push_back(mediumTensor);
    }

    // --- 步骤 3: 克隆计算算子 (对接 Medium Tensor) ---
    auto newOp = cast<linalg::GenericOp>(rewriter.clone(*op.getOperation()));
    newOp.getInputsMutable().assign(newInputs);
    newOp.getOutputsMutable().assign(mediumTensors); // 算子输出到 Medium Tensor

    for (auto [idx, medium] : llvm::enumerate(mediumTensors)) {
      newOp.getResult(idx).setType(medium.getType());
    }
    newOp->setAttr("npu.dma_inserted", rewriter.getUnitAttr());

    // --- 步骤 4: 为输出插入 MVOUT ---
    // 逻辑：medium tensor(new) -> [newOp] -> newOp.getResult -> mvout -> old output tensor
    SmallVector<Value> finalResults;
    for (auto [idx, mediumTensor] : llvm::enumerate(mediumTensors)) {
      Value originalOutput = op.getOutputs()[idx];
      Value computedResult = newOp.getResult(idx); // 获取算子真实的计算产物
      Value mvoutResult = createDmaOp(rewriter, loc, computedResult,
          "npu_dma_mvout", 0, "output", originalOutput);
      finalResults.push_back(mvoutResult);
    }

    // --- 步骤 5: 替换原算子 ---
    rewriter.replaceOp(op, finalResults);

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
    patterns.add<NpuMataddInsertDmaPattern>(context);
    patterns.add<NpuGeneralInsertDmaPattern>(context);
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
