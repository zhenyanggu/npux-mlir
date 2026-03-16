//=============================================================================
// src/Conversion/NpuPartition/NpuInsertDma.cpp
// This file inserts explicit mvin/mvout operations
// around NPU-executable linalg.generic operations.
//=============================================================================
#include "mlir/Dialect/Arith/IR/Arith.h"
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
// Pattern: NpuConvInsertDmaPattern
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

    SmallVector<Value> newOutputs;
    newOutputs.push_back(op.getOutputs()[0]);

    auto newOp = cast<linalg::GenericOp>(rewriter.clone(*op.getOperation()));
    newOp.getInputsMutable().assign(newInputs);
    newOp.getOutputsMutable().assign(newOutputs);
    newOp->setAttr("npu.dma_inserted", rewriter.getUnitAttr());

    Value newConvResult = newOp.getResult(0);
    rewriter.replaceOp(op, newConvResult);

    if (stage == "tail" || stage == "single") {
      for (Operation *user : newConvResult.getUsers()) {
        auto genericUser = dyn_cast<linalg::GenericOp>(user);
        if (!genericUser) continue;

        auto libCallAttr = genericUser->getAttrOfType<StringAttr>("library_call");
        if (libCallAttr && libCallAttr.getValue() == "mv_acc_to_spm") {
          rewriter.setInsertionPoint(genericUser);
          Location uLoc = genericUser.getLoc();
          Value oldOutput = genericUser.getOutputs()[0];
          Value mediumTensor = createNpuMediumTensor(rewriter, uLoc, oldOutput, 2);

          auto newMvAcc = cast<linalg::GenericOp>(
              rewriter.clone(*genericUser.getOperation()));
          newMvAcc.getInputsMutable().assign(newConvResult);
          newMvAcc.getOutputsMutable().assign(mediumTensor);
          newMvAcc.getResult(0).setType(mediumTensor.getType());

          Value mvoutResult = createDmaOp(rewriter, uLoc, newMvAcc.getResult(0),
              "npu_dma_mvout", 0, "", oldOutput);
          rewriter.replaceOp(genericUser, mvoutResult);
          break;
        }
      }
    }
    return success();
  }
};

//=============================================================================
// Pattern: NpuGemmInsertDmaPattern
//=============================================================================
struct NpuGemmInsertDmaPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted")) return failure();

    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr) return failure();
    StringRef opName = libCallAttr.getValue();
    if (opName != "npu_gemm" && opName != "npu_matmul") return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu") return failure();

    Location loc = op.getLoc();
    StringRef stage = "single";
    if (auto stageAttr = op->getAttrOfType<StringAttr>("npu.loop_stage")) {
      stage = stageAttr.getValue();
    }

    SmallVector<Value> newInputs;
    for (auto it : llvm::enumerate(op.getInputs())) {
      size_t index = it.index();
      Value operand = it.value();
      if (index < 2) {
        Value processedInput =
            createDmaOp(rewriter, loc, operand, "npu_dma_mvin", 2, "");
        newInputs.push_back(processedInput);
      } else {
        newInputs.push_back(operand);
      }
    }

    SmallVector<Value> newOutputs;
    newOutputs.push_back(op.getOutputs()[0]);

    auto newOp = cast<linalg::GenericOp>(rewriter.clone(*op.getOperation()));
    newOp.getInputsMutable().assign(newInputs);
    newOp.getOutputsMutable().assign(newOutputs);
    newOp->setAttr("npu.dma_inserted", rewriter.getUnitAttr());

    Value newGemmResult = newOp.getResult(0);
    rewriter.replaceOp(op, newGemmResult);

    if (stage == "tail" || stage == "single") {
      for (Operation *user : newGemmResult.getUsers()) {
        auto genericUser = dyn_cast<linalg::GenericOp>(user);
        if (!genericUser) continue;

        auto userLibCallAttr = genericUser->getAttrOfType<StringAttr>("library_call");
        if (userLibCallAttr && userLibCallAttr.getValue() == "mv_acc_to_spm") {
          rewriter.setInsertionPoint(genericUser);
          Location uLoc = genericUser.getLoc();
          Value oldOutput = genericUser.getOutputs()[0];
          Value mediumTensor = createNpuMediumTensor(rewriter, uLoc, oldOutput, 2);

          auto newMvAcc = cast<linalg::GenericOp>(
              rewriter.clone(*genericUser.getOperation()));
          newMvAcc.getInputsMutable().assign(newGemmResult);
          newMvAcc.getOutputsMutable().assign(mediumTensor);
          newMvAcc.getResult(0).setType(mediumTensor.getType());

          Value mvoutResult = createDmaOp(rewriter, uLoc, newMvAcc.getResult(0),
              "npu_dma_mvout", 0, "", oldOutput);
          rewriter.replaceOp(genericUser, mvoutResult);
          break;
        }
      }
    }
    return success();
  }
};

//=============================================================================
// Pattern: NpuMataddInsertDmaPattern
// 专门处理 npu_matadd:
// 1. 输入通过 mvin 时提升至 i32 精度，且 encoding=3
// 2. 输出使用 encoding=2，保持原精度
//=============================================================================
struct NpuMataddInsertDmaPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted")) return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu") return failure();

    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr || libCallAttr.getValue() != "npu_matadd") return failure();

    Location loc = op.getLoc();
    Type i32Type = rewriter.getI32Type();

    // --- 步骤 1: 处理输入，插入提升至 i32 且 encoding=3 的 MVIN ---
    SmallVector<Value> newInputs;
    for (Value operand : op.getInputs()) {
      auto inputType = cast<RankedTensorType>(operand.getType());
      
      // 新建带有 i32 且 encoding=3 的输出 TensorType
      auto mvinOutType = RankedTensorType::get(
          inputType.getShape(), i32Type, rewriter.getI64IntegerAttr(3));
          
      Value finalDest = rewriter.create<bufferization::AllocTensorOp>(
          loc, mvinOutType, ValueRange{});

      SmallVector<utils::IteratorType> iteratorTypes(
          inputType.getRank(), utils::IteratorType::parallel);
      SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(inputType.getRank()));

      auto mvinOp = rewriter.create<linalg::GenericOp>(loc, mvinOutType,
          /*inputs=*/ValueRange{operand},
          /*outputs=*/ValueRange{finalDest},
          /*indexingMaps=*/maps,
          /*iteratorTypes=*/iteratorTypes,
          /*bodyBuilder=*/
          [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value val = args[0];
            // 将较低精度的 Int 提升至 i32 (这里默认是有符号提升，适配大多数量化场景)
            if (val.getType() != i32Type && isa<IntegerType>(val.getType())) {
              val = b.create<arith::ExtSIOp>(nestedLoc, i32Type, val);
            }
            b.create<linalg::YieldOp>(nestedLoc, val);
          });

      mvinOp->setAttr("library_call", rewriter.getStringAttr("npu_dma_mvin"));
      mvinOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
      
      newInputs.push_back(mvinOp.getResult(0));
    }

    // --- 步骤 2: 准备 Medium Tensor (分配 encoding=2，精度保持原样) ---
    SmallVector<Value> mediumTensors;
    for (Value originalOutput : op.getOutputs()) {
      Value mediumTensor = createNpuMediumTensor(rewriter, loc, originalOutput, 2);
      mediumTensors.push_back(mediumTensor);
    }

    // --- 步骤 3: 克隆 Matadd 算子并对接新输入/输出 ---
    int64_t rank = cast<RankedTensorType>(newInputs[0].getType()).getRank();
    SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    auto newOp = rewriter.create<linalg::GenericOp>(loc,
        TypeRange{mediumTensors[0].getType()},
        newInputs,        // 现在都是 i32, encoding=3
        mediumTensors,    // 输出目标，如 i8, encoding=2
        maps,
        iteratorTypes,
        /*bodyBuilder=*/
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            // args[0] (LHS) 和 args[1] (RHS) 已经是 i32 
            Value lhs = args[0];
            Value rhs = args[1];
            
            // 1. 在 i32 精度下做加法以保证 IR 合法
            Value addRes = b.create<arith::AddIOp>(nestedLoc, lhs, rhs);
            
            // 2. 将 i32 结果截断 (Truncate) 回原输出精度 (例如 i8)
            Type outType = args[2].getType();
            Value finalRes = addRes;
            if (addRes.getType() != outType && isa<IntegerType>(outType)) {
                finalRes = b.create<arith::TruncIOp>(nestedLoc, outType, addRes);
            }
            
            b.create<linalg::YieldOp>(nestedLoc, finalRes);
        });

    // 将原算子身上的所有 Attribute (scale, zp, library_call 等) 原封不动抄过来
    for (NamedAttribute attr : op->getAttrs()) {
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    // 打上防重复标记
    newOp->setAttr("npu.dma_inserted", rewriter.getUnitAttr());

    // --- 步骤 4: 插入 MVOUT ---
    SmallVector<Value> finalResults;
    for (auto [idx, mediumTensor] : llvm::enumerate(mediumTensors)) {
      Value originalOutput = op.getOutputs()[idx];
      Value computedResult = newOp.getResult(idx); 
      // mvout 输入是 encoding 2，输出被映射回 0 
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
// Pattern: NpuGeneralInsertDmaPattern
// For unary ops (1 input, 1 output), insert mvin before and mvout after.
//=============================================================================
struct NpuGeneralInsertDmaPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted")) return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu") return failure();

    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr) return failure();
    
    StringRef opName = libCallAttr.getValue();
    if (opName == "npu_dma_mvin" || opName == "npu_dma_mvout" || opName == "mv_acc_to_spm")
      return failure();

    // 排除被专用 Pattern 处理的算子，包含 npu_matadd
    if (opName.contains("conv") || opName.contains("gemm") || 
        opName.contains("matmul") || opName == "npu_matadd")
      return failure();

    Location loc = op.getLoc();

    SmallVector<Value> newInputs;
    for (auto [idx, input] : llvm::enumerate(op.getInputs())) {
      if (opName == "npu_maxpool" && idx == 1) {
        auto tensorType = cast<RankedTensorType>(input.getType());
        auto newType = changeEncoding(tensorType, 2, rewriter);
        Value dummySram = rewriter.create<bufferization::AllocTensorOp>(
            loc, newType, ValueRange{});
        newInputs.push_back(dummySram);
        continue;
      }
      Value mvinResult =
          createDmaOp(rewriter, loc, input, "npu_dma_mvin", 2, "input");
      newInputs.push_back(mvinResult);
    }

    SmallVector<Value> mediumTensors;
    for (Value originalOutput : op.getOutputs()) {
      Value mediumTensor =
          createNpuMediumTensor(rewriter, loc, originalOutput, 2);
      mediumTensors.push_back(mediumTensor);
    }

    auto newOp = cast<linalg::GenericOp>(rewriter.clone(*op.getOperation()));
    newOp.getInputsMutable().assign(newInputs);
    newOp.getOutputsMutable().assign(mediumTensors);

    for (auto [idx, medium] : llvm::enumerate(mediumTensors)) {
      newOp.getResult(idx).setType(medium.getType());
    }
    newOp->setAttr("npu.dma_inserted", rewriter.getUnitAttr());

    SmallVector<Value> finalResults;
    for (auto [idx, mediumTensor] : llvm::enumerate(mediumTensors)) {
      Value originalOutput = op.getOutputs()[idx];
      Value computedResult = newOp.getResult(idx);
      Value mvoutResult = createDmaOp(rewriter, loc, computedResult,
          "npu_dma_mvout", 0, "output", originalOutput);
      finalResults.push_back(mvoutResult);
    }

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

    // 加入新增的 Matadd 专用 Pattern
    patterns.add<NpuConvInsertDmaPattern>(context);
    patterns.add<NpuGemmInsertDmaPattern>(context);
    patterns.add<NpuMataddInsertDmaPattern>(context);
    patterns.add<NpuGeneralInsertDmaPattern>(context);

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