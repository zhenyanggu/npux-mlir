//=============================================================================
// src/Conversion/NpuPartition/NpuInsertDma.cpp
// This file inserts explicit mvin/mvout operations
// around NPU-executable npucore operations.
//=============================================================================
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

struct FixedPointParams {
  int16_t multiplier;
  int16_t shift;
};

static FixedPointParams getFixedPointParams(double scale) {
  if (std::abs(scale) < 1e-8)
    return {0, 0};
  int exponent;
  double mantissa = std::frexp(scale, &exponent);
  double mantissa_scaled = std::round(mantissa * 32768.0);
  if (mantissa_scaled >= 32768.0) {
    mantissa_scaled /= 2.0;
    exponent += 1;
  }
  return {static_cast<int16_t>(mantissa_scaled),
          static_cast<int16_t>(exponent - 15)};
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
// Helper: Create native npucore DMA op (mvin or mvout).
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

  StringAttr dmaTypeAttr = rewriter.getStringAttr(dmaType);
  auto colDimAttr = rewriter.getI32IntegerAttr(0);
  auto isQuantAttr = rewriter.getBoolAttr(false);
  auto quantZeroAttr = rewriter.getI32IntegerAttr(0);
  auto quantScaleAttr = rewriter.getI64IntegerAttr(0);
  auto quantShiftAttr = rewriter.getI64IntegerAttr(0);

  Operation *dmaOp = nullptr;
  if (dmaName == "npu_dma_mvin") {
    dmaOp = rewriter
                .create<npucore::DmaMvinOp>(loc, TypeRange{outType},
                    ValueRange{input}, ValueRange{finalDest}, dmaTypeAttr,
                    colDimAttr, isQuantAttr, quantZeroAttr, quantScaleAttr,
                    quantShiftAttr)
                .getOperation();
  } else {
    dmaOp = rewriter
                .create<npucore::DmaMvoutOp>(loc, TypeRange{outType},
                    ValueRange{input}, ValueRange{finalDest}, dmaTypeAttr,
                    colDimAttr, isQuantAttr, quantZeroAttr, quantScaleAttr,
                    quantShiftAttr)
                .getOperation();
  }

  dmaOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
  if (!dmaType.empty())
    dmaOp->setAttr("npu.dma_type", rewriter.getStringAttr(dmaType));

  return dmaOp->getResult(0);
}

//=============================================================================
// Helper: Copy npucore.gelu attributes onto the DMA-rewritten op.
//=============================================================================
static void cloneNpucoreGeluAttrs(
    PatternRewriter &rewriter, npucore::GeluOp source, npucore::GeluOp target) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "in_scale" ||
        name == "in_zp" || name == "out_scale" || name == "out_zp")
      continue;
    target->setAttr(attr.getName(), attr.getValue());
  }
  target->setAttr("npu.dma_inserted", rewriter.getUnitAttr());
}

//=============================================================================
// Helper: Copy npucore.softmax attributes onto the DMA-rewritten op.
//=============================================================================
static void cloneNpucoreSoftmaxAttrs(PatternRewriter &rewriter,
    npucore::SoftmaxOp source, npucore::SoftmaxOp target) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "in_scale" ||
        name == "in_zp" || name == "out_scale" || name == "out_zp" ||
        name == "axis")
      continue;
    target->setAttr(attr.getName(), attr.getValue());
  }
  target->setAttr("npu.dma_inserted", rewriter.getUnitAttr());
}

//=============================================================================
// Helper: Copy npucore.layernorm attributes onto the DMA-rewritten op.
//=============================================================================
static void cloneNpucoreLayerNormAttrs(PatternRewriter &rewriter,
    npucore::LayerNormOp source, npucore::LayerNormOp target) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "in_scale" ||
        name == "in_zp" || name == "out_scale" || name == "out_zp" ||
        name == "axis" || name == "epsilon")
      continue;
    target->setAttr(attr.getName(), attr.getValue());
  }
  target->setAttr("npu.dma_inserted", rewriter.getUnitAttr());
}

//=============================================================================
// Helper: Copy npucore.matadd attributes onto the DMA-rewritten op.
//=============================================================================
static void cloneNpucoreMatAddAttrs(PatternRewriter &rewriter,
    npucore::MatAddOp source, npucore::MatAddOp target) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "in1_scale" ||
        name == "in1_zp" || name == "in2_scale" || name == "in2_zp" ||
        name == "out_scale" || name == "out_zp")
      continue;
    target->setAttr(attr.getName(), attr.getValue());
  }
  target->setAttr("npu.dma_inserted", rewriter.getUnitAttr());
}

//=============================================================================
// Helper: Copy npucore.matmul attributes onto the DMA-rewritten op.
//=============================================================================
static void cloneNpucoreMatMulAttrs(PatternRewriter &rewriter,
    npucore::MatMulOp source, npucore::MatMulOp target) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "lhs_scale" ||
        name == "lhs_zp" || name == "rhs_scale" || name == "rhs_zp" ||
        name == "out_scale" || name == "out_zp" || name == "with_bias" ||
        name == "do_relu" || name == "relu_type")
      continue;
    target->setAttr(attr.getName(), attr.getValue());
  }
  target->setAttr("npu.dma_inserted", rewriter.getUnitAttr());
}

//=============================================================================
// Helper: Copy npucore.maxpool attributes onto the DMA-rewritten op.
//=============================================================================
static void cloneNpucoreMaxPoolAttrs(PatternRewriter &rewriter,
    npucore::MaxPoolOp source, npucore::MaxPoolOp target) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "in_scale" ||
        name == "in_zp" || name == "out_scale" || name == "out_zp" ||
        name == "kernel_shape" || name == "strides" ||
        name == "dilations" || name == "pads")
      continue;
    target->setAttr(attr.getName(), attr.getValue());
  }
  target->setAttr("npu.dma_inserted", rewriter.getUnitAttr());
}

//=============================================================================
// Helper: Copy npucore.transpose attributes onto the DMA-rewritten op.
//=============================================================================
static void cloneNpucoreTransposeAttrs(PatternRewriter &rewriter,
    npucore::TransposeOp source, npucore::TransposeOp target) {
  for (NamedAttribute attr : source->getAttrs()) {
    if (attr.getName().strref() == "operandSegmentSizes")
      continue;
    target->setAttr(attr.getName(), attr.getValue());
  }
  target->setAttr("npu.dma_inserted", rewriter.getUnitAttr());
}

//=============================================================================
// Helper: Copy npucore.layout_nchw_to_nchwc32 attributes onto the
// DMA-rewritten op.
//=============================================================================
static void cloneNpucoreLayoutNchwToNchwc32Attrs(PatternRewriter &rewriter,
    npucore::LayoutNchwToNchwc32Op source,
    npucore::LayoutNchwToNchwc32Op target) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "tile_factor")
      continue;
    target->setAttr(attr.getName(), attr.getValue());
  }
  target->setAttr("npu.dma_inserted", rewriter.getUnitAttr());
}

//=============================================================================
// Helper: Copy npucore.layout_nchwc32_to_nchw attributes onto the DMA-rewritten
// op.
//=============================================================================
static void cloneNpucoreLayoutNchwc32ToNchwAttrs(PatternRewriter &rewriter,
    npucore::LayoutNchwc32ToNchwOp source,
    npucore::LayoutNchwc32ToNchwOp target) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "tile_factor")
      continue;
    target->setAttr(attr.getName(), attr.getValue());
  }
  target->setAttr("npu.dma_inserted", rewriter.getUnitAttr());
}

//=============================================================================
// Helper: Copy npucore.conv attributes onto the DMA-rewritten op.
//=============================================================================
static void cloneNpucoreConvAttrs(
    PatternRewriter &rewriter, npucore::ConvOp source, npucore::ConvOp target) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "in_scale" ||
        name == "in_zp" || name == "w_scale" || name == "w_zp" ||
        name == "out_scale" || name == "out_zp" || name == "pads" ||
        name == "strides" || name == "dilations" || name == "group" ||
        name == "do_relu" || name == "relu_type")
      continue;
    target->setAttr(attr.getName(), attr.getValue());
  }
  target->setAttr("npu.dma_inserted", rewriter.getUnitAttr());
}

//=============================================================================
// Helper: Copy npucore.mv_acc_to_spm attributes onto the DMA-rewritten op.
//=============================================================================
static void cloneNpucoreMvAccToSpmAttrs(PatternRewriter &rewriter,
    npucore::MvAccToSpmOp source, npucore::MvAccToSpmOp target) {
  for (NamedAttribute attr : source->getAttrs()) {
    if (attr.getName().strref() == "operandSegmentSizes")
      continue;
    target->setAttr(attr.getName(), attr.getValue());
  }
  target->setAttr("npu.dma_inserted", rewriter.getUnitAttr());
}

//=============================================================================
// Pattern: NpucoreGeluInsertDmaPattern
// Insert mvin before npucore.gelu and mvout after it.
//=============================================================================
struct NpucoreGeluInsertDmaPattern : public OpRewritePattern<npucore::GeluOp> {
  using OpRewritePattern<npucore::GeluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::GeluOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted"))
      return failure();
    if (!op.hasTensorSemantics())
      return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    Location loc = op.getLoc();
    Value input = op.getInputs().front();
    Value originalOutput = op.getOutputs().front();

    Value sramInput =
        createDmaOp(rewriter, loc, input, "npu_dma_mvin", 2, "input");
    Value sramOutput = createNpuMediumTensor(rewriter, loc, originalOutput, 2);

    auto newOp = rewriter.create<npucore::GeluOp>(loc,
        TypeRange{sramOutput.getType()}, ValueRange{sramInput},
        ValueRange{sramOutput}, op.getInScaleAttr(), op.getInZpAttr(),
        op.getOutScaleAttr(), op.getOutZpAttr());
    cloneNpucoreGeluAttrs(rewriter, op, newOp);

    Value finalResult = createDmaOp(rewriter, loc, newOp.getResultTensors()[0],
        "npu_dma_mvout", 0, "output", originalOutput);
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

//=============================================================================
// Pattern: NpucoreSoftmaxInsertDmaPattern
// Insert mvin before npucore.softmax and mvout after it.
//=============================================================================
struct NpucoreSoftmaxInsertDmaPattern
    : public OpRewritePattern<npucore::SoftmaxOp> {
  using OpRewritePattern<npucore::SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::SoftmaxOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted"))
      return failure();
    if (!op.hasTensorSemantics())
      return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    Location loc = op.getLoc();
    Value input = op.getInputs().front();
    Value originalOutput = op.getOutputs().front();

    Value sramInput =
        createDmaOp(rewriter, loc, input, "npu_dma_mvin", 2, "input");
    Value sramOutput = createNpuMediumTensor(rewriter, loc, originalOutput, 2);

    auto newOp = rewriter.create<npucore::SoftmaxOp>(loc,
        TypeRange{sramOutput.getType()}, ValueRange{sramInput},
        ValueRange{sramOutput}, op.getInScaleAttr(), op.getInZpAttr(),
        op.getOutScaleAttr(), op.getOutZpAttr(), op.getAxisAttr());
    cloneNpucoreSoftmaxAttrs(rewriter, op, newOp);

    Value finalResult = createDmaOp(rewriter, loc, newOp.getResultTensors()[0],
        "npu_dma_mvout", 0, "output", originalOutput);
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

//=============================================================================
// Pattern: NpucoreLayerNormInsertDmaPattern
// Insert mvin before npucore.layernorm and mvout after it.
//=============================================================================
struct NpucoreLayerNormInsertDmaPattern
    : public OpRewritePattern<npucore::LayerNormOp> {
  using OpRewritePattern<npucore::LayerNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::LayerNormOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted"))
      return failure();
    if (!op.hasTensorSemantics())
      return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    Location loc = op.getLoc();
    Value input = op.getInputs().front();
    Value originalOutput = op.getOutputs().front();

    Value sramInput =
        createDmaOp(rewriter, loc, input, "npu_dma_mvin", 2, "input");
    Value sramOutput = createNpuMediumTensor(rewriter, loc, originalOutput, 2);

    auto newOp = rewriter.create<npucore::LayerNormOp>(loc,
        TypeRange{sramOutput.getType()}, ValueRange{sramInput},
        ValueRange{sramOutput}, op.getInScaleAttr(), op.getInZpAttr(),
        op.getOutScaleAttr(), op.getOutZpAttr(), op.getAxisAttr(),
        op.getEpsilonAttr());
    cloneNpucoreLayerNormAttrs(rewriter, op, newOp);

    Value finalResult = createDmaOp(rewriter, loc, newOp.getResultTensors()[0],
        "npu_dma_mvout", 0, "output", originalOutput);
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

//=============================================================================
// Pattern: NpucoreMatAddInsertDmaPattern
// Insert ACC mvin before npucore.matadd and mvout after it.
//=============================================================================
struct NpucoreMatAddInsertDmaPattern
    : public OpRewritePattern<npucore::MatAddOp> {
  using OpRewritePattern<npucore::MatAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::MatAddOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted"))
      return failure();
    if (!op.hasTensorSemantics())
      return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    Location loc = op.getLoc();
    Type i32Type = rewriter.getI32Type();

    double in1Scale = op.getIn1Scale().convertToDouble();
    double in2Scale = op.getIn2Scale().convertToDouble();
    double outScale = op.getOutScale().convertToDouble();
    bool scalesSame = std::abs(in1Scale - in2Scale) < 1e-6;

    SmallVector<Value> newInputs;
    for (auto it : llvm::enumerate(op.getInputs())) {
      int64_t operandIdx = it.index();
      Value operand = it.value();
      auto inputType = cast<RankedTensorType>(operand.getType());
      auto mvinOutType = RankedTensorType::get(
          inputType.getShape(), i32Type, rewriter.getI64IntegerAttr(3));

      Value finalDest = rewriter.create<bufferization::AllocTensorOp>(
          loc, mvinOutType, ValueRange{});

      auto mvinOp = rewriter.create<npucore::DmaMvinOp>(loc,
          TypeRange{mvinOutType}, ValueRange{operand}, ValueRange{finalDest},
          rewriter.getStringAttr(""), rewriter.getI32IntegerAttr(0),
          rewriter.getBoolAttr(false), rewriter.getI32IntegerAttr(0),
          rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(0));
      mvinOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

      int64_t zpVal = operandIdx == 0 ? op.getIn1Zp() : op.getIn2Zp();
      double currentInScale = operandIdx == 0 ? in1Scale : in2Scale;
      bool isQuant = false;
      double mvinScaleVal = 1.0;

      if (!scalesSame) {
        mvinScaleVal = currentInScale / outScale;
        isQuant = true;
      } else if (zpVal != 0) {
        isQuant = true;
      }

      if (isQuant) {
        FixedPointParams qParams = getFixedPointParams(mvinScaleVal);
        mvinOp->setAttr(
            "quant_scale", rewriter.getI64IntegerAttr(qParams.multiplier));
        mvinOp->setAttr(
            "quant_shift", rewriter.getI64IntegerAttr(qParams.shift));
        mvinOp->setAttr("quant_zero", rewriter.getI32IntegerAttr(zpVal));
        mvinOp->setAttr("is_quant", rewriter.getBoolAttr(true));
      }

      newInputs.push_back(mvinOp.getResultTensors().front());
    }

    auto originalOutputType =
        cast<RankedTensorType>(op.getOutputs().front().getType());
    auto sramOutputType = RankedTensorType::get(originalOutputType.getShape(),
        originalOutputType.getElementType(), rewriter.getI64IntegerAttr(2));
    Value sramOutput = rewriter.create<bufferization::AllocTensorOp>(
        loc, sramOutputType, ValueRange{});

    auto newOp = rewriter.create<npucore::MatAddOp>(loc,
        TypeRange{sramOutputType}, newInputs, ValueRange{sramOutput},
        op.getIn1ScaleAttr(), op.getIn1ZpAttr(), op.getIn2ScaleAttr(),
        op.getIn2ZpAttr(), op.getOutScaleAttr(), op.getOutZpAttr());

    if (!scalesSame) {
      newOp->setAttr("in1_scale", rewriter.getF32FloatAttr(1.0f));
      newOp->setAttr("in2_scale", rewriter.getF32FloatAttr(1.0f));
      newOp->setAttr("out_scale", rewriter.getF32FloatAttr(1.0f));
    }
    cloneNpucoreMatAddAttrs(rewriter, op, newOp);

    Value finalResult = createDmaOp(rewriter, loc, newOp.getResultTensors()[0],
        "npu_dma_mvout", 0, "output", op.getOutputs().front());
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

//=============================================================================
// Pattern: NpucoreMatMulInsertDmaPattern
// Insert mvin before npucore.matmul and mvout after npucore.mv_acc_to_spm.
//=============================================================================
struct NpucoreMatMulInsertDmaPattern
    : public OpRewritePattern<npucore::MatMulOp> {
  using OpRewritePattern<npucore::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::MatMulOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted"))
      return failure();
    if (!op.hasTensorSemantics())
      return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    Location loc = op.getLoc();
    StringRef stage = "single";
    if (auto stageAttr = op->getAttrOfType<StringAttr>("npu.loop_stage"))
      stage = stageAttr.getValue();

    SmallVector<Value> newInputs;
    for (auto it : llvm::enumerate(op.getInputs())) {
      int64_t operandIdx = it.index();
      Value operand = it.value();
      if (operandIdx < 2) {
        newInputs.push_back(
            createDmaOp(rewriter, loc, operand, "npu_dma_mvin", 2, "input"));
      } else {
        newInputs.push_back(operand);
      }
    }

    Value originalOutput = op.getOutputs().front();
    auto newOp = rewriter.create<npucore::MatMulOp>(loc,
        TypeRange{originalOutput.getType()}, newInputs, ValueRange{originalOutput},
        op.getLhsScaleAttr(), op.getLhsZpAttr(), op.getRhsScaleAttr(),
        op.getRhsZpAttr(), op.getOutScaleAttr(), op.getOutZpAttr(),
        op.getWithBiasAttr(), op.getDoReluAttr(), op.getReluTypeAttr());
    cloneNpucoreMatMulAttrs(rewriter, op, newOp);

    Value newMatmulResult = newOp.getResultTensors().front();
    rewriter.replaceOp(op, newMatmulResult);

    if (stage != "tail" && stage != "single")
      return success();

    for (Operation *user :
        llvm::make_early_inc_range(newMatmulResult.getUsers())) {
      auto moveOp = dyn_cast<npucore::MvAccToSpmOp>(user);
      if (!moveOp || !moveOp.hasTensorSemantics())
        continue;

      rewriter.setInsertionPoint(moveOp);
      Value moveOutput = moveOp.getOutputs().front();
      Value mediumTensor =
          createNpuMediumTensor(rewriter, moveOp.getLoc(), moveOutput, 2);

      auto newMoveOp = rewriter.create<npucore::MvAccToSpmOp>(moveOp.getLoc(),
          TypeRange{mediumTensor.getType()}, ValueRange{newMatmulResult},
          ValueRange{mediumTensor});
      cloneNpucoreMvAccToSpmAttrs(rewriter, moveOp, newMoveOp);

      Value finalResult = createDmaOp(rewriter, moveOp.getLoc(),
          newMoveOp.getResultTensors().front(), "npu_dma_mvout", 0, "",
          moveOutput);
      rewriter.replaceOp(moveOp, finalResult);
      break;
    }

    return success();
  }
};

//=============================================================================
// Pattern: NpucoreMaxPoolInsertDmaPattern
// Insert mvin before npucore.maxpool and mvout after it.
//=============================================================================
struct NpucoreMaxPoolInsertDmaPattern
    : public OpRewritePattern<npucore::MaxPoolOp> {
  using OpRewritePattern<npucore::MaxPoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::MaxPoolOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted"))
      return failure();
    if (!op.hasTensorSemantics())
      return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    Location loc = op.getLoc();
    Value input = op.getInputs().front();
    Value originalOutput = op.getOutputs().front();

    Value sramInput =
        createDmaOp(rewriter, loc, input, "npu_dma_mvin", 2, "input");
    Value sramOutput = createNpuMediumTensor(rewriter, loc, originalOutput, 2);

    auto newOp = rewriter.create<npucore::MaxPoolOp>(loc,
        TypeRange{sramOutput.getType()}, ValueRange{sramInput},
        ValueRange{sramOutput}, op.getInScaleAttr(), op.getInZpAttr(),
        op.getOutScaleAttr(), op.getOutZpAttr(), op.getKernelShapeAttr(),
        op.getStridesAttr(), op.getDilationsAttr(), op.getPadsAttr());
    cloneNpucoreMaxPoolAttrs(rewriter, op, newOp);

    Value finalResult = createDmaOp(rewriter, loc, newOp.getResultTensors()[0],
        "npu_dma_mvout", 0, "output", originalOutput);
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

//=============================================================================
// Pattern: NpucoreTransposeInsertDmaPattern
// Insert mvin before npucore.transpose and mvout after it.
//=============================================================================
struct NpucoreTransposeInsertDmaPattern
    : public OpRewritePattern<npucore::TransposeOp> {
  using OpRewritePattern<npucore::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::TransposeOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted"))
      return failure();
    if (!op.hasTensorSemantics())
      return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    Location loc = op.getLoc();
    Value input = op.getInputs().front();
    Value originalOutput = op.getOutputs().front();

    Value sramInput =
        createDmaOp(rewriter, loc, input, "npu_dma_mvin", 2, "input");
    Value sramOutput = createNpuMediumTensor(rewriter, loc, originalOutput, 2);

    auto newOp = rewriter.create<npucore::TransposeOp>(loc,
        TypeRange{sramOutput.getType()}, ValueRange{sramInput},
        ValueRange{sramOutput});
    cloneNpucoreTransposeAttrs(rewriter, op, newOp);

    Value finalResult = createDmaOp(rewriter, loc, newOp.getResultTensors()[0],
        "npu_dma_mvout", 0, "output", originalOutput);
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

//=============================================================================
// Pattern: NpucoreLayoutNchwToNchwc32InsertDmaPattern
// Insert mvin before npucore.layout_nchw_to_nchwc32 and mvout after it.
//=============================================================================
struct NpucoreLayoutNchwToNchwc32InsertDmaPattern
    : public OpRewritePattern<npucore::LayoutNchwToNchwc32Op> {
  using OpRewritePattern<npucore::LayoutNchwToNchwc32Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(npucore::LayoutNchwToNchwc32Op op,
      PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted"))
      return failure();
    if (!op.hasTensorSemantics())
      return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    Location loc = op.getLoc();
    Value input = op.getInputs().front();
    Value originalOutput = op.getOutputs().front();

    Value sramInput =
        createDmaOp(rewriter, loc, input, "npu_dma_mvin", 2, "input");
    Value sramOutput = createNpuMediumTensor(rewriter, loc, originalOutput, 2);

    auto newOp = rewriter.create<npucore::LayoutNchwToNchwc32Op>(loc,
        TypeRange{sramOutput.getType()}, ValueRange{sramInput},
        ValueRange{sramOutput}, op.getTileFactorAttr());
    cloneNpucoreLayoutNchwToNchwc32Attrs(rewriter, op, newOp);

    Value finalResult = createDmaOp(rewriter, loc, newOp.getResultTensors()[0],
        "npu_dma_mvout", 0, "output", originalOutput);
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

//=============================================================================
// Pattern: NpucoreLayoutNchwc32ToNchwInsertDmaPattern
// Insert mvin before npucore.layout_nchwc32_to_nchw and mvout after it.
//=============================================================================
struct NpucoreLayoutNchwc32ToNchwInsertDmaPattern
    : public OpRewritePattern<npucore::LayoutNchwc32ToNchwOp> {
  using OpRewritePattern<npucore::LayoutNchwc32ToNchwOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(npucore::LayoutNchwc32ToNchwOp op,
      PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted"))
      return failure();
    if (!op.hasTensorSemantics())
      return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    Location loc = op.getLoc();
    Value input = op.getInputs().front();
    Value originalOutput = op.getOutputs().front();

    Value sramInput =
        createDmaOp(rewriter, loc, input, "npu_dma_mvin", 2, "input");
    Value sramOutput = createNpuMediumTensor(rewriter, loc, originalOutput, 2);

    auto newOp = rewriter.create<npucore::LayoutNchwc32ToNchwOp>(loc,
        TypeRange{sramOutput.getType()}, ValueRange{sramInput},
        ValueRange{sramOutput}, op.getTileFactorAttr());
    cloneNpucoreLayoutNchwc32ToNchwAttrs(rewriter, op, newOp);

    Value finalResult = createDmaOp(rewriter, loc, newOp.getResultTensors()[0],
        "npu_dma_mvout", 0, "output", originalOutput);
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

//=============================================================================
// Pattern: NpucoreConvInsertDmaPattern
// Insert mvin before npucore.conv and mvout after npucore.mv_acc_to_spm.
//=============================================================================
struct NpucoreConvInsertDmaPattern : public OpRewritePattern<npucore::ConvOp> {
  using OpRewritePattern<npucore::ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::ConvOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.dma_inserted"))
      return failure();
    if (!op.hasTensorSemantics())
      return failure();

    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu")
      return failure();

    Location loc = op.getLoc();
    StringRef stage = "single";
    if (auto stageAttr = op->getAttrOfType<StringAttr>("npu.loop_stage"))
      stage = stageAttr.getValue();

    SmallVector<Value> newInputs;
    for (auto it : llvm::enumerate(op.getInputs())) {
      int64_t operandIdx = it.index();
      Value operand = it.value();
      if (operandIdx == 0) {
        newInputs.push_back(
            createDmaOp(rewriter, loc, operand, "npu_dma_mvin", 2, "input"));
      } else if (operandIdx == 1) {
        newInputs.push_back(
            createDmaOp(rewriter, loc, operand, "npu_dma_mvin", 2, "weight"));
      } else {
        newInputs.push_back(operand);
      }
    }

    Value originalOutput = op.getOutputs().front();
    auto newOp = rewriter.create<npucore::ConvOp>(loc,
        TypeRange{originalOutput.getType()}, newInputs, ValueRange{originalOutput},
        op.getInScaleAttr(), op.getInZpAttr(), op.getWScaleAttr(),
        op.getWZpAttr(), op.getOutScaleAttr(), op.getOutZpAttr(),
        op.getPadsAttr(), op.getStridesAttr(), op.getDilationsAttr(),
        op.getGroupAttr(), op.getDoReluAttr(), op.getReluTypeAttr());
    cloneNpucoreConvAttrs(rewriter, op, newOp);

    Value newConvResult = newOp.getResultTensors().front();
    rewriter.replaceOp(op, newConvResult);

    if (stage != "tail" && stage != "single")
      return success();

    for (Operation *user : llvm::make_early_inc_range(newConvResult.getUsers())) {
      auto moveOp = dyn_cast<npucore::MvAccToSpmOp>(user);
      if (!moveOp || !moveOp.hasTensorSemantics())
        continue;

      rewriter.setInsertionPoint(moveOp);
      Value moveOutput = moveOp.getOutputs().front();
      Value mediumTensor = createNpuMediumTensor(rewriter, moveOp.getLoc(), moveOutput, 2);

      auto newMoveOp = rewriter.create<npucore::MvAccToSpmOp>(moveOp.getLoc(),
          TypeRange{mediumTensor.getType()}, ValueRange{newConvResult},
          ValueRange{mediumTensor});
      cloneNpucoreMvAccToSpmAttrs(rewriter, moveOp, newMoveOp);

      Value finalResult = createDmaOp(rewriter, moveOp.getLoc(),
          newMoveOp.getResultTensors().front(), "npu_dma_mvout", 0, "",
          moveOutput);
      rewriter.replaceOp(moveOp, finalResult);
      break;
    }

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
    return "Insert explicit mvin/mvout npucore ops around NPU compute ops.";
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<NpucoreGeluInsertDmaPattern>(context);
    patterns.add<NpucoreSoftmaxInsertDmaPattern>(context);
    patterns.add<NpucoreLayerNormInsertDmaPattern>(context);
    patterns.add<NpucoreMatAddInsertDmaPattern>(context);
    patterns.add<NpucoreMatMulInsertDmaPattern>(context);
    patterns.add<NpucoreMaxPoolInsertDmaPattern>(context);
    patterns.add<NpucoreTransposeInsertDmaPattern>(context);
    patterns.add<NpucoreLayoutNchwToNchwc32InsertDmaPattern>(context);
    patterns.add<NpucoreLayoutNchwc32ToNchwInsertDmaPattern>(context);
    patterns.add<NpucoreConvInsertDmaPattern>(context);

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
