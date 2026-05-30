//==============================================================//
// src/Conversion/NpuPartition/NpuCore/MatMul.cpp
// this file implements the conversion of Gemm/MatMul to npucore
// matmul/transpose for NPU partitioning.
//==============================================================//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/NpuPartition/NpuCoreConversionHelper.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"

using namespace mlir;

namespace {

static float getScalarFloat(Value v, float defaultVal = 1.0f) {
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue()))
      return static_cast<float>(floatAttr.getValueAsDouble());
  }
  if (auto constOp = v.getDefiningOp<ONNXConstantOp>()) {
    if (auto dense = dyn_cast<DenseElementsAttr>(constOp.getValueAttr()))
      return dense.getValues<float>()[0];
  }
  return defaultVal;
}

static int64_t getScalarInt(Value v, int64_t defaultVal = 0) {
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  }
  if (auto constOp = v.getDefiningOp<ONNXConstantOp>()) {
    if (auto dense = dyn_cast<DenseElementsAttr>(constOp.getValueAttr())) {
      if (dense.getElementType().isInteger(8))
        return dense.getValues<int8_t>()[0];
      if (dense.getElementType().isInteger(32))
        return dense.getValues<int32_t>()[0];
      if (dense.getElementType().isInteger(64))
        return dense.getValues<int64_t>()[0];
    }
  }
  return defaultVal;
}

static Value materializeToEncoding1(
    OpBuilder &builder, Location loc, Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return value;

  auto encodedType = npux::addEncoding1(type, builder);
  if (value.getDefiningOp<ONNXConstantOp>() ||
      value.getDefiningOp<arith::ConstantOp>()) {
    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < type.getRank(); ++i) {
      if (type.isDynamicDim(i))
        dynamicSizes.push_back(builder.create<tensor::DimOp>(loc, value, i));
    }
    Value alloc = builder.create<bufferization::AllocTensorOp>(
        loc, encodedType, dynamicSizes);
    return builder
        .create<bufferization::MaterializeInDestinationOp>(loc, value, alloc)
        .getResult();
  }

  value.setType(encodedType);
  return value;
}

static SmallVector<Value> getMatMulDynamicSizes(OpBuilder &builder, Location loc,
    Value lhs, Value rhs, ArrayRef<int64_t> outputShape) {
  SmallVector<Value> dynamicSizes;
  if (outputShape.empty())
    return dynamicSizes;

  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  int64_t outRank = outputShape.size();

  for (int64_t i = 0; i < outRank; ++i) {
    if (!ShapedType::isDynamic(outputShape[i]))
      continue;

    if (outRank == 2) {
      if (i == 0) {
        dynamicSizes.push_back(
            builder.create<tensor::DimOp>(loc, lhs, lhsType.getRank() - 2));
      } else {
        dynamicSizes.push_back(
            builder.create<tensor::DimOp>(loc, rhs, rhsType.getRank() - 1));
      }
      continue;
    }

    if (i == 0) {
      Value batchRef = lhsType.getRank() == 3 ? lhs : rhs;
      dynamicSizes.push_back(builder.create<tensor::DimOp>(loc, batchRef, 0));
    } else if (i == 1) {
      dynamicSizes.push_back(
          builder.create<tensor::DimOp>(loc, lhs, lhsType.getRank() - 2));
    } else {
      dynamicSizes.push_back(
          builder.create<tensor::DimOp>(loc, rhs, rhsType.getRank() - 1));
    }
  }
  return dynamicSizes;
}

static Value buildNpucoreTranspose(
    ConversionPatternRewriter &rewriter, Location loc, Value input) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || inputType.getRank() < 2)
    return input;

  SmallVector<int64_t> outputShape(inputType.getShape());
  std::swap(
      outputShape[inputType.getRank() - 1], outputShape[inputType.getRank() - 2]);
  auto outputType = RankedTensorType::get(
      outputShape, inputType.getElementType(), rewriter.getI64IntegerAttr(1));

  SmallVector<Value> dynamicSizes;
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    int64_t sourceDim = i;
    if (i == inputType.getRank() - 2)
      sourceDim = inputType.getRank() - 1;
    else if (i == inputType.getRank() - 1)
      sourceDim = inputType.getRank() - 2;
    if (outputShape[i] == ShapedType::kDynamic)
      dynamicSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, input, sourceDim));
  }

  Value init = rewriter.create<bufferization::AllocTensorOp>(
      loc, outputType, dynamicSizes);
  auto transposeOp = rewriter.create<npucore::TransposeOp>(
      loc, TypeRange{outputType}, ValueRange{input}, ValueRange{init});
  transposeOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
  return transposeOp.getResultTensors().front();
}

static Value createNpucoreMatMulOp(ConversionPatternRewriter &rewriter,
    Location loc, SmallVector<Value> inputs, RankedTensorType outputType,
    float lhsScale, int64_t lhsZp, float rhsScale, int64_t rhsZp,
    float outScale, int64_t outZp, int64_t doRelu, int64_t reluType,
    bool transA, bool transB) {
  SmallVector<Value> processedInputs;
  for (Value input : inputs)
    processedInputs.push_back(materializeToEncoding1(rewriter, loc, input));

  Value lhs = processedInputs[0];
  Value rhs = processedInputs[1];
  if (transA)
    lhs = buildNpucoreTranspose(rewriter, loc, lhs);
  if (transB)
    rhs = buildNpucoreTranspose(rewriter, loc, rhs);

  SmallVector<Value> dynamicSizes =
      getMatMulDynamicSizes(rewriter, loc, lhs, rhs, outputType.getShape());

  auto i32OutputType = RankedTensorType::get(
      outputType.getShape(), rewriter.getI32Type(), rewriter.getI64IntegerAttr(1));
  Value accInit = rewriter.create<bufferization::AllocTensorOp>(
      loc, i32OutputType, dynamicSizes);

  SmallVector<Value> matmulInputs = {lhs, rhs};
  bool hasBias = processedInputs.size() == 3;
  if (hasBias)
    matmulInputs.push_back(processedInputs[2]);

  auto matmulOp = rewriter.create<npucore::MatMulOp>(loc, TypeRange{i32OutputType},
      matmulInputs, ValueRange{accInit}, rewriter.getF32FloatAttr(lhsScale),
      rewriter.getI32IntegerAttr(lhsZp), rewriter.getF32FloatAttr(rhsScale),
      rewriter.getI32IntegerAttr(rhsZp), rewriter.getF32FloatAttr(outScale),
      rewriter.getI32IntegerAttr(outZp), rewriter.getI32IntegerAttr(hasBias ? 1 : 0),
      rewriter.getI32IntegerAttr(doRelu), rewriter.getI32IntegerAttr(reluType));
  matmulOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

  auto encodedOutputType = npux::addEncoding1(outputType, rewriter);
  Value outputInit = rewriter.create<bufferization::AllocTensorOp>(
      loc, encodedOutputType, dynamicSizes);
  auto moveOp = rewriter.create<npucore::MvAccToSpmOp>(loc,
      TypeRange{encodedOutputType}, ValueRange{matmulOp.getResultTensors().front()},
      ValueRange{outputInit});
  moveOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
  return moveOp.getResultTensors().front();
}

struct GemmToNpucore final : public OpConversionPattern<ONNXGemmOp> {
  using OpConversionPattern<ONNXGemmOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGemmOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto dequantA = op.getA().getDefiningOp<ONNXDequantizeLinearOp>();
    auto dequantB = op.getB().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantA || !dequantB)
      return failure();

    if (!op.getResult().hasOneUse())
      return failure();
    auto quantOp =
        dyn_cast<ONNXQuantizeLinearOp>(*op.getResult().getUsers().begin());
    if (!quantOp)
      return failure();

    int64_t doRelu = 0;
    int64_t reluType = 0;
    ONNXQuantizeLinearOp finalQuantOp = quantOp;
    if (quantOp.getResult().hasOneUse()) {
      if (auto dqOp = dyn_cast<ONNXDequantizeLinearOp>(
              *quantOp.getResult().getUsers().begin())) {
        if (dqOp.getResult().hasOneUse()) {
          Operation *actOp = *dqOp.getResult().getUsers().begin();
          bool isRelu = isa<ONNXReluOp>(actOp);
          bool isLeaky = isa<ONNXLeakyReluOp>(actOp);
          if ((isRelu || isLeaky) && actOp->getResult(0).hasOneUse()) {
            if (auto qOp = dyn_cast<ONNXQuantizeLinearOp>(
                    *actOp->getResult(0).getUsers().begin())) {
              bool canFuse = true;
              if (isLeaky) {
                auto leakyOp = cast<ONNXLeakyReluOp>(actOp);
                float alpha = 0.01f;
                if (auto alphaAttr = leakyOp.getAlphaAttr())
                  alpha = alphaAttr.getValueAsDouble();
                if (std::abs(alpha - 0.1f) < 1e-5)
                  reluType = 2;
                else if (std::abs(alpha - 0.2f) < 1e-5)
                  reluType = 3;
                else if (std::abs(alpha - 0.01f) < 1e-5)
                  reluType = 4;
                else
                  canFuse = false;
              }
              if (canFuse) {
                doRelu = 1;
                finalQuantOp = qOp;
              }
            }
          }
        }
      }
    }

    SmallVector<Value> inputs = {dequantA.getX(), dequantB.getX()};
    if (!isa<NoneType>(op.getC().getType())) {
      if (auto dequantC = op.getC().getDefiningOp<ONNXDequantizeLinearOp>())
        inputs.push_back(dequantC.getX());
      else
        inputs.push_back(op.getC());
    }

    auto outputType =
        cast<RankedTensorType>(finalQuantOp.getResult().getType());
    auto paramsA = npux::getScalarQuantParams(dequantA);
    auto paramsB = npux::getScalarQuantParams(dequantB);
    auto paramsOut = npux::getScalarQuantParams(finalQuantOp);

    Value result = createNpucoreMatMulOp(rewriter, op.getLoc(), inputs,
        outputType, paramsA.scale, paramsA.zeroPoint, paramsB.scale,
        paramsB.zeroPoint, paramsOut.scale, paramsOut.zeroPoint, doRelu,
        reluType, op.getTransA(), op.getTransB());

    auto restoredOutput = rewriter.create<ONNXDequantizeLinearOp>(op.getLoc(),
        op.getResult().getType(), result, finalQuantOp.getYScale(),
        finalQuantOp.getYZeroPoint());
    rewriter.replaceOp(op, restoredOutput.getResult());

    finalQuantOp.getResult().setType(result.getType());
    rewriter.replaceOp(finalQuantOp, result);
    return success();
  }
};

struct MatMulToNpucore final : public OpConversionPattern<ONNXMatMulOp> {
  using OpConversionPattern<ONNXMatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXMatMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto dequantA = op.getA().getDefiningOp<ONNXDequantizeLinearOp>();
    auto dequantB = op.getB().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantA || !dequantB)
      return failure();

    if (!op.getResult().hasOneUse())
      return failure();
    auto quantOp =
        dyn_cast<ONNXQuantizeLinearOp>(*op.getResult().getUsers().begin());
    if (!quantOp)
      return failure();

    auto outputType = cast<RankedTensorType>(quantOp.getResult().getType());
    auto paramsA = npux::getScalarQuantParams(dequantA);
    auto paramsB = npux::getScalarQuantParams(dequantB);
    auto paramsOut = npux::getScalarQuantParams(quantOp);

    Value result = createNpucoreMatMulOp(rewriter, op.getLoc(),
        {dequantA.getX(), dequantB.getX()}, outputType, paramsA.scale,
        paramsA.zeroPoint, paramsB.scale, paramsB.zeroPoint, paramsOut.scale,
        paramsOut.zeroPoint, 0, 0, false, false);

    auto restoredOutput = rewriter.create<ONNXDequantizeLinearOp>(op.getLoc(),
        op.getResult().getType(), result, quantOp.getYScale(),
        quantOp.getYZeroPoint());
    rewriter.replaceOp(op, restoredOutput.getResult());

    quantOp.getResult().setType(result.getType());
    rewriter.replaceOp(quantOp, result);
    return success();
  }
};

struct QLinearMatMulToNpucore final
    : public OpConversionPattern<ONNXQLinearMatMulOp> {
  using OpConversionPattern<ONNXQLinearMatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXQLinearMatMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    int64_t doRelu = 0;
    int64_t reluType = 0;
    Operation *replaceTarget = op;
    float outScale = getScalarFloat(op.getYScale());
    int64_t outZp = getScalarInt(op.getYZeroPoint());
    auto outputType = cast<RankedTensorType>(op.getResult().getType());

    if (op.getResult().hasOneUse()) {
      if (auto dqOp = dyn_cast<ONNXDequantizeLinearOp>(
              *op.getResult().getUsers().begin())) {
        if (dqOp.getResult().hasOneUse()) {
          Operation *actOp = *dqOp.getResult().getUsers().begin();
          bool isRelu = isa<ONNXReluOp>(actOp);
          bool isLeaky = isa<ONNXLeakyReluOp>(actOp);
          if ((isRelu || isLeaky) && actOp->getResult(0).hasOneUse()) {
            if (auto qOp = dyn_cast<ONNXQuantizeLinearOp>(
                    *actOp->getResult(0).getUsers().begin())) {
              bool canFuse = true;
              if (isLeaky) {
                auto leakyOp = cast<ONNXLeakyReluOp>(actOp);
                float alpha = 0.01f;
                if (auto alphaAttr = leakyOp.getAlphaAttr())
                  alpha = alphaAttr.getValueAsDouble();
                if (std::abs(alpha - 0.1f) < 1e-5)
                  reluType = 2;
                else if (std::abs(alpha - 0.2f) < 1e-5)
                  reluType = 3;
                else if (std::abs(alpha - 0.01f) < 1e-5)
                  reluType = 4;
                else
                  canFuse = false;
              }
              if (canFuse) {
                doRelu = 1;
                replaceTarget = qOp;
                outScale = getScalarFloat(qOp.getYScale());
                outZp = getScalarInt(qOp.getYZeroPoint());
                outputType = cast<RankedTensorType>(qOp.getResult().getType());
              }
            }
          }
        }
      }
    }

    Value result = createNpucoreMatMulOp(rewriter, op.getLoc(),
        {op.getA(), op.getB()}, outputType, getScalarFloat(op.getAScale()),
        getScalarInt(op.getAZeroPoint()), getScalarFloat(op.getBScale()),
        getScalarInt(op.getBZeroPoint()), outScale, outZp, doRelu, reluType,
        false, false);

    op.getResult().setType(result.getType());
    rewriter.replaceOp(op, result);
    if (replaceTarget != op.getOperation()) {
      replaceTarget->getResult(0).setType(result.getType());
      rewriter.replaceOp(replaceTarget, result);
    }
    return success();
  }
};

} // namespace

void npux::populateNpucoreMatMulPatterns(RewritePatternSet &patterns) {
  patterns.add<GemmToNpucore, MatMulToNpucore, QLinearMatMulToNpucore>(
      patterns.getContext());
}
