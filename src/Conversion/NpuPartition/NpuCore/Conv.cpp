//==============================================================//
// src/Conversion/NpuPartition/NpuCore/Conv.cpp
// this file implements the conversion of Conv to npucore for NPU
// partitioning.
//==============================================================//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Compiler/NpuConfig.hpp"
#include "src/Conversion/NpuPartition/NpuCoreConversionHelper.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include <cmath>

using namespace mlir;

namespace {

/// Pack a constant weight tensor from OIHW to the original NCHWc32 layout.
static Value createPackedWeightConstant(OpBuilder &rewriter, Location loc,
    Value weightInput, int64_t inTileFactor, int64_t outTileFactor) {
  ElementsAttr valueAttr;
  if (auto onnxConst = weightInput.getDefiningOp<ONNXConstantOp>()) {
    if (auto attr = onnxConst.getValueAttr())
      valueAttr = dyn_cast<ElementsAttr>(attr);
  } else if (auto arithConst = weightInput.getDefiningOp<arith::ConstantOp>()) {
    valueAttr = cast<ElementsAttr>(arithConst.getValue());
  }

  if (!valueAttr)
    return nullptr;

  auto weightType = cast<RankedTensorType>(weightInput.getType());
  ArrayRef<int64_t> shape = weightType.getShape();
  int64_t oc = shape[0];
  int64_t ic = shape[1];
  int64_t h = shape[2];
  int64_t w = shape[3];

  int64_t ocChunk = (oc + outTileFactor - 1) / outTileFactor;
  int64_t icChunk = (ic + inTileFactor - 1) / inTileFactor;
  SmallVector<int64_t> packedShape = {
      ocChunk, icChunk, h, w, inTileFactor, outTileFactor};

  int64_t totalElements =
      ocChunk * icChunk * h * w * inTileFactor * outTileFactor;
  Type elemType = weightType.getElementType();
  if (!elemType.isInteger(8))
    return nullptr;

  std::vector<int8_t> packedData(totalElements, 0);
  auto values = valueAttr.getValues<int8_t>();

  for (int64_t ocIdx = 0; ocIdx < oc; ++ocIdx) {
    for (int64_t icIdx = 0; icIdx < ic; ++icIdx) {
      for (int64_t hIdx = 0; hIdx < h; ++hIdx) {
        for (int64_t wIdx = 0; wIdx < w; ++wIdx) {
          int8_t val =
              values[ocIdx * (ic * h * w) + icIdx * (h * w) + hIdx * w + wIdx];
          int64_t ocOuter = ocIdx / outTileFactor;
          int64_t ocInner = ocIdx % outTileFactor;
          int64_t icOuter = icIdx / inTileFactor;
          int64_t icInner = icIdx % inTileFactor;

          int64_t targetIdx =
              ocOuter * (icChunk * h * w * inTileFactor * outTileFactor) +
              icOuter * (h * w * inTileFactor * outTileFactor) +
              hIdx * (w * inTileFactor * outTileFactor) +
              wIdx * (inTileFactor * outTileFactor) +
              icInner * outTileFactor + ocInner;
          packedData[targetIdx] = val;
        }
      }
    }
  }

  auto newType = RankedTensorType::get(packedShape, elemType);
  auto newAttr = DenseElementsAttr::get(newType, llvm::ArrayRef(packedData));
  return rewriter.create<ONNXConstantOp>(loc, Attribute(), newAttr).getOutput();
}

/// Pack a constant bias tensor to the original tiled layout.
static Value createPackedBiasConstant(
    OpBuilder &rewriter, Location loc, Value biasInput, int64_t tileFactor) {
  ElementsAttr valueAttr;
  if (auto onnxConst = biasInput.getDefiningOp<ONNXConstantOp>()) {
    if (auto attr = onnxConst.getValueAttr())
      valueAttr = dyn_cast<ElementsAttr>(attr);
  } else if (auto arithConst = biasInput.getDefiningOp<arith::ConstantOp>()) {
    valueAttr = cast<ElementsAttr>(arithConst.getValue());
  }

  if (!valueAttr)
    return nullptr;

  auto biasType = cast<RankedTensorType>(biasInput.getType());
  int64_t c = biasType.getShape()[0];
  Type elemType = biasType.getElementType();
  if (!elemType.isInteger(32) && !elemType.isSignlessInteger(32) &&
      !elemType.isSignedInteger(32)) {
    return nullptr;
  }

  int64_t cChunk = (c + tileFactor - 1) / tileFactor;
  SmallVector<int64_t> packedShape = {cChunk, tileFactor};
  int64_t totalElements = cChunk * tileFactor;

  std::vector<int32_t> packedData(totalElements, 0);
  auto values = valueAttr.getValues<int32_t>();
  for (int64_t i = 0; i < c; ++i)
    packedData[i] = values[i];

  auto newType = RankedTensorType::get(packedShape, elemType);
  auto newAttr = DenseElementsAttr::get(newType, llvm::ArrayRef(packedData));
  return rewriter.create<ONNXConstantOp>(loc, Attribute(), newAttr).getOutput();
}

/// Lower ONNX Conv to a packed npucore.conv that mirrors the original linalg
/// conv path.
struct ConvToNpucore final : public OpConversionPattern<ONNXConvOp> {
  using OpConversionPattern<ONNXConvOp>::OpConversionPattern;

  SmallVector<int64_t> getIntArrayAttr(
      Operation *op, StringRef name, int64_t defaultVal, int size) const {
    if (auto attr = op->getAttrOfType<ArrayAttr>(name)) {
      SmallVector<int64_t> values;
      for (auto val : attr.getValue())
        values.push_back(cast<IntegerAttr>(val).getInt());
      return values;
    }
    return SmallVector<int64_t>(size, defaultVal);
  }

  LogicalResult matchAndRewrite(ONNXConvOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto strides = getIntArrayAttr(op, "strides", 1, 2);
    auto dilations = getIntArrayAttr(op, "dilations", 1, 2);
    auto pads = getIntArrayAttr(op, "pads", 0, 4);

    int64_t strideH = strides[0];
    int64_t strideW = strides[1];
    int64_t dilationH = dilations[0];
    int64_t dilationW = dilations[1];

    std::string nodeName;
    if (auto nameAttr = op->getAttrOfType<StringAttr>("onnx_node_name"))
      nodeName = nameAttr.getValue().str();
    StringAttr nodeNameAttr = rewriter.getStringAttr(nodeName);

    int64_t tOh = 1;
    int64_t tOw = 1;
    int64_t tIc = 32;
    int64_t tOc = 32;
    auto defaultConv = npux::NPUConfig::getInstance().getConvTileSize();
    if (defaultConv.size() == 4) {
      tOh = defaultConv[0];
      tOw = defaultConv[1];
      tIc = defaultConv[2];
      tOc = defaultConv[3];
    }

    auto customConfig =
        npux::NPUConfig::getInstance().getCustomLayerConfig(nodeName);
    bool isCustomized = false;
    if (customConfig.has_value()) {
      tOh = customConfig->t_oh;
      tOw = customConfig->t_ow;
      tIc = customConfig->t_ic;
      tOc = customConfig->t_oc;
      isCustomized = true;
    }

    Value originInput = op.getX();
    Value weightInput = op.getW();
    Value biasInput = op.getB();

    double inScale = 1.0;
    int64_t inZp = 0;
    double wScale = 1.0;
    int64_t wZp = 0;

    auto stripDequant =
        [&](Value &val, double *scaleOut = nullptr, int64_t *zpOut = nullptr) {
          if (auto dequantOp = val.getDefiningOp<ONNXDequantizeLinearOp>()) {
            val = dequantOp.getX();
            if (scaleOut && zpOut) {
              auto params = npux::getScalarQuantParams(dequantOp);
              *scaleOut = params.scale;
              *zpOut = params.zeroPoint;
            }
          }
        };

    stripDequant(originInput, &inScale, &inZp);
    stripDequant(weightInput, &wScale, &wZp);
    if (!isa<NoneType>(biasInput.getType()))
      stripDequant(biasInput);

    if (!op.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(
          op, "Conv result must have exactly one QuantizeLinear consumer");
    }
    auto quantOpResultUser =
        dyn_cast<ONNXQuantizeLinearOp>(*op.getResult().getUsers().begin());
    if (!quantOpResultUser) {
      return rewriter.notifyMatchFailure(
          op, "Conv result consumer must be QuantizeLinear");
    }

    int64_t doRelu = 0;
    int64_t reluType = 0;
    ONNXQuantizeLinearOp finalQuantOp = quantOpResultUser;
    if (quantOpResultUser.getResult().hasOneUse()) {
      if (auto dqOp = dyn_cast<ONNXDequantizeLinearOp>(
              *quantOpResultUser.getResult().getUsers().begin())) {
        if (dqOp.getResult().hasOneUse()) {
          Operation *actOp = *dqOp.getResult().getUsers().begin();
          bool isRelu = isa<ONNXReluOp>(actOp);
          bool isLeaky = isa<ONNXLeakyReluOp>(actOp);
          if ((isRelu || isLeaky) && actOp->getResult(0).hasOneUse()) {
            if (auto finalQOp = dyn_cast<ONNXQuantizeLinearOp>(
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
              } else {
                reluType = 0;
              }

              if (canFuse) {
                doRelu = 1;
                finalQuantOp = finalQOp;
              }
            }
          }
        }
      }
    }

    auto qParams = npux::getScalarQuantParams(finalQuantOp);
    double outScale = qParams.scale;
    int64_t outZp = qParams.zeroPoint;

    auto inputType = cast<RankedTensorType>(originInput.getType());
    auto inputType1 = npux::addEncoding1(inputType, rewriter);
    originInput.setType(inputType1);

    auto outputType = cast<RankedTensorType>(finalQuantOp.getResult().getType());
    int64_t n = inputType.getShape()[0];
    int64_t ic = inputType.getShape()[1];
    int64_t h = inputType.getShape()[2];
    int64_t w = inputType.getShape()[3];
    int64_t oc = outputType.getShape()[1];
    int64_t oh = outputType.getShape()[2];
    int64_t ow = outputType.getShape()[3];

    auto kernelShape = getIntArrayAttr(op, "kernel_shape", 1, 2);
    int64_t kH = kernelShape[0];
    int64_t kW = kernelShape[1];

    std::string autoPad = "NOTSET";
    if (auto autoPadAttr = op->getAttrOfType<StringAttr>("auto_pad"))
      autoPad = autoPadAttr.getValue().str();

    if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
      int64_t pTotalH = (oh - 1) * strideH + (kH - 1) * dilationH + 1 - h;
      int64_t pTotalW = (ow - 1) * strideW + (kW - 1) * dilationW + 1 - w;
      pTotalH = std::max<int64_t>(0, pTotalH);
      pTotalW = std::max<int64_t>(0, pTotalW);

      int64_t pTop = 0;
      int64_t pBottom = 0;
      int64_t pLeft = 0;
      int64_t pRight = 0;
      if (autoPad == "SAME_UPPER") {
        pTop = pTotalH / 2;
        pBottom = pTotalH - pTop;
        pLeft = pTotalW / 2;
        pRight = pTotalW - pLeft;
      } else {
        pBottom = pTotalH / 2;
        pTop = pTotalH - pBottom;
        pRight = pTotalW / 2;
        pLeft = pTotalW - pRight;
      }
      pads = {pTop, pLeft, pBottom, pRight};
    } else if (autoPad == "VALID") {
      pads = {0, 0, 0, 0};
    }

    bool hasPadding = llvm::any_of(pads, [](int64_t pad) { return pad > 0; });
    int64_t paddedH = h;
    int64_t paddedW = w;
    if (hasPadding) {
      paddedH += pads[0] + pads[2];
      paddedW += pads[1] + pads[3];
    }

    int64_t inTileFactor = (ic != ShapedType::kDynamic && ic < 32) ? ic : 32;
    int64_t outTileFactor = (oc != ShapedType::kDynamic && oc < 32) ? oc : 32;

    SmallVector<int64_t> packedInputShape = {
        n, (ic + inTileFactor - 1) / inTileFactor, paddedH, paddedW,
        inTileFactor};
    if (!inputType.hasStaticShape()) {
      packedInputShape = {ShapedType::kDynamic, ShapedType::kDynamic,
          ShapedType::kDynamic, ShapedType::kDynamic, inTileFactor};
    }
    auto packedInputType = RankedTensorType::get(
        packedInputShape, inputType.getElementType());
    auto packedInputType1 = npux::addEncoding1(packedInputType, rewriter);

    SmallVector<int64_t> packedOutputShape = {
        n, (oc + outTileFactor - 1) / outTileFactor, oh, ow, outTileFactor};
    auto packedOutputType = RankedTensorType::get(
        packedOutputShape, outputType.getElementType());
    auto packedOutputType1 = npux::addEncoding1(packedOutputType, rewriter);

    Value packedWeight = createPackedWeightConstant(
        rewriter, loc, weightInput, inTileFactor, outTileFactor);
    if (!packedWeight)
      return rewriter.notifyMatchFailure(op, "Conv weight must be constant");

    Value packedBias = nullptr;
    if (!isa<NoneType>(biasInput.getType())) {
      packedBias =
          createPackedBiasConstant(rewriter, loc, biasInput, outTileFactor);
      if (!packedBias)
        return rewriter.notifyMatchFailure(
            op, "Conv bias must be constant int32");
    }

    Value paddedInput = originInput;
    if (hasPadding) {
      SmallVector<OpFoldResult> low(4, rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> high(4, rewriter.getIndexAttr(0));
      low[2] = rewriter.getIndexAttr(pads[0]);
      low[3] = rewriter.getIndexAttr(pads[1]);
      high[2] = rewriter.getIndexAttr(pads[2]);
      high[3] = rewriter.getIndexAttr(pads[3]);

      Type elemType = inputType.getElementType();
      Value padValue = isa<FloatType>(elemType)
                           ? rewriter.create<arith::ConstantOp>(
                                 loc, rewriter.getFloatAttr(elemType, 0.0))
                           : rewriter.create<arith::ConstantOp>(
                                 loc, rewriter.getZeroAttr(elemType));

      SmallVector<int64_t> paddedShape = {n, ic, paddedH, paddedW};
      if (!inputType.hasStaticShape()) {
        paddedShape = {ShapedType::kDynamic, ShapedType::kDynamic,
            ShapedType::kDynamic, ShapedType::kDynamic};
      }
      auto paddedType = RankedTensorType::get(paddedShape, elemType);
      auto paddedType1 = npux::addEncoding1(paddedType, rewriter);
      paddedInput = rewriter
                        .create<tensor::PadOp>(loc, paddedType1, originInput,
                            low, high, padValue, /*nofold=*/false)
                        .getResult();
    }

    Value packedInputAlloc = rewriter.create<bufferization::AllocTensorOp>(
        loc, packedInputType1, ValueRange{});
    auto packInputOp = rewriter.create<npucore::LayoutNchwToNchwc32Op>(loc,
        TypeRange{packedInputType1}, ValueRange{paddedInput},
        ValueRange{packedInputAlloc}, rewriter.getI64IntegerAttr(inTileFactor));
    packInputOp->setAttr("npu.layer_name", nodeNameAttr);
    packInputOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    Value convInput = packInputOp.getResultTensors().front();

    auto packedWeightType = cast<RankedTensorType>(packedWeight.getType());
    auto packedWeightType1 = npux::addEncoding1(packedWeightType, rewriter);
    Value weightAlloc = rewriter.create<bufferization::AllocTensorOp>(
        loc, packedWeightType1, ValueRange{});
    Value copiedWeight =
        rewriter.create<bufferization::MaterializeInDestinationOp>(
            loc, packedWeight, weightAlloc)
            .getResult();

    Value copiedBias = nullptr;
    if (packedBias) {
      auto packedBiasType = cast<RankedTensorType>(packedBias.getType());
      auto packedBiasType1 = npux::addEncoding1(packedBiasType, rewriter);
      Value biasAlloc = rewriter.create<bufferization::AllocTensorOp>(
          loc, packedBiasType1, ValueRange{});
      copiedBias =
          rewriter.create<bufferization::MaterializeInDestinationOp>(
              loc, packedBias, biasAlloc)
              .getResult();
    }

    auto outputTypeI32 = RankedTensorType::get(packedOutputShape,
        rewriter.getI32Type());
    auto outputTypeI32_1 = npux::addEncoding1(outputTypeI32, rewriter);
    Value i32Alloc = rewriter.create<bufferization::AllocTensorOp>(
        loc, outputTypeI32_1, ValueRange{});
    Value outputAlloc = rewriter.create<bufferization::AllocTensorOp>(
        loc, packedOutputType1, ValueRange{});

    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(AffineMap::get(9, 0,
        {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(4),
            rewriter.getAffineDimExpr(2) * strideH +
                rewriter.getAffineDimExpr(5) * dilationH,
            rewriter.getAffineDimExpr(3) * strideW +
                rewriter.getAffineDimExpr(6) * dilationW,
            rewriter.getAffineDimExpr(7)},
        rewriter.getContext()));
    indexingMaps.push_back(AffineMap::get(9, 0,
        {rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(4),
            rewriter.getAffineDimExpr(5), rewriter.getAffineDimExpr(6),
            rewriter.getAffineDimExpr(7), rewriter.getAffineDimExpr(8)},
        rewriter.getContext()));
    if (copiedBias) {
      indexingMaps.push_back(AffineMap::get(9, 0,
          {rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(8)},
          rewriter.getContext()));
    }
    indexingMaps.push_back(AffineMap::get(9, 0,
        {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1),
            rewriter.getAffineDimExpr(2), rewriter.getAffineDimExpr(3),
            rewriter.getAffineDimExpr(8)},
        rewriter.getContext()));

    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::reduction, utils::IteratorType::reduction,
        utils::IteratorType::reduction, utils::IteratorType::reduction,
        utils::IteratorType::parallel};

    SmallVector<Value> convInputs = {convInput, copiedWeight};
    if (copiedBias)
      convInputs.push_back(copiedBias);

    auto convOp = rewriter.create<npucore::ConvOp>(loc, TypeRange{outputTypeI32_1},
        convInputs, ValueRange{i32Alloc}, rewriter.getF32FloatAttr(inScale),
        rewriter.getIntegerAttr(rewriter.getI32Type(), inZp),
        rewriter.getF32FloatAttr(wScale),
        rewriter.getIntegerAttr(rewriter.getI32Type(), wZp),
        rewriter.getF32FloatAttr(outScale),
        rewriter.getIntegerAttr(rewriter.getI16Type(), outZp),
        rewriter.getI64ArrayAttr(pads), rewriter.getI64ArrayAttr(strides),
        rewriter.getI64ArrayAttr(dilations),
        rewriter.getIntegerAttr(rewriter.getI64Type(), 1),
        rewriter.getI32IntegerAttr(doRelu),
        rewriter.getI32IntegerAttr(reluType));
    convOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    if (isCustomized) {
      SmallVector<int64_t, 4> bestTile = {tOh, tOw, tIc, tOc};
      convOp->setAttr("npu.dse_tiling", rewriter.getI64ArrayAttr(bestTile));
    }
    if (!nodeName.empty())
      convOp->setAttr("npu.layer_name", nodeNameAttr);

    auto quantOp = rewriter.create<npucore::MvAccToSpmOp>(loc,
        TypeRange{packedOutputType1}, ValueRange{convOp.getResultTensors()[0]},
        ValueRange{outputAlloc});
    quantOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    if (!nodeName.empty())
      quantOp->setAttr("npu.layer_name", nodeNameAttr);

    auto outputType1 = npux::addEncoding1(outputType, rewriter);
    Value outputInit = rewriter.create<bufferization::AllocTensorOp>(
        loc, outputType1, ValueRange{});
    auto unpackOp = rewriter.create<npucore::LayoutNchwc32ToNchwOp>(loc,
        TypeRange{outputType1}, ValueRange{quantOp.getResultTensors()[0]},
        ValueRange{outputInit}, rewriter.getI64IntegerAttr(outTileFactor));
    unpackOp->setAttr("npu.layer_name", nodeNameAttr);
    unpackOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    Value finalResult = unpackOp.getResultTensors().front();

    auto restoredOutput = rewriter.create<ONNXDequantizeLinearOp>(loc,
        op.getResult().getType(), finalResult, finalQuantOp.getYScale(),
        finalQuantOp.getYZeroPoint());
    rewriter.replaceOp(op, restoredOutput.getResult());

    finalQuantOp.getResult().setType(finalResult.getType());
    rewriter.replaceOp(finalQuantOp, finalResult);
    return success();
  }
};

} // namespace

void npux::populateNpucoreConvPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvToNpucore>(patterns.getContext());
}
