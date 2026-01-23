//==============================================================
//src/Conversion/NpuPartition/Linalg/Conv.cpp
// this file implements the conversion of ONNX operations to linalg operations
// for NPU partitioning.
//==============================================================

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Compiler/NpuConfig.hpp"
#include "src/Conversion/NpuPartition/LinalgConversionHelper.hpp"

using namespace mlir;
using namespace npux;

namespace {
    struct ConvToLinalg : public OpConversionPattern<ONNXConvOp> {
  using OpConversionPattern<ONNXConvOp>::OpConversionPattern;

  static bool isHardwareSupported(ONNXConvOp op) { return true; }

  // 辅助函数：获取 ONNX 属性中的 Int 数组
  SmallVector<int64_t> getIntArrayAttr(
      Operation *op, StringRef name, int64_t defaultVal, int size) const {
    if (auto attr = op->getAttrOfType<ArrayAttr>(name)) {
      SmallVector<int64_t> values;
      for (auto val : attr.getValue()) {
        values.push_back(mlir::cast<IntegerAttr>(val).getInt());
      }
      return values;
    }
    return SmallVector<int64_t>(size, defaultVal);
  }

  LogicalResult matchAndRewrite(ONNXConvOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    SmallVector<Operation *> opsToErase;

    // ============================================================
    // 0. 准备属性 (Strides, Dilations, Pads)
    // ============================================================
    auto strides = getIntArrayAttr(op, "strides", 1, 2);
    auto dilations = getIntArrayAttr(op, "dilations", 1, 2);
    auto pads = getIntArrayAttr(op, "pads", 0, 4); // [top, left, bottom, right]

    int64_t strideH = strides[0];
    int64_t strideW = strides[1];
    int64_t dilationH = dilations[0];
    int64_t dilationW = dilations[1];
    int64_t padTop = pads[0];
    int64_t padLeft = pads[1];

    std::string nodeName = "";
    if (auto nameAttr = op->getAttrOfType<StringAttr>("onnx_node_name")) {
      nodeName = nameAttr.getValue().str();
    }

    // 2. 准备默认值 (Tile Size)
    int64_t t_oh = 1, t_ow = 1, t_ic = 32, t_oc = 32;

    auto defaultConv = npux::NPUConfig::getInstance().getConvTileSize();
    if (defaultConv.size() == 4) {
      t_oh = defaultConv[0]; t_ow = defaultConv[1];
      t_ic = defaultConv[2]; t_oc = defaultConv[3];
    }

    auto customConfig =
        npux::NPUConfig::getInstance().getCustomLayerConfig(nodeName);

    bool isCustomized = false;
    if (customConfig.has_value()) {
      t_oh = customConfig->t_oh; t_ow = customConfig->t_ow;
      t_ic = customConfig->t_ic; t_oc = customConfig->t_oc;
      isCustomized = true;
    }

    // ============================================================
    // 1. 提取输入和参数 (反量化处理)
    // ============================================================
    Value originInput = op.getX();
    Value weightInput = op.getW();
    Value biasInput = op.getB();

    double inScale = 1.0;
    int64_t inZp = 0;
    if (auto dequantOp = originInput.getDefiningOp<ONNXDequantizeLinearOp>()) {
      originInput = dequantOp.getX();
      auto params = getScalarQuantParams(dequantOp);
      inScale = params.scale;
      inZp = params.zeroPoint;
      opsToErase.push_back(dequantOp);
    }
    if (auto dequantOp = weightInput.getDefiningOp<ONNXDequantizeLinearOp>()) {
      weightInput = dequantOp.getX();
      opsToErase.push_back(dequantOp);
    }
    if (!mlir::isa<NoneType>(biasInput.getType())) {
      if (auto dequantOp = biasInput.getDefiningOp<ONNXDequantizeLinearOp>()) {
        biasInput = dequantOp.getX();
        opsToErase.push_back(dequantOp);
      }
    }

    if (!op.getResult().hasOneUse()) return failure();
    auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(
        *op.getResult().getUsers().begin());
    if (!quantOp) return failure();

    auto qParams = getScalarQuantParams(quantOp);
    double outScale = qParams.scale;
    int64_t outZp = qParams.zeroPoint;

    auto inputType = mlir::dyn_cast<RankedTensorType>(originInput.getType());
    auto weightType = mlir::dyn_cast<RankedTensorType>(weightInput.getType());
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(quantOp.getResult().getType());

    if (!inputType || !weightType || !outputType) return failure();
    int64_t inTileFactor = 32;
    int64_t outTileFactor = 32;

    int64_t inputC = inputType.getShape()[1];
    if (inputC != ShapedType::kDynamic && inputC < 32) inTileFactor = inputC;
    
    int64_t outputC = outputType.getShape()[1];
    if (outputC != ShapedType::kDynamic && outputC < 32) outTileFactor = outputC;

    // ==============================================================================
    // 阶段 1: Pack (Outside Region)
    // ==============================================================================

    // --- Pack Input ---
    SmallVector<int64_t> packedInputShape;
    ArrayRef<int64_t> inShape = inputType.getShape();
    if (inputType.hasStaticShape()) {
      packedInputShape.push_back(inShape[0]); // N
      packedInputShape.push_back((inShape[1] + inTileFactor - 1) / inTileFactor); 
      packedInputShape.push_back(inShape[2]); // H
      packedInputShape.push_back(inShape[3]); // W
      packedInputShape.push_back(inTileFactor); // C_inner
    } else {
      packedInputShape = SmallVector<int64_t>(5, ShapedType::kDynamic);
      packedInputShape[4] = inTileFactor;
    }
    auto packedInputType =
        RankedTensorType::get(packedInputShape, inputType.getElementType());

    SmallVector<Value> inputPackedDynSizes =
        getDynamicSizes(rewriter, loc, originInput, inputType.getShape());
    Value packedInputInit = rewriter.create<tensor::EmptyOp>(
        loc, packedInputType, inputPackedDynSizes);
    Value inputPaddingVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(inputType.getElementType(), inZp));

    auto packInputOp = rewriter.create<linalg::PackOp>(loc, originInput,
        packedInputInit, ArrayRef<int64_t>{1},
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(inTileFactor)},
        inputPaddingVal);

    // --- Pack Weight ---
    SmallVector<int64_t> weightInnerDims = {1, 0};
    SmallVector<OpFoldResult> weightInnerTiles = {
        rewriter.getIndexAttr(inTileFactor), 
        rewriter.getIndexAttr(outTileFactor)};

    SmallVector<int64_t> packedWeightShape;
    ArrayRef<int64_t> wShape = weightType.getShape();
    packedWeightShape.push_back((wShape[0] + outTileFactor - 1) / outTileFactor);
    packedWeightShape.push_back((wShape[1] + inTileFactor - 1) / inTileFactor);
    packedWeightShape.push_back(wShape[2]);
    packedWeightShape.push_back(wShape[3]);
    packedWeightShape.push_back(inTileFactor);
    packedWeightShape.push_back(outTileFactor);

    auto packedWeightType =
        RankedTensorType::get(packedWeightShape, weightType.getElementType());
    Value weightPaddingVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(weightType.getElementType()));
    Value packedWeightInit =
        rewriter.create<tensor::EmptyOp>(loc, packedWeightType, ValueRange{});

    auto packWeightOp =
        rewriter.create<linalg::PackOp>(loc, weightInput, packedWeightInit,
            weightInnerDims, weightInnerTiles, weightPaddingVal);

    // --- Pack Bias ---
    Value processedBias = nullptr;
    if (!mlir::isa<NoneType>(biasInput.getType())) {
      auto biasType = mlir::dyn_cast<RankedTensorType>(biasInput.getType());
      int64_t cDim = biasType.getShape()[0];
      int64_t paddedC = (cDim + outTileFactor - 1) / outTileFactor * outTileFactor;

      Value paddedBias = biasInput;
      if (paddedC > cDim) {
        auto paddedBiasType =
            RankedTensorType::get({paddedC}, biasType.getElementType());
        Value zeroVal = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(biasType.getElementType()));
        auto padOp = rewriter.create<tensor::PadOp>(loc, paddedBiasType,
            biasInput, SmallVector<OpFoldResult>(1, rewriter.getIndexAttr(0)),
            SmallVector<OpFoldResult>(1, rewriter.getIndexAttr(paddedC - cDim)),
            zeroVal);
        paddedBias = padOp.getResult();
      }

      SmallVector<int64_t> packedBiasShape = {
          paddedC / outTileFactor, outTileFactor};
      auto packedBiasType =
          RankedTensorType::get(packedBiasShape, biasType.getElementType());
      Value packedBiasInit =
          rewriter.create<tensor::EmptyOp>(loc, packedBiasType, ValueRange{});

      auto packBiasOp = rewriter.create<linalg::PackOp>(loc, paddedBias,
          packedBiasInit, ArrayRef<int64_t>{0},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(outTileFactor)},
          rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(biasType.getElementType())));
      processedBias = packBiasOp.getResult();
    }

    // ==============================================================================
    // 阶段 2: Pad Input (Outside Region)
    // ==============================================================================
    
    // 默认使用 Pack 后的输入
    Value paddedInput = packInputOp.getResult();
    auto inputRankedType = mlir::cast<RankedTensorType>(packedInputType);

    // 如果存在 Padding，则创建 tensor.pad
    if (padTop > 0 || padLeft > 0 || pads[2] > 0 || pads[3] > 0) {
      int64_t padBottom = pads[2];
      int64_t padRight = pads[3];
      int rank = 5; // [N, C_outer, H, W, C_inner]

      SmallVector<OpFoldResult> low(rank, rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> high(rank, rewriter.getIndexAttr(0));

      low[2] = rewriter.getIndexAttr(padTop);
      low[3] = rewriter.getIndexAttr(padLeft);
      high[2] = rewriter.getIndexAttr(padBottom);
      high[3] = rewriter.getIndexAttr(padRight);

      // Pad Value 使用 Input ZeroPoint
      Value padValue = rewriter.create<arith::ConstantOp>(loc,
          rewriter.getIntegerAttr(inputRankedType.getElementType(), inZp));

      SmallVector<int64_t> paddedShape =
          llvm::to_vector(inputRankedType.getShape());
      if (paddedShape[2] != ShapedType::kDynamic)
        paddedShape[2] += (padTop + padBottom);
      if (paddedShape[3] != ShapedType::kDynamic)
        paddedShape[3] += (padLeft + padRight);

      auto paddedType = RankedTensorType::get(
          paddedShape, inputRankedType.getElementType());

      auto padOp = rewriter.create<tensor::PadOp>(
          loc, paddedType, paddedInput, low, high, padValue);

      // 更新输入为 Pad 后的结果
      paddedInput = padOp.getResult();
    }

    // ==============================================================================
    // 阶段 3: 创建 Execute Region (Only Compute)
    // ==============================================================================
    
    // Output Type Calculation
    SmallVector<int64_t> packedOutputShape;
    ArrayRef<int64_t> outShape = outputType.getShape();
    if (outputType.hasStaticShape()) {
      packedOutputShape.push_back(outShape[0]);
      packedOutputShape.push_back((outShape[1] + outTileFactor - 1) / outTileFactor);
      packedOutputShape.push_back(outShape[2]);
      packedOutputShape.push_back(outShape[3]);
      packedOutputShape.push_back(outTileFactor);
    } else {
      packedOutputShape = SmallVector<int64_t>(5, ShapedType::kDynamic);
      packedOutputShape[4] = outTileFactor;
    }
    auto packedOutputType =
        RankedTensorType::get(packedOutputShape, outputType.getElementType());

    auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(loc, packedOutputType);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&executeRegion.getRegion());

      // --- Compute (Linalg Generic) ---
      
      SmallVector<Value> outPackedDynSizes = getDynamicSizes(
          rewriter, loc, originInput, inputType.getShape()); // Note: Adjust logic if output dims differ significantly in dynamic case
          
      Value regionAlloc = rewriter.create<bufferization::AllocTensorOp>(
          loc, packedOutputType, outPackedDynSizes);

      // 定义 Maps
      int nLoops = 9;
      SmallVector<utils::IteratorType> iteratorTypes = {
          utils::IteratorType::parallel,  // n
          utils::IteratorType::parallel,  // oc_chunk
          utils::IteratorType::parallel,  // oh
          utils::IteratorType::parallel,  // ow
          utils::IteratorType::reduction, // ic_chunk
          utils::IteratorType::reduction, // kh
          utils::IteratorType::reduction, // kw
          utils::IteratorType::reduction, // ic_block
          utils::IteratorType::parallel   // oc_block
      };

      SmallVector<AffineMap> indexingMaps;

      // 1. Input Map: [N, IC_outer, H_in, W_in, IC_inner]
      {
        auto n = rewriter.getAffineDimExpr(0);
        auto ic_chunk = rewriter.getAffineDimExpr(4);
        auto ic_block = rewriter.getAffineDimExpr(7);
        auto oh = rewriter.getAffineDimExpr(2);
        auto kh = rewriter.getAffineDimExpr(5);
        // 注意：这里不需要减去 padTop，因为输入已经是 Pad 过的 Tensor，坐标系已包含 Padding
        auto h_in = oh * strideH + kh * dilationH; 
        auto ow = rewriter.getAffineDimExpr(3);
        auto kw = rewriter.getAffineDimExpr(6);
        auto w_in = ow * strideW + kw * dilationW;

        indexingMaps.push_back(AffineMap::get(nLoops, 0,
            {n, ic_chunk, h_in, w_in, ic_block}, rewriter.getContext()));
      }

      // 2. Weight Map
      {
        auto oc_chunk = rewriter.getAffineDimExpr(1);
        auto ic_chunk = rewriter.getAffineDimExpr(4);
        auto kh = rewriter.getAffineDimExpr(5);
        auto kw = rewriter.getAffineDimExpr(6);
        auto ic_block = rewriter.getAffineDimExpr(7);
        auto oc_block = rewriter.getAffineDimExpr(8);
        indexingMaps.push_back(AffineMap::get(nLoops, 0,
            {oc_chunk, ic_chunk, kh, kw, ic_block, oc_block},
            rewriter.getContext()));
      }

      // 3. Bias Map
      if (processedBias) {
        auto oc_chunk = rewriter.getAffineDimExpr(1);
        auto oc_block = rewriter.getAffineDimExpr(8);
        indexingMaps.push_back(AffineMap::get(
            nLoops, 0, {oc_chunk, oc_block}, rewriter.getContext()));
      }

      // 4. Output Map
      {
        auto n = rewriter.getAffineDimExpr(0);
        auto oc_chunk = rewriter.getAffineDimExpr(1);
        auto oh = rewriter.getAffineDimExpr(2);
        auto ow = rewriter.getAffineDimExpr(3);
        auto oc_block = rewriter.getAffineDimExpr(8);
        indexingMaps.push_back(AffineMap::get(
            nLoops, 0, {n, oc_chunk, oh, ow, oc_block}, rewriter.getContext()));
      }

      SmallVector<Value> genericInputs = {
          paddedInput, packWeightOp.getResult()};
      if (processedBias)
        genericInputs.push_back(processedBias);

      auto linalgOp = rewriter.create<linalg::GenericOp>(loc, packedOutputType,
          genericInputs, regionAlloc, indexingMaps, iteratorTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value acc = args.back();
            Value input = args[0];
            Value weight = args[1];

            Value update;
            if (mlir::isa<FloatType>(acc.getType())) {
              Value sum = b.create<arith::AddFOp>(loc, input, weight);
              if (processedBias && args.size() > 3) {
                sum = b.create<arith::AddFOp>(loc, sum, args[2]);
              }
              update = b.create<arith::AddFOp>(loc, acc, sum);
            } else {
              Type accType = acc.getType();
              auto castToAccType = [&](Value val) -> Value {
                Type valType = val.getType();
                if (valType == accType) return val;
                unsigned accWidth = accType.getIntOrFloatBitWidth();
                unsigned valWidth = valType.getIntOrFloatBitWidth();
                if (valWidth > accWidth) {
                  return b.create<arith::TruncIOp>(loc, accType, val);
                } else {
                  return b.create<arith::ExtSIOp>(loc, accType, val);
                }
              };
              Value castInput = castToAccType(input);
              Value castWeight = castToAccType(weight);
              Value sum = b.create<arith::AddIOp>(loc, castInput, castWeight);
              if (processedBias && args.size() > 3) {
                Value castBias = castToAccType(args[2]);
                sum = b.create<arith::AddIOp>(loc, sum, castBias);
              }
              update = b.create<arith::AddIOp>(loc, acc, sum);
            }
            b.create<linalg::YieldOp>(loc, update);
          });

      // 设置属性
      linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_conv"));
      linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
      auto copyAttr = [&](StringRef name) {
        if (auto attr = op->getAttr(name)) linalgOp->setAttr(name, attr);
      };
      copyAttr("pads"); copyAttr("dilations"); copyAttr("strides");
      copyAttr("group"); copyAttr("kernel_shape"); copyAttr("auto_pad");
      
      linalgOp->setAttr("in_scale", rewriter.getF32FloatAttr(inScale));
      linalgOp->setAttr("in_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), inZp));
      linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
      linalgOp->setAttr("out_zp", rewriter.getIntegerAttr(rewriter.getI16Type(), outZp));
      
      if (isCustomized) {
        SmallVector<int64_t, 4> bestTile = {t_oh, t_ow, t_ic, t_oc};
        linalgOp->setAttr("npu.dse_tiling", rewriter.getI64ArrayAttr(bestTile));
      }
      if (!nodeName.empty()) {
        linalgOp->setAttr("npu.layer_name", rewriter.getStringAttr(nodeName));
      }

      rewriter.create<scf::YieldOp>(loc, linalgOp.getResults());
    }

    // ==============================================================================
    // 阶段 4: Unpack (Outside Region)
    // ==============================================================================
    SmallVector<Value> unpackDynamicSizes =
        getDynamicSizes(rewriter, loc, originInput, outputType.getShape());
    Value unpackDestInit =
        rewriter.create<tensor::EmptyOp>(loc, outputType, unpackDynamicSizes);

    auto unpackOp = rewriter.create<linalg::UnPackOp>(loc,
        executeRegion.getResults()[0], unpackDestInit, ArrayRef<int64_t>{1},
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(outTileFactor)});

    rewriter.replaceOp(quantOp, unpackOp->getResults());

    rewriter.eraseOp(op);
    for (auto *dqOp : opsToErase)
      if (dqOp->hasOneUse()) rewriter.eraseOp(dqOp);

    return success();
  }
};
}

void npux::populateLinalgConvPattern(RewritePatternSet &patterns) {
  patterns.add<ConvToLinalg>(patterns.getContext());
}