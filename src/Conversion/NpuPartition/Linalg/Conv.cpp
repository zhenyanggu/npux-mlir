//==============================================================
// src/Conversion/NpuPartition/Linalg/Conv.cpp
// this file implements the conversion of ONNX operations to linalg operations
// for NPU partitioning.
//==============================================================

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
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace npux;

namespace {

// ============================================================
// Helper: 创建用于 Layout 转换的 Generic Op (Call CAPI)
// ============================================================
Operation *createLayoutGeneric(OpBuilder &rewriter, Location loc, Value input,
    Value outputInit, StringRef libraryCallName, int64_t n, int64_t c,
    int64_t h, int64_t w, int64_t tileSize) {

  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  auto outputType = mlir::cast<RankedTensorType>(outputInit.getType());

  int64_t inputRank = inputType.getRank();
  int64_t outputRank = outputType.getRank();
  int64_t maxRank = std::max(inputRank, outputRank); // 应该是 5

  SmallVector<utils::IteratorType> iterators(
      maxRank, utils::IteratorType::parallel);
  SmallVector<AffineMap> indexingMaps;

  auto context = rewriter.getContext();

  // 定义 Affine 表达式符号
  // d0: N, d1: C_chunk, d2: H, d3: W, d4: C_block
  auto d0 = rewriter.getAffineDimExpr(0);
  auto d1 = rewriter.getAffineDimExpr(1);
  auto d2 = rewriter.getAffineDimExpr(2);
  auto d3 = rewriter.getAffineDimExpr(3);
  auto d4 = rewriter.getAffineDimExpr(4);

  // 1. 构建 5D (Tiled) 的 Map: Identity (d0, d1, d2, d3, d4)
  auto tiledMap = AffineMap::getMultiDimIdentityMap(5, context);

  // 2. 构建 4D (Flat) 的 Map: (d0, d1*32 + d4, d2, d3)
  // 逻辑：Flat Channel = ChunkIdx * 32 + BlockIdx
  SmallVector<AffineExpr> flatExprs = {d0, d1 * tileSize + d4, d2, d3};
  auto flatMap = AffineMap::get(5, 0, flatExprs, context);

  // 根据是 Pack (4D->5D) 还是 Unpack (5D->4D) 决定 Map 顺序
  if (inputRank < outputRank) {
    // Case: NCHW -> NCHWc32 (Pack)
    indexingMaps.push_back(flatMap);  // Input uses Flat Map
    indexingMaps.push_back(tiledMap); // Output uses Tiled Map
  } else {
    // Case: NCHWc32 -> NCHW (Unpack)
    indexingMaps.push_back(tiledMap); // Input uses Tiled Map
    indexingMaps.push_back(flatMap);  // Output uses Flat Map
  }

  auto op = rewriter.create<linalg::GenericOp>(loc, outputType,
      ValueRange{input}, outputInit, indexingMaps, iterators,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args[0]); // Dummy Body
      });

  op->setAttr("library_call", rewriter.getStringAttr(libraryCallName));
  op->setAttr("npu.target", rewriter.getStringAttr("npu"));
  op->setAttr("params_n", rewriter.getI32IntegerAttr(n));
  op->setAttr("params_c", rewriter.getI32IntegerAttr(c));
  op->setAttr("params_h", rewriter.getI32IntegerAttr(h));
  op->setAttr("params_w", rewriter.getI32IntegerAttr(w));

  return op;
}

// ============================================================
// Helper: 编译期 Weight 重排 (OIHW -> NCHWc32)
// ============================================================
Value createPackedWeightConstant(OpBuilder &rewriter, Location loc,
    Value weightInput, int64_t inTileFactor, int64_t outTileFactor) {

  ElementsAttr valueAttr;
  if (auto onnxConst = weightInput.getDefiningOp<ONNXConstantOp>()) {
    if (auto attr = onnxConst.getValueAttr())
      valueAttr = mlir::dyn_cast<ElementsAttr>(attr);
  } else if (auto arithConst = weightInput.getDefiningOp<arith::ConstantOp>()) {
    valueAttr = mlir::cast<ElementsAttr>(arithConst.getValue());
  }

  if (!valueAttr)
    return nullptr;

  auto weightType = mlir::cast<RankedTensorType>(weightInput.getType());
  ArrayRef<int64_t> shape = weightType.getShape(); // [OC, IC, H, W]
  int64_t OC = shape[0];
  int64_t IC = shape[1];
  int64_t H = shape[2];
  int64_t W = shape[3];

  int64_t oc_chunk = (OC + outTileFactor - 1) / outTileFactor;
  int64_t ic_chunk = (IC + inTileFactor - 1) / inTileFactor;
  SmallVector<int64_t> packedShape = {
      oc_chunk, ic_chunk, H, W, inTileFactor, outTileFactor};

  int64_t totalElements =
      oc_chunk * ic_chunk * H * W * inTileFactor * outTileFactor;
  Type elemType = weightType.getElementType();

  DenseElementsAttr newAttr;

  if (elemType.isInteger(8)) {
    std::vector<int8_t> packedData(totalElements, 0);
    auto values = valueAttr.getValues<int8_t>();

    for (int64_t oc = 0; oc < OC; ++oc) {
      for (int64_t ic = 0; ic < IC; ++ic) {
        for (int64_t h = 0; h < H; ++h) {
          for (int64_t w = 0; w < W; ++w) {
            int8_t val = values[oc * (IC * H * W) + ic * (H * W) + h * W + w];

            int64_t o_c = oc / outTileFactor, o_b = oc % outTileFactor;
            int64_t i_c = ic / inTileFactor, i_b = ic % inTileFactor;

            int64_t targetIdx =
                o_c * (ic_chunk * H * W * inTileFactor * outTileFactor) +
                i_c * (H * W * inTileFactor * outTileFactor) +
                h * (W * inTileFactor * outTileFactor) +
                w * (inTileFactor * outTileFactor) + i_b * outTileFactor + o_b;

            packedData[targetIdx] = val;
          }
        }
      }
    }
    auto newType = RankedTensorType::get(packedShape, elemType);
    newAttr = DenseElementsAttr::get(newType, llvm::ArrayRef(packedData));
  }
  else {
    return nullptr;
  }

  auto newConstOp = rewriter.create<ONNXConstantOp>(loc, Attribute(), newAttr);
  return newConstOp.getOutput();
}

// ============================================================
// Helper: 编译期 Bias 处理 (Int32, Pad & Reshape to [OC/32, 32])
// ============================================================
Value createPackedBiasConstant(
    OpBuilder &rewriter, Location loc, Value biasInput, int64_t tileFactor) {
  ElementsAttr valueAttr;
  if (auto onnxConst = biasInput.getDefiningOp<ONNXConstantOp>()) {
    if (auto attr = onnxConst.getValueAttr())
      valueAttr = mlir::dyn_cast<ElementsAttr>(attr);
  } else if (auto arithConst = biasInput.getDefiningOp<arith::ConstantOp>()) {
    valueAttr = mlir::cast<ElementsAttr>(arithConst.getValue());
  }

  if (!valueAttr)
    return nullptr;

  auto biasType = mlir::cast<RankedTensorType>(biasInput.getType());
  int64_t C = biasType.getShape()[0];
  Type elemType = biasType.getElementType();

  if (!elemType.isInteger(32) && !elemType.isSignlessInteger(32) &&
      !elemType.isSignedInteger(32)) {
    return nullptr;
  }

  int64_t c_chunk = (C + tileFactor - 1) / tileFactor;
  SmallVector<int64_t> packedShape = {c_chunk, tileFactor};
  int64_t totalElements = c_chunk * tileFactor;

  std::vector<int32_t> packedData(totalElements, 0);
  auto values = valueAttr.getValues<int32_t>();

  for (int64_t i = 0; i < C; ++i) {
    packedData[i] = values[i];
  }

  auto newType = RankedTensorType::get(packedShape, elemType);
  auto newAttr = DenseElementsAttr::get(newType, llvm::ArrayRef(packedData));

  auto newConstOp = rewriter.create<ONNXConstantOp>(loc, Attribute(), newAttr);
  return newConstOp.getOutput();
}

// ============================================================
// Main Pass: ConvToLinalg
// ============================================================
struct ConvToLinalg : public OpConversionPattern<ONNXConvOp> {
  using OpConversionPattern<ONNXConvOp>::OpConversionPattern;

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

    // 0. 属性解析
    auto strides = getIntArrayAttr(op, "strides", 1, 2);
    auto dilations = getIntArrayAttr(op, "dilations", 1, 2);
    auto pads = getIntArrayAttr(op, "pads", 0, 4);

    int64_t strideH = strides[0];
    int64_t strideW = strides[1];
    int64_t dilationH = dilations[0];
    int64_t dilationW = dilations[1];

    std::string nodeName = "";
    if (auto nameAttr = op->getAttrOfType<StringAttr>("onnx_node_name")) {
      nodeName = nameAttr.getValue().str();
    }

    // 1. 提取输入和参数
    int64_t t_oh = 1, t_ow = 1, t_ic = 32, t_oc = 32;

    auto defaultConv = npux::NPUConfig::getInstance().getConvTileSize();
    if (defaultConv.size() == 4) {
      t_oh = defaultConv[0];
      t_ow = defaultConv[1];
      t_ic = defaultConv[2];
      t_oc = defaultConv[3];
    }

    auto customConfig =
        npux::NPUConfig::getInstance().getCustomLayerConfig(nodeName);

    bool isCustomized = false;
    if (customConfig.has_value()) {
      t_oh = customConfig->t_oh;
      t_ow = customConfig->t_ow;
      t_ic = customConfig->t_ic;
      t_oc = customConfig->t_oc;
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
          auto params = getScalarQuantParams(dequantOp);
          *scaleOut = params.scale;
          *zpOut = params.zeroPoint;
        }
        opsToErase.push_back(dequantOp);
      }
    };

    stripDequant(originInput, &inScale, &inZp);
    stripDequant(weightInput, &wScale, &wZp);
    if (!mlir::isa<NoneType>(biasInput.getType())) {
      stripDequant(biasInput);
    }

    if (!op.getResult().hasOneUse())
      return failure();
    auto quantOpResultUser = mlir::dyn_cast<ONNXQuantizeLinearOp>(
        *op.getResult().getUsers().begin());
    if (!quantOpResultUser)
      return failure();

    // ==============================================================================
    // 阶段 1.5: 前向搜索 Relu 融合 (Q -> DQ -> Act -> Q)
    // ==============================================================================
    int64_t do_relu = 0;
    int64_t relu_type = 0;
    ONNXQuantizeLinearOp finalQuantOp = quantOpResultUser; // 默认停在第一层 Quantize

    if (quantOpResultUser.getResult().hasOneUse()) {
      if (auto dqOp = mlir::dyn_cast<ONNXDequantizeLinearOp>(
              *quantOpResultUser.getResult().getUsers().begin())) {
        if (dqOp.getResult().hasOneUse()) {
          Operation *actOp = *dqOp.getResult().getUsers().begin();
          bool is_relu = mlir::isa<ONNXReluOp>(actOp);
          bool is_leaky = mlir::isa<ONNXLeakyReluOp>(actOp);

          // 此处预留了 relu6(1) 的枚举, 如果后续碰到基于 clip 或者定制的 relu6 op，可以通过额外标记 is_relu6 来处理
          if ((is_relu || is_leaky) && actOp->getResult(0).hasOneUse()) {
            if (auto finalQOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(
                    *actOp->getResult(0).getUsers().begin())) {
              bool can_fuse = true;
              if (is_leaky) {
                auto leakyOp = mlir::cast<ONNXLeakyReluOp>(actOp);
                float alpha = 0.01f;
                if (auto alphaAttr = leakyOp.getAlphaAttr()) {
                  alpha = alphaAttr.getValueAsDouble();
                }
                // 使用浮点偏差检查，以防进度问题
                if (std::abs(alpha - 0.1) < 1e-5) relu_type = 2;
                else if (std::abs(alpha - 0.2) < 1e-5) relu_type = 3;
                else if (std::abs(alpha - 0.01) < 1e-5) relu_type = 4;
                else {
                  llvm::errs() << "Warning: unsupported LeakyRelu alpha " << alpha
                               << ", skipping fusion.\n";
                  can_fuse = false;
                }
              } else {
                relu_type = 0; // relu
              }

              // 检查通过，合并后续计算图
              if (can_fuse) {
                do_relu = 1;
                finalQuantOp = finalQOp;
                // 将被融合掉的中间 op 压入擦除列表
                opsToErase.push_back(quantOpResultUser);
                opsToErase.push_back(dqOp);
                opsToErase.push_back(actOp);
              }
            }
          }
        }
      }
    }

    // 从最终融合保留的 Quantize 节点上提取 params
    auto qParams = getScalarQuantParams(finalQuantOp);
    double outScale = qParams.scale;
    int64_t outZp = qParams.zeroPoint;

    auto inputType = mlir::cast<RankedTensorType>(originInput.getType());
    auto outputType =
        mlir::cast<RankedTensorType>(finalQuantOp.getResult().getType());

    int64_t N = inputType.getShape()[0];
    int64_t IC = inputType.getShape()[1];
    int64_t H = inputType.getShape()[2];
    int64_t W = inputType.getShape()[3];

    int64_t OC = outputType.getShape()[1];
    int64_t OH = outputType.getShape()[2];
    int64_t OW = outputType.getShape()[3];

    int64_t inTileFactor = (IC != ShapedType::kDynamic && IC < 32) ? IC : 32;
    int64_t outTileFactor = (OC != ShapedType::kDynamic && OC < 32) ? OC : 32;

    SmallVector<int64_t> packedInputShape = {
        N, (IC + inTileFactor - 1) / inTileFactor, H, W, inTileFactor};
    if (!inputType.hasStaticShape()) {
      packedInputShape = {ShapedType::kDynamic, ShapedType::kDynamic,
          ShapedType::kDynamic, ShapedType::kDynamic, inTileFactor};
    }
    auto packedInputType =
        RankedTensorType::get(packedInputShape, inputType.getElementType());

    SmallVector<int64_t> packedOutputShape = {
        N, (OC + outTileFactor - 1) / outTileFactor, OH, OW, outTileFactor};
    auto packedOutputType =
        RankedTensorType::get(packedOutputShape, outputType.getElementType());

    // ==============================================================================
    // 阶段 2: Weight & Bias Packing (Compile-time)
    // ==============================================================================
    Value packedWeight = createPackedWeightConstant(
        rewriter, loc, weightInput, inTileFactor, outTileFactor);
    if (!packedWeight)
      return rewriter.notifyMatchFailure(op, "Conv weight must be constant");

    Value packedBias = nullptr;
    if (!mlir::isa<NoneType>(biasInput.getType())) {
      packedBias =
          createPackedBiasConstant(rewriter, loc, biasInput, outTileFactor);
      if (!packedBias)
        return rewriter.notifyMatchFailure(
            op, "Conv bias must be constant int32");
    }

    // ==============================================================================
    // 阶段 3: 创建统一的 Execute Region (Input Packing -> Padding -> 5D Convolution)
    // ==============================================================================

    auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(loc, outputType);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *body = rewriter.createBlock(&executeRegion.getRegion());

      // --- 1. Input Layout Transform (NCHW -> NCHWc32) ---
      Value packedInputAlloc = rewriter.create<bufferization::AllocTensorOp>(
          loc, packedInputType, ValueRange{});

      Operation *packInputOp =
          createLayoutGeneric(rewriter, loc, originInput, packedInputAlloc,
              "npu_layout_nchw_to_nchwc32", N, IC, H, W, inTileFactor);

      Value packedInput = packInputOp->getResult(0);
      Value convInput = packedInput;

      // --- 2. Padding (Inside Region) ---
      bool hasPadding = false;
      for (int64_t p : pads)
        if (p > 0)
          hasPadding = true;

      if (hasPadding) {
        int64_t padTop = pads[0];
        int64_t padLeft = pads[1];
        int64_t padBottom = pads[2];
        int64_t padRight = pads[3];

        int rank = 5;
        SmallVector<OpFoldResult> low(rank, rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> high(rank, rewriter.getIndexAttr(0));

        low[2] = rewriter.getIndexAttr(padTop);
        low[3] = rewriter.getIndexAttr(padLeft);
        high[2] = rewriter.getIndexAttr(padBottom);
        high[3] = rewriter.getIndexAttr(padRight);

        Value padValue;
        Type elemType = inputType.getElementType();
        if (isa<FloatType>(elemType)) {
          padValue = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getFloatAttr(elemType, 0.0));
        } else {
          padValue = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getZeroAttr(elemType));
        }

        auto padOp = rewriter.create<tensor::PadOp>(loc,
            /*resultType=*/nullptr, packedInput, low, high, padValue,
            /*nofold=*/false);
        convInput = padOp.getResult();
      }

      // --- 3. Compute (Packed Conv 全维度处理) ---
      
      // 最终 i8 结果缓冲区 [N, OC_chunk, OH, OW, 32]
      Value outputAlloc = rewriter.create<bufferization::AllocTensorOp>(
          loc, packedOutputType, ValueRange{});

      // 中间 i32 计算缓冲区
      auto outputTypeI32 =
          RankedTensorType::get(packedOutputShape, rewriter.getI32Type());
      Value i32Alloc = rewriter.create<bufferization::AllocTensorOp>(
          loc, outputTypeI32, ValueRange{});

      // 定义 9 维迭代空间 (加入 N 作为 parallel iterator)
      SmallVector<utils::IteratorType> iteratorTypes = {
          utils::IteratorType::parallel,  // N (d0)
          utils::IteratorType::parallel,  // oc_c (d1)
          utils::IteratorType::parallel,  // oh (d2)
          utils::IteratorType::parallel,  // ow (d3)
          utils::IteratorType::reduction, // ic_c (d4)
          utils::IteratorType::reduction, // kh (d5)
          utils::IteratorType::reduction, // kw (d6)
          utils::IteratorType::reduction, // ic_b (d7)
          utils::IteratorType::parallel   // oc_b (d8)
      };

      SmallVector<AffineMap> indexingMaps;
      // Input Map: (N, ic_c, h_in, w_in, ic_b)
      indexingMaps.push_back(AffineMap::get(9, 0,
          {rewriter.getAffineDimExpr(0),
           rewriter.getAffineDimExpr(4),
           rewriter.getAffineDimExpr(2) * strideH +
               rewriter.getAffineDimExpr(5) * dilationH,
           rewriter.getAffineDimExpr(3) * strideW +
               rewriter.getAffineDimExpr(6) * dilationW,
           rewriter.getAffineDimExpr(7)},
          rewriter.getContext()));
      
      // Weight Map: (oc_c, ic_c, kh, kw, ic_b, oc_b)
      indexingMaps.push_back(AffineMap::get(9, 0,
          {rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(4),
           rewriter.getAffineDimExpr(5), rewriter.getAffineDimExpr(6),
           rewriter.getAffineDimExpr(7), rewriter.getAffineDimExpr(8)},
          rewriter.getContext()));
      
      // Bias Map: (oc_c, oc_b) - 如果存在
      if (packedBias)
        indexingMaps.push_back(AffineMap::get(9, 0,
            {rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(8)},
            rewriter.getContext()));
      
      // Output Map: (N, oc_c, oh, ow, oc_b)
      indexingMaps.push_back(AffineMap::get(9, 0,
          {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1),
           rewriter.getAffineDimExpr(2), rewriter.getAffineDimExpr(3),
           rewriter.getAffineDimExpr(8)},
          rewriter.getContext()));

      SmallVector<Value> genericInputs = {convInput, packedWeight};
      if (packedBias)
        genericInputs.push_back(packedBias);

      // 创建核心的 5D 卷积 (N 一并在内处理)
      auto convOp = rewriter.create<linalg::GenericOp>(loc, outputTypeI32,
          genericInputs, i32Alloc, indexingMaps, iteratorTypes,
          [&](OpBuilder &nestedB, Location nestedLoc, ValueRange args) {
            Value in = args[0];
            Value weight = args[1];
            Value acc = args.back(); // i32

            Value inI32 = nestedB.create<arith::ExtSIOp>(
                nestedLoc, nestedB.getI32Type(), in);
            Value weightI32 = nestedB.create<arith::ExtSIOp>(
                nestedLoc, nestedB.getI32Type(), weight);
            Value prod = nestedB.create<arith::MulIOp>(
                nestedLoc, inI32, weightI32);
            Value sum =
                nestedB.create<arith::AddIOp>(nestedLoc, acc, prod);

            if (packedBias && args.size() > 3) {
              sum =
                  nestedB.create<arith::AddIOp>(nestedLoc, sum, args[2]);
            }
            nestedB.create<linalg::YieldOp>(nestedLoc, sum);
          });

      convOp->setAttr("library_call", rewriter.getStringAttr("npu_conv"));
      convOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
      auto copyAttr = [&](StringRef name) {
        if (auto attr = op->getAttr(name))
          convOp->setAttr(name, attr);
      };
      copyAttr("pads");
      copyAttr("dilations");
      copyAttr("strides");
      
      convOp->setAttr("in_scale", rewriter.getF32FloatAttr(inScale));
      convOp->setAttr(
          "in_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), inZp));
        convOp->setAttr("w_scale", rewriter.getF32FloatAttr(wScale));
        convOp->setAttr(
          "w_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), wZp));
      convOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
      convOp->setAttr("out_zp",
          rewriter.getIntegerAttr(rewriter.getI16Type(), outZp));

      // 写入 Relu 融合属性
      convOp->setAttr("do_relu", rewriter.getI32IntegerAttr(do_relu));
      if (do_relu == 1) {
        convOp->setAttr("relu_type", rewriter.getI32IntegerAttr(relu_type));
      }

      if (isCustomized) {
        SmallVector<int64_t, 4> bestTile = {t_oh, t_ow, t_ic, t_oc};
        convOp->setAttr(
            "npu.dse_tiling", rewriter.getI64ArrayAttr(bestTile));
      }
      if (!nodeName.empty()) {
        convOp->setAttr(
            "npu.layer_name", rewriter.getStringAttr(nodeName));
      }

      // --- 4. 5D 量化 ---
      auto identityMap5D = rewriter.getMultiDimIdentityMap(5);
      auto quantOp = rewriter.create<linalg::GenericOp>(loc, packedOutputType,
          ValueRange{convOp.getResult(0)}, ValueRange{outputAlloc},
          SmallVector<AffineMap>{identityMap5D, identityMap5D},
          SmallVector<utils::IteratorType>(
              5, utils::IteratorType::parallel),
          [&](OpBuilder &nestedB, Location nestedLoc, ValueRange args) {
            Value res = nestedB.create<arith::TruncIOp>(
                nestedLoc, nestedB.getI8Type(), args[0]);
            nestedB.create<linalg::YieldOp>(nestedLoc, res);
          });

      quantOp->setAttr("library_call", rewriter.getStringAttr("mv_acc_to_spm"));
      quantOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

      Value packedConvResult = quantOp.getResult(0);

      // ==============================================================================
      // 阶段 4: Output Layout Transform (NCHWc32 -> NCHW)
      // ==============================================================================
      Value outputInit = rewriter.create<bufferization::AllocTensorOp>(
          loc, outputType, ValueRange{});

      Operation *unpackOp =
          createLayoutGeneric(rewriter, loc, packedConvResult, outputInit,
              "npu_layout_nchwc32_to_nchw", N, OC, OH, OW, outTileFactor);

      rewriter.create<scf::YieldOp>(loc, unpackOp->getResult(0));
    }

    Value finalResult = executeRegion.getResult(0);
    // 替换对象改为 finalQuantOp 
    rewriter.replaceOp(finalQuantOp, finalResult);

    // 反向安全擦除被替换后悬空无用的所有节点
    rewriter.eraseOp(op);
    for (auto *opToErase : llvm::reverse(opsToErase)) {
      if (opToErase->hasOneUse()) {
        rewriter.eraseOp(opToErase);
      }
    }

    return success();
  }
};
} // namespace

void npux::populateLinalgConvPattern(RewritePatternSet &patterns) {
  patterns.add<ConvToLinalg>(patterns.getContext());
}