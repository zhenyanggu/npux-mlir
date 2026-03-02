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

  // 示例：处理 F32，如果是 Int8 需要增加对应的分支
  if (elemType.isInteger(8)) {
    // 容器使用 int8_t
    std::vector<int8_t> packedData(totalElements, 0);
    // 从属性中获取 int8 数据
    auto values = valueAttr.getValues<int8_t>();

    for (int64_t oc = 0; oc < OC; ++oc) {
      for (int64_t ic = 0; ic < IC; ++ic) {
        for (int64_t h = 0; h < H; ++h) {
          for (int64_t w = 0; w < W; ++w) {
            // 注意：这里假设源数据布局是 [OC, IC, H, W]
            // 如果 IR 中已经是其他布局，需要调整索引计算
            int8_t val = values[oc * (IC * H * W) + ic * (H * W) + h * W + w];

            int64_t o_c = oc / outTileFactor, o_b = oc % outTileFactor;
            int64_t i_c = ic / inTileFactor, i_b = ic % inTileFactor;

            // NPU 目标布局: [OC_chunk, IC_chunk, H, W, IC_block, OC_block]
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
  // 若 Weight 是 Int8，请在此添加 else if (elemType.isInteger(8)) 逻辑
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

  // 确保 Bias 是 Int32
  if (!elemType.isInteger(32) && !elemType.isSignlessInteger(32) &&
      !elemType.isSignedInteger(32)) {
    return nullptr;
  }

  int64_t c_chunk = (C + tileFactor - 1) / tileFactor;
  SmallVector<int64_t> packedShape = {c_chunk, tileFactor};
  int64_t totalElements = c_chunk * tileFactor;

  // 专门处理 Int32
  std::vector<int32_t> packedData(totalElements, 0); // 默认补 0
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

    // Helper lambda to strip Dequantize
    auto stripDequant = [&](Value &val, bool extractParams = false) {
      if (auto dequantOp = val.getDefiningOp<ONNXDequantizeLinearOp>()) {
        val = dequantOp.getX();
        if (extractParams) {
          auto params = getScalarQuantParams(dequantOp);
          inScale = params.scale;
          inZp = params.zeroPoint;
        }
        opsToErase.push_back(dequantOp);
      }
    };

    stripDequant(originInput, true);
    stripDequant(weightInput);
    if (!mlir::isa<NoneType>(biasInput.getType())) {
      stripDequant(biasInput);
    }

    if (!op.getResult().hasOneUse())
      return failure();
    auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(
        *op.getResult().getUsers().begin());
    if (!quantOp)
      return failure();

    auto qParams = getScalarQuantParams(quantOp);
    double outScale = qParams.scale;
    int64_t outZp = qParams.zeroPoint;

    auto inputType = mlir::cast<RankedTensorType>(originInput.getType());
    auto outputType =
        mlir::cast<RankedTensorType>(quantOp.getResult().getType());

    // 原始 NCHW 维度
    int64_t N = inputType.getShape()[0];
    int64_t IC = inputType.getShape()[1];
    int64_t H = inputType.getShape()[2];
    int64_t W = inputType.getShape()[3];

    int64_t OC = outputType.getShape()[1];
    int64_t OH = outputType.getShape()[2];
    int64_t OW = outputType.getShape()[3];

    int64_t inTileFactor = (IC != ShapedType::kDynamic && IC < 32) ? IC : 32;
    int64_t outTileFactor = (OC != ShapedType::kDynamic && OC < 32) ? OC : 32;

    // 准备 Packed Input 的 Type
    SmallVector<int64_t> packedInputShape = {
        N, (IC + inTileFactor - 1) / inTileFactor, H, W, inTileFactor};
    if (!inputType.hasStaticShape()) {
      packedInputShape = {ShapedType::kDynamic, ShapedType::kDynamic,
          ShapedType::kDynamic, ShapedType::kDynamic, inTileFactor};
    }
    auto packedInputType =
        RankedTensorType::get(packedInputShape, inputType.getElementType());

    // 准备 Packed Output 的 Type
    SmallVector<int64_t> packedOutputShape = {
        N, (OC + outTileFactor - 1) / outTileFactor, OH, OW, outTileFactor};
    auto packedOutputType =
        RankedTensorType::get(packedOutputShape, outputType.getElementType());

    // ==============================================================================
    // 阶段 2: Weight & Bias Packing (Compile-time)
    // 注意：常量折叠（Weight/Bias Packing）通常保留在 Region 外部或作为
    // ConstantOp 存在
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
    // 核心修改：创建统一的 Execute Region
    // 包含：Input Packing -> Padding -> Convolution
    // ==============================================================================

    // Region 的返回值是 Packed 的卷积结果
    auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(loc, outputType);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *body = rewriter.createBlock(&executeRegion.getRegion());

      // --- 1. Input Layout Transform (NCHW -> NCHWc32) ---

      // 在 Region 内部分配 Input Packing 的 Buffer
      Value packedInputAlloc = rewriter.create<bufferization::AllocTensorOp>(
          loc, packedInputType, ValueRange{});
      ;

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

      // --- 3. Compute (Packed Conv) ---

      Value outputAlloc = rewriter.create<bufferization::AllocTensorOp>(
          loc, packedOutputType, ValueRange{});

      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value nLoopUpper = rewriter.create<arith::ConstantIndexOp>(loc, N);

      auto nLoop = rewriter.create<scf::ForOp>(loc, zero, nLoopUpper, one,
          ValueRange{outputAlloc},
          [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            Value outAccum = iterArgs[0]; // 这是最终的 i8 Tensor

            OpFoldResult oneAttr = b.getIndexAttr(1);
            OpFoldResult zeroAttr = b.getIndexAttr(0);
            SmallVector<OpFoldResult> offsets = {
                iv, zeroAttr, zeroAttr, zeroAttr, zeroAttr};
            SmallVector<OpFoldResult> strides = {
                oneAttr, oneAttr, oneAttr, oneAttr, oneAttr};

            // --- 1. 准备 Input Slice ---
            auto convInputType =
                mlir::cast<RankedTensorType>(convInput.getType());
            auto inputShape = convInputType.getShape();
            SmallVector<OpFoldResult> inputSizes = {oneAttr};
            for (int i = 1; i < 5; ++i) {
              if (convInputType.isDynamicDim(i))
                inputSizes.push_back(
                    b.create<tensor::DimOp>(loc, convInput, i).getResult());
              else
                inputSizes.push_back(b.getIndexAttr(inputShape[i]));
            }
            auto sliceInputType = RankedTensorType::get(
                {inputShape[1], inputShape[2], inputShape[3], inputShape[4]},
                convInputType.getElementType());
            Value inputSlice = b.create<tensor::ExtractSliceOp>(
                loc, sliceInputType, convInput, offsets, inputSizes, strides);

            // --- 2. 准备中间 i32 Slice 的类型和分配 ---
            auto outAccType = mlir::cast<RankedTensorType>(outAccum.getType());
            auto outputShape = outAccType.getShape(); // [N, OC_c, OH, OW, 32]

            // 中间 i32 容器的形状 [OC_c, OH, OW, 32]
            SmallVector<int64_t> i32SliceShape = {
                outputShape[1], outputShape[2], outputShape[3], outputShape[4]};
            auto sliceOutputTypeI32 =
                RankedTensorType::get(i32SliceShape, b.getI32Type());

            // 分配临时的 i32 累加器 (通常在 NPU 上是 Accumulator 寄存器)
            Value i32Alloc = b.create<bufferization::AllocTensorOp>(
                loc, sliceOutputTypeI32, ValueRange{});

            // --- 3. 创建卷积 Generic Op (输出 i32) ---
            SmallVector<utils::IteratorType> iteratorTypes = {
                utils::IteratorType::parallel,  // oc_c
                utils::IteratorType::parallel,  // oh
                utils::IteratorType::parallel,  // ow
                utils::IteratorType::reduction, // ic_c
                utils::IteratorType::reduction, // kh
                utils::IteratorType::reduction, // kw
                utils::IteratorType::reduction, // ic_b
                utils::IteratorType::parallel   // oc_b
            };

            SmallVector<AffineMap> indexingMaps;
            // Input Map: (ic_c, h_in, w_in, ic_b)
            indexingMaps.push_back(AffineMap::get(8, 0,
                {b.getAffineDimExpr(3),
                    b.getAffineDimExpr(1) * strideH +
                        b.getAffineDimExpr(4) * dilationH,
                    b.getAffineDimExpr(2) * strideW +
                        b.getAffineDimExpr(5) * dilationW,
                    b.getAffineDimExpr(6)},
                b.getContext()));
            // Weight Map: (oc_c, ic_c, kh, kw, ic_b, oc_b)
            indexingMaps.push_back(AffineMap::get(8, 0,
                {b.getAffineDimExpr(0), b.getAffineDimExpr(3),
                    b.getAffineDimExpr(4), b.getAffineDimExpr(5),
                    b.getAffineDimExpr(6), b.getAffineDimExpr(7)},
                b.getContext()));
            // Bias Map (if exists)
            if (packedBias)
              indexingMaps.push_back(AffineMap::get(8, 0,
                  {b.getAffineDimExpr(0), b.getAffineDimExpr(7)},
                  b.getContext()));
            // Output Map: (oc_c, oh, ow, oc_b)
            indexingMaps.push_back(AffineMap::get(8, 0,
                {b.getAffineDimExpr(0), b.getAffineDimExpr(1),
                    b.getAffineDimExpr(2), b.getAffineDimExpr(7)},
                b.getContext()));

            SmallVector<Value> genericInputs = {inputSlice, packedWeight};
            if (packedBias)
              genericInputs.push_back(packedBias);

            auto convOp = b.create<linalg::GenericOp>(loc, sliceOutputTypeI32,
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

            // 设置卷积属性
            convOp->setAttr("library_call", b.getStringAttr("npu_conv"));
            convOp->setAttr("npu.target", b.getStringAttr("npu"));
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
            convOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
            convOp->setAttr("out_zp",
                rewriter.getIntegerAttr(rewriter.getI16Type(), outZp));
            if (isCustomized) {
              SmallVector<int64_t, 4> bestTile = {t_oh, t_ow, t_ic, t_oc};
              convOp->setAttr(
                  "npu.dse_tiling", rewriter.getI64ArrayAttr(bestTile));
            }
            if (!nodeName.empty()) {
              convOp->setAttr(
                  "npu.layer_name", rewriter.getStringAttr(nodeName));
            }

            Value result4D = convOp.getResult(0);

            // ==========================================
            // Step C: Insert Slice back to 5D (rank-restore)
            // ==========================================

            // 1. 提前计算 outputSizes，供 Extract 和 Insert 共同使用
            SmallVector<OpFoldResult> outputSizes = {oneAttr};
            for (int i = 1; i < 5; ++i)
              outputSizes.push_back(b.getIndexAttr(outputShape[i]));

            auto sliceOutputTypeI8 =
                RankedTensorType::get(i32SliceShape, b.getI8Type());

            // 2. 核心修复：不再使用 alloc_tensor，而是从外层 iter_args
            // (outAccum) 中把目标切片取出来
            Value i8Dest = b.create<tensor::ExtractSliceOp>(loc,
                sliceOutputTypeI8, outAccum, offsets, outputSizes, strides);

            // 定义 4D 的 Identity Map 用于 Element-wise 搬移
            auto identityMap4D = b.getMultiDimIdentityMap(4);

            // 3. 将取出来的 i8Dest 作为 outs 传入 GenericOp
            auto quantOp = b.create<linalg::GenericOp>(loc, sliceOutputTypeI8,
                ValueRange{convOp.getResult(0)}, ValueRange{i8Dest},
                SmallVector<AffineMap>{identityMap4D, identityMap4D},
                SmallVector<utils::IteratorType>(
                    4, utils::IteratorType::parallel),
                [&](OpBuilder &nestedB, Location nestedLoc, ValueRange args) {
                  // 这里实现量化逻辑，最简单的是 Truncate
                  Value res = nestedB.create<arith::TruncIOp>(
                      nestedLoc, nestedB.getI8Type(), args[0]);
                  nestedB.create<linalg::YieldOp>(nestedLoc, res);
                });

            quantOp->setAttr("library_call", b.getStringAttr("mv_acc_to_spm"));
            quantOp->setAttr("npu.target", b.getStringAttr("npu"));

            // --- 5. 将结果完美就地塞回 5D Accumulator ---
            Value updatedAccum = b.create<tensor::InsertSliceOp>(loc,
                quantOp.getResult(0), outAccum, offsets, outputSizes, strides);

            b.create<scf::YieldOp>(loc, updatedAccum);
          });

      Value loopResult = nLoop.getResult(0);

      // ==============================================================================
      // 阶段 4: Output Layout Transform (NCHWc32 -> NCHW) - 移入 Region 内部
      // ==============================================================================

      // 【修正 2】在 Region 内部直接使用 rewriter (此时已指向 body 结尾)
      // 分配 4D 最终输出的空间
      Value outputInit = rewriter.create<bufferization::AllocTensorOp>(
          loc, outputType, ValueRange{});

      Operation *unpackOp =
          createLayoutGeneric(rewriter, loc, loopResult, outputInit,
              "npu_layout_nchwc32_to_nchw", N, OC, OH, OW, outTileFactor);

      // 【修正 4】Yield 最终的 4D 结果
      rewriter.create<scf::YieldOp>(loc, unpackOp->getResult(0));
    }

    // Region 外部直接获取结果
    Value finalResult = executeRegion.getResult(0); // 此时是 4D NCHW
    rewriter.replaceOp(quantOp, finalResult);

    rewriter.eraseOp(op);
    for (auto *dqOp : opsToErase)
      if (dqOp->hasOneUse())
        rewriter.eraseOp(dqOp);

    return success();
  }
};
} // namespace

void npux::populateLinalgConvPattern(RewritePatternSet &patterns) {
  patterns.add<ConvToLinalg>(patterns.getContext());
}
