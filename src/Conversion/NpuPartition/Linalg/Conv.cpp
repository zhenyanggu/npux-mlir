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

  // 辅助函数：获取 ONNX 属性中的 Int 数组，如果不存在则返回默认值
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
    // 默认值：strides=[1,1], dilations=[1,1], pads=[0,0,0,0]
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

    // 2. 准备默认值 (或者从 Hardware Config 获取默认 Conv Tile)
    int64_t t_oh = 1, t_ow = 1, t_ic = 32, t_oc = 32;

    auto defaultConv = npux::NPUConfig::getInstance().getConvTileSize();
    if (defaultConv.size() == 4) { // [t_oh, t_ow, t_ic, t_oc]
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
      // llvm::outs() << "Applied DSE tiling for " << nodeName << "\n";
    }

    // ============================================================
    // 1. 提取输入和参数 (反量化处理保持不变)
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

    if (!op.getResult().hasOneUse())
      return failure();
    Operation *userOp = *op.getResult().getUsers().begin();
    auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(userOp);
    if (!quantOp)
      return failure();

    auto qParams = getScalarQuantParams(quantOp);
    double outScale = qParams.scale;
    int64_t outZp = qParams.zeroPoint;

    auto inputType = mlir::dyn_cast<RankedTensorType>(originInput.getType());
    auto weightType = mlir::dyn_cast<RankedTensorType>(weightInput.getType());
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(quantOp.getResult().getType());

    if (!inputType || !weightType || !outputType)
      return failure();
    int64_t tileFactor = 32;

    // ============================================================
    // 创建 Region
    // ============================================================
    auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(loc, outputType);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&executeRegion.getRegion());

      // --------------------------------------------------------
      // Step 1: Input Packing -> [N, C/32, H, W, 32]
      // --------------------------------------------------------
      int64_t actChannelDim = 1;
      SmallVector<int64_t> packedInputShape;
      ArrayRef<int64_t> inShape = inputType.getShape();
      // 构造 Packed Shape: [N, C_outer, H, W, C_inner]
      if (inputType.hasStaticShape()) {
        packedInputShape.push_back(inShape[0]); // N
        packedInputShape.push_back(
            (inShape[1] + tileFactor - 1) / tileFactor); // C_outer
        packedInputShape.push_back(inShape[2]);          // H
        packedInputShape.push_back(inShape[3]);          // W
        packedInputShape.push_back(tileFactor);          // C_inner
      } else {
        packedInputShape = SmallVector<int64_t>(5, ShapedType::kDynamic);
        packedInputShape[4] = tileFactor;
      }
      auto packedInputType =
          RankedTensorType::get(packedInputShape, inputType.getElementType());

      // 获取动态 Size (略微简化，假设已有 helper)
      SmallVector<Value> inputPackedDynSizes =
          getDynamicSizes(rewriter, loc, originInput, inputType.getShape());
      Value packedInputInit = rewriter.create<tensor::EmptyOp>(
          loc, packedInputType, inputPackedDynSizes);
      Value inputPaddingVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(inputType.getElementType(), inZp));

      auto packInputOp = rewriter.create<linalg::PackOp>(loc, originInput,
          packedInputInit, ArrayRef<int64_t>{actChannelDim},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tileFactor)},
          inputPaddingVal);

      // --------------------------------------------------------
      // Step 2: Weight Packing -> [Out/32, In/32, kH, kW, In_32, Out_32]
      // --------------------------------------------------------
      // 注意：Weight 原始格式为 [Out, In, kH, kW]
      // 这里的 inner_dims_pos 指定为 {1, 0}，意味着先 Pack 维度1 (In)，再 Pack
      // 维度0 (Out) 结果的 Inner 维度顺序会追加为 [In_inner, Out_inner]，即
      // [cin32, cout32]
      SmallVector<int64_t> weightInnerDims = {1, 0};
      SmallVector<OpFoldResult> weightInnerTiles = {
          rewriter.getIndexAttr(tileFactor), rewriter.getIndexAttr(tileFactor)};

      SmallVector<int64_t> packedWeightShape;
      ArrayRef<int64_t> wShape = weightType.getShape();
      // Shape: [Out_outer, In_outer, kH, kW, In_inner, Out_inner]
      packedWeightShape.push_back(
          (wShape[0] + tileFactor - 1) / tileFactor); // Out_outer
      packedWeightShape.push_back(
          (wShape[1] + tileFactor - 1) / tileFactor); // In_outer
      packedWeightShape.push_back(wShape[2]);         // kH
      packedWeightShape.push_back(wShape[3]);         // kW
      packedWeightShape.push_back(tileFactor);        // In_inner
      packedWeightShape.push_back(tileFactor);        // Out_inner

      auto packedWeightType =
          RankedTensorType::get(packedWeightShape, weightType.getElementType());
      Value weightPaddingVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(weightType.getElementType()));
      Value packedWeightInit =
          rewriter.create<tensor::EmptyOp>(loc, packedWeightType, ValueRange{});

      auto packWeightOp =
          rewriter.create<linalg::PackOp>(loc, weightInput, packedWeightInit,
              weightInnerDims, weightInnerTiles, weightPaddingVal);

      // --------------------------------------------------------
      // Step 3: Bias Processing (Broadcast shape)
      // --------------------------------------------------------
      // Bias 通常是 1D [Out]，我们需要将其变为 [Out/32, 32] 或 [Out/32, 1, 1,
      // 32] 以便广播 为了配合 map，我们将其 reshape/pad 为 [Out/32, 32]
      Value processedBias = nullptr;
      if (!mlir::isa<NoneType>(biasInput.getType())) {
        auto biasType = mlir::dyn_cast<RankedTensorType>(biasInput.getType());
        int64_t cDim = biasType.getShape()[0];
        int64_t paddedC = (cDim + tileFactor - 1) / tileFactor * tileFactor;

        // 1. Pad 到 32 的倍数
        Value paddedBias = biasInput;
        if (paddedC > cDim) {
          auto paddedBiasType =
              RankedTensorType::get({paddedC}, biasType.getElementType());
          Value zeroVal = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getZeroAttr(biasType.getElementType()));
          auto padOp = rewriter.create<tensor::PadOp>(loc, paddedBiasType,
              biasInput, SmallVector<OpFoldResult>(1, rewriter.getIndexAttr(0)),
              SmallVector<OpFoldResult>(
                  1, rewriter.getIndexAttr(paddedC - cDim)),
              zeroVal); // nofold
          paddedBias = padOp.getResult();
        }

        // 2. Expand/Reshape 到 [Out/32, 32] (Pack)
        // 这里为了简单，直接使用 linalg.pack 或者 expand_shape。使用 Pack
        // 保持一致性。
        SmallVector<int64_t> packedBiasShape = {
            paddedC / tileFactor, tileFactor};
        auto packedBiasType =
            RankedTensorType::get(packedBiasShape, biasType.getElementType());
        Value packedBiasInit =
            rewriter.create<tensor::EmptyOp>(loc, packedBiasType, ValueRange{});

        // 将 1D Bias Pack 成 2D [Out_outer, Out_inner]
        auto packBiasOp = rewriter.create<linalg::PackOp>(loc, paddedBias,
            packedBiasInit, ArrayRef<int64_t>{0},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tileFactor)});
        processedBias = packBiasOp.getResult();
      }

      // --------------------------------------------------------
      // Step 4: Compute (Linalg Generic) - 9 Loops
      // --------------------------------------------------------
      // 目标 Output Shape: [N, C_out/32, H, W, 32]
      SmallVector<int64_t> packedOutputShape;
      ArrayRef<int64_t> outShape = outputType.getShape();
      if (outputType.hasStaticShape()) {
        packedOutputShape.push_back(outShape[0]); // N
        packedOutputShape.push_back(
            (outShape[1] + tileFactor - 1) / tileFactor); // C_out_outer
        packedOutputShape.push_back(outShape[2]);         // H
        packedOutputShape.push_back(outShape[3]);         // W
        packedOutputShape.push_back(tileFactor);          // C_out_inner
      } else {
        packedOutputShape = SmallVector<int64_t>(5, ShapedType::kDynamic);
        packedOutputShape[4] = tileFactor;
      }
      auto packedOutputType =
          RankedTensorType::get(packedOutputShape, outputType.getElementType());

      SmallVector<Value> outPackedDynSizes = getDynamicSizes(
          rewriter, loc, originInput, inputType.getShape()); // 需适配 output
      Value regionAlloc = rewriter.create<bufferization::AllocTensorOp>(
          loc, packedOutputType, outPackedDynSizes);

      Value paddedInput = packInputOp.getResult();
      auto inputRankedType = mlir::cast<RankedTensorType>(packedInputType);

      // 只有当存在非零 Padding 时才创建 PadOp (优化 IR 大小)
      if (padTop > 0 || padLeft > 0 || pads[2] > 0 || pads[3] > 0) {
        int64_t padBottom = pads[2];
        int64_t padRight = pads[3];
        int rank = 5; // [N, C_outer, H, W, C_inner]

        // 1. 准备 Padding 的边界 (Low, High)
        SmallVector<OpFoldResult> low(rank, rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> high(rank, rewriter.getIndexAttr(0));

        // Pad 在 H(dim 2) 和 W(dim 3) 维度
        low[2] = rewriter.getIndexAttr(padTop);
        low[3] = rewriter.getIndexAttr(padLeft);
        high[2] = rewriter.getIndexAttr(padBottom);
        high[3] = rewriter.getIndexAttr(padRight);

        // 2. 准备 Padding 值 (使用 Input Zero Point)
        // 注意：Input/Weight/Acc 在这里都视作整数处理
        Value padValue = rewriter.create<arith::ConstantOp>(loc,
            rewriter.getIntegerAttr(inputRankedType.getElementType(), inZp));

        // 3. 推导 Pad 后的 Type
        SmallVector<int64_t> paddedShape =
            llvm::to_vector(inputRankedType.getShape());
        // 如果是静态 Shape，手动加上 pad；如果是动态，tensor.pad 会自动推导
        if (paddedShape[2] != ShapedType::kDynamic)
          paddedShape[2] += (padTop + padBottom);
        if (paddedShape[3] != ShapedType::kDynamic)
          paddedShape[3] += (padLeft + padRight);

        auto paddedType = RankedTensorType::get(
            paddedShape, inputRankedType.getElementType());

        // 4. 创建 tensor.pad
        // nofold
        // 属性通常用于阻止某些激进的常量折叠，这里可以不加，视管线情况而定
        auto padOp = rewriter.create<tensor::PadOp>(
            loc, paddedType, paddedInput, low, high, padValue);

        paddedInput = padOp.getResult();
      }

      // ------------------------------------------------------
      // 核心：Affine Maps 定义 (9 Dimensions)
      // ------------------------------------------------------
      // Loops:
      // d0: n (Batch)
      // d1: oc_chunk (Out Channel Outer)
      // d2: oh (Output Height)
      // d3: ow (Output Width)
      // d4: ic_chunk (Input Channel Outer) - Reduction
      // d5: kh (Kernel Height) - Reduction
      // d6: kw (Kernel Width) - Reduction
      // d7: ic_block (Input Channel Inner) - Reduction (对应 Cin32)
      // d8: oc_block (Output Channel Inner) - Parallel  (对应 Cout32)

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
      // H_in = oh * strideH + kh * dilationH - padTop
      // W_in = ow * strideW + kw * dilationW - padLeft
      {
        auto n = rewriter.getAffineDimExpr(0);
        auto ic_chunk = rewriter.getAffineDimExpr(4);
        auto ic_block = rewriter.getAffineDimExpr(7);

        auto oh = rewriter.getAffineDimExpr(2);
        auto kh = rewriter.getAffineDimExpr(5);

        // 错误写法 (当前): auto h_in = oh * strideH + kh * dilationH - padTop;
        // 正确写法 (修改后):
        auto h_in = oh * strideH + kh * dilationH;

        auto ow = rewriter.getAffineDimExpr(3);
        auto kw = rewriter.getAffineDimExpr(6);

        // 错误写法 (当前): auto w_in = ow * strideW + kw * dilationW - padLeft;
        // 正确写法 (修改后):
        auto w_in = ow * strideW + kw * dilationW;

        indexingMaps.push_back(AffineMap::get(nLoops, 0,
            {n, ic_chunk, h_in, w_in, ic_block}, rewriter.getContext()));
      }

      // 2. Weight Map: [OC_outer, IC_outer, KH, KW, IC_inner, OC_inner]
      // 对应之前的 Pack: [Cout/32, Cin/32, k, k, cin32, cout32]
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

      // 3. Bias Map (Optional): [OC_outer, OC_inner] (Broadcast over N, H, W)
      if (processedBias) {
        auto oc_chunk = rewriter.getAffineDimExpr(1);
        auto oc_block = rewriter.getAffineDimExpr(8);
        indexingMaps.push_back(AffineMap::get(
            nLoops, 0, {oc_chunk, oc_block}, rewriter.getContext()));
      }

      // 4. Output Map: [N, OC_outer, H_out, W_out, OC_inner]
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
            // args: [input, weight, (bias), output_accumulator]
            // 最后一个 arg 是 accumulator (因为有 reduction iterator)
            Value acc = args.back();
            Value input = args[0];
            Value weight = args[1];

            // 构造一个 Dummy 的 MAC (Multiply-Accumulate) 或者简单的 Add
            // 真正的逻辑是 acc + input * weight，这里为了占位，简单做加法即可
            // 只要 acc 被改变了，这个 reduction loop 就不会被删掉。

            Value update;
            if (mlir::isa<FloatType>(acc.getType())) {
              Value sum = b.create<arith::AddFOp>(loc, input, weight);
              if (processedBias && args.size() > 3) { // 有 bias
                sum = b.create<arith::AddFOp>(loc, sum, args[2]);
              }
              update = b.create<arith::AddFOp>(loc, acc, sum);
            } else {
              // Integer Logic
              // -----------------------------------------------------------
              // FIX: arith.addi 要求操作数类型严格一致。
              // Bias 可能是 i32，而 Input/Weight/Acc 可能是 i8。
              // 我们需要定义一个 helper 将所有值 cast 到 acc 的类型。
              // -----------------------------------------------------------
              Type accType = acc.getType();

              auto castToAccType = [&](Value val) -> Value {
                Type valType = val.getType();
                if (valType == accType)
                  return val;

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

      // 设置所有属性供 CAPI 使用
      linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_conv"));
      linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
      auto copyAttr = [&](StringRef name) {
        if (auto attr = op->getAttr(name))
          linalgOp->setAttr(name, attr);
      };
      copyAttr("pads");
      copyAttr("dilations");
      copyAttr("strides");
      copyAttr("group");
      copyAttr("kernel_shape");
      copyAttr("auto_pad");
      linalgOp->setAttr("in_scale", rewriter.getF32FloatAttr(inScale));
      linalgOp->setAttr(
          "in_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), inZp));
      linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
      linalgOp->setAttr(
          "out_zp", rewriter.getIntegerAttr(rewriter.getI16Type(), outZp));
      if (isCustomized) {
        // 写入一个 Attribute 标记，供后端 Tiling Pass 使用
        SmallVector<int64_t, 4> bestTile = {t_oh, t_ow, t_ic, t_oc};
        linalgOp->setAttr("npu.dse_tiling", rewriter.getI64ArrayAttr(bestTile));
      }

      // 调试用：把名字也写进去方便看 IR
      if (!nodeName.empty()) {
        linalgOp->setAttr("npu.layer_name", rewriter.getStringAttr(nodeName));
      }

      // --------------------------------------------------------
      // Step 5: Output Unpack
      // --------------------------------------------------------
      SmallVector<Value> unpackDynamicSizes =
          getDynamicSizes(rewriter, loc, originInput, outputType.getShape());
      Value unpackDestInit =
          rewriter.create<tensor::EmptyOp>(loc, outputType, unpackDynamicSizes);

      auto unpackOp = rewriter.create<linalg::UnPackOp>(loc,
          linalgOp.getResults()[0], unpackDestInit, ArrayRef<int64_t>{1},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tileFactor)});

      rewriter.create<scf::YieldOp>(loc, unpackOp->getResults());
    }

    rewriter.replaceOp(quantOp, executeRegion.getResults());

    rewriter.eraseOp(op);
    for (auto *dqOp : opsToErase)
      if (dqOp->use_empty())
        rewriter.eraseOp(dqOp);

    return success();
  }
};
}

void npux::populateLinalgConvPattern(RewritePatternSet &patterns) {
  patterns.add<ConvToLinalg>(patterns.getContext());
}