//==============================================================
//src/Conversion/NpuPartition/Lianlg/Unary.cpp
// this file implements the conversion of unary operations (Gelu, Softmax)
// to linalg operations for NPU partitioning.
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
#include "src/Conversion/NpuPartition/LinalgConversionHelper.hpp"

using namespace mlir;

namespace npux {
static Value createPackedUnaryOp(
    ConversionPatternRewriter &rewriter, Location loc,
    Value quantizedInput,        // 已是 Int8 的输入
    RankedTensorType inputType,  // 输入类型
    RankedTensorType outputType, // 输出类型 (NCHW)
    double inScale, int64_t inZp, double outScale, int64_t outZp,
    StringRef libCallName, // "npu_gelu" 或 "npu_softmax"
    std::function<void(Operation *)> attrHook =
        nullptr // 用于设置额外属性的回调
) {
  int64_t tileFactor = 32;
  int64_t channelDimPos = 1;
  SmallVector<OpFoldResult> innerTilesOFR = {rewriter.getIndexAttr(tileFactor)};
  SmallVector<int64_t> innerDimsPos = {channelDimPos};

  // 1. 计算 Packed Type
  SmallVector<int64_t> packedShape;
  ArrayRef<int64_t> inShape = inputType.getShape();
  if (inputType.hasStaticShape()) {
    for (int i = 0; i < 4; ++i) {
      if (i == channelDimPos)
        packedShape.push_back((inShape[i] + tileFactor - 1) / tileFactor);
      else
        packedShape.push_back(inShape[i]);
    }
    packedShape.push_back(tileFactor);
  } else {
    packedShape = SmallVector<int64_t>(5, ShapedType::kDynamic);
    packedShape[4] = tileFactor;
  }
  auto packedType =
      RankedTensorType::get(packedShape, inputType.getElementType());

  // 2. 创建 Region
  auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(loc, outputType);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&executeRegion.getRegion());

    // --- Pack ---
    SmallVector<Value> packedDynamicSizes =
        getDynamicSizes(rewriter, loc, quantizedInput, inputType.getShape());
    Value packedInit =
        rewriter.create<tensor::EmptyOp>(loc, packedType, packedDynamicSizes);
    Value paddingVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(inputType.getElementType(), inZp));

    auto packOp = rewriter.create<linalg::PackOp>(loc, quantizedInput,
        packedInit, innerDimsPos, innerTilesOFR, paddingVal);

    // --- Compute (Generic) ---
    Value regionAlloc = rewriter.create<bufferization::AllocTensorOp>(
        loc, packedType, packedDynamicSizes);

    SmallVector<AffineMap, 2> indexingMaps = {
        rewriter.getMultiDimIdentityMap(5), rewriter.getMultiDimIdentityMap(5)};
    SmallVector<utils::IteratorType> iteratorTypes(
        5, utils::IteratorType::parallel);

    auto linalgOp = rewriter.create<linalg::GenericOp>(loc,
        /*resultTypes=*/packedType,
        /*inputs=*/packOp.getResult(),
        /*outputs=*/regionAlloc, indexingMaps, iteratorTypes,
        /*bodyBuilder=*/[&](OpBuilder &b, Location loc, ValueRange args) {
          Value input = args[0];
          Value result = input;
          Type elemType = input.getType();

          if (mlir::isa<FloatType>(elemType)) {
            result = b.create<arith::AddFOp>(loc, input, input);
          } else if (mlir::isa<IntegerType>(elemType)) {
            result = b.create<arith::AddIOp>(loc, input, input);
          }

          b.create<linalg::YieldOp>(loc, result);
        });

    // 设置通用属性
    linalgOp->setAttr("library_call", rewriter.getStringAttr(libCallName));
    linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    linalgOp->setAttr("in_scale", rewriter.getF32FloatAttr(inScale));
    linalgOp->setAttr(
        "in_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), inZp));
    linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
    linalgOp->setAttr(
        "out_zp", rewriter.getIntegerAttr(rewriter.getI16Type(), outZp));

    // 设置特定 OP 的属性 (如 Softmax 的 axis)
    if (attrHook)
      attrHook(linalgOp);

    // --- Unpack ---
    SmallVector<Value> unpackDynamicSizes =
        getDynamicSizes(rewriter, loc, quantizedInput, outputType.getShape());
    Value unpackDestInit =
        rewriter.create<tensor::EmptyOp>(loc, outputType, unpackDynamicSizes);

    auto unpackOp = rewriter.create<linalg::UnPackOp>(loc,
        linalgOp.getResults()[0], unpackDestInit, innerDimsPos, innerTilesOFR);

    rewriter.create<scf::YieldOp>(loc, unpackOp->getResults());
  }

  return executeRegion.getResults()[0];
}

// ============================================================================
// 1. Gelu Pattern (复用逻辑)
// ============================================================================
struct GeluToLinalg : public OpConversionPattern<ONNXGeluOp> {
  using OpConversionPattern<ONNXGeluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGeluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value originInput = op.getX();
    auto dequantOp = originInput.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantOp)
      return failure();

    Value quantizedInput = dequantOp.getX();
    auto inputType = mlir::dyn_cast<RankedTensorType>(quantizedInput.getType());
    if (!inputType || inputType.getRank() != 4)
      return failure();

    if (!op.getResult().hasOneUse())
      return failure();
    auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(
        *op.getResult().getUsers().begin());
    if (!quantOp)
      return failure();
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(quantOp.getResult().getType());

    auto inParams = getScalarQuantParams(dequantOp);
    auto outParams = getScalarQuantParams(quantOp);

    Value result = createPackedUnaryOp(rewriter, op.getLoc(), quantizedInput,
        inputType, outputType, inParams.scale, inParams.zeroPoint,
        outParams.scale, outParams.zeroPoint, "npu_gelu");

    rewriter.replaceOp(quantOp, result);
    rewriter.eraseOp(op);
    if (dequantOp->hasOneUse())
      rewriter.eraseOp(dequantOp);
    return success();
  }
};

// ============================================================================
// 2. Softmax Pattern (复用逻辑)
// ============================================================================
struct SoftmaxToLinalg
    : public OpConversionPattern<ONNXSoftmaxOp> { // 假设使用标准 Softmax
  using OpConversionPattern<ONNXSoftmaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXSoftmaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value originInput = op.getInput(); // ONNXSoftmax 通常叫 input
    auto dequantOp = originInput.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantOp)
      return failure();

    Value quantizedInput = dequantOp.getX();
    auto inputType = mlir::dyn_cast<RankedTensorType>(quantizedInput.getType());
    if (!inputType || inputType.getRank() != 4)
      return failure();

    if (!op.getResult().hasOneUse())
      return failure();
    auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(
        *op.getResult().getUsers().begin());
    if (!quantOp)
      return failure();
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(quantOp.getResult().getType());

    auto inParams = getScalarQuantParams(dequantOp);
    auto outParams = getScalarQuantParams(quantOp);

    // 获取 axis 属性
    int64_t axis = op.getAxis();

    Value result =
        createPackedUnaryOp(rewriter, op.getLoc(), quantizedInput, inputType,
            outputType, inParams.scale, inParams.zeroPoint, outParams.scale,
            outParams.zeroPoint, "npu_softmax", [&](Operation *genericOp) {
              // Hook: 设置 Softmax 特有的 axis 属性
              genericOp->setAttr("axis", rewriter.getI64IntegerAttr(axis));
            });

    rewriter.replaceOp(quantOp, result);
    rewriter.eraseOp(op);
    if (dequantOp->hasOneUse())
      rewriter.eraseOp(dequantOp);
    return success();
  }
};
}

void npux::populateLinalgUnaryPatterns(RewritePatternSet &patterns) {
  patterns.add<GeluToLinalg, SoftmaxToLinalg>(patterns.getContext());
}