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
using namespace npux;

namespace {

// 通用的 Body 构建器，用于填充 linalg.generic 的 Region
static void createUnaryBody(OpBuilder &b, Location loc, ValueRange args) {
  Value input = args[0];
  Value result = input;
  Type elemType = input.getType();

  // 这里的计算逻辑通常被 library_call 覆盖，但在 IR 层面必须合法
  if (mlir::isa<FloatType>(elemType)) {
    result = b.create<arith::AddFOp>(loc, input, input);
  } else if (mlir::isa<IntegerType>(elemType)) {
    result = b.create<arith::AddIOp>(loc, input, input);
  }

  b.create<linalg::YieldOp>(loc, result);
}

static Value createPackedUnaryOp(
    ConversionPatternRewriter &rewriter, Location loc,
    Value quantizedInput,            // 已是 Int8 的输入
    RankedTensorType inputType,      // 输入类型
    RankedTensorType outputType,     // 输出类型
    double inScale, int64_t inZp, double outScale, int64_t outZp,
    StringRef libCallName,           // "npu_gelu" 或 "npu_softmax"
    std::function<void(Operation *)> attrHook = nullptr
) {
  int64_t rank = inputType.getRank();
  bool isSpatial = (rank == 4);

  // ==========================================================
  // 路径 A: 非 4D 数据 (Flat/Vector)
  // 不需要 Pack/Unpack，但仍需包裹在 scf.execute_region 中
  // ==========================================================
  if (!isSpatial) {
    // 1. 创建 execute_region，返回类型就是最终的 outputType
    auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(loc, outputType);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&executeRegion.getRegion());

      // 2. 在 Region 内部创建输出 Buffer
      SmallVector<Value> dynamicSizes =
          getDynamicSizes(rewriter, loc, quantizedInput, inputType.getShape());
      Value regionAlloc = rewriter.create<bufferization::AllocTensorOp>(
          loc, outputType, dynamicSizes);

      // 3. 构建 Indexing Maps (Identity)
      SmallVector<AffineMap, 2> indexingMaps = {
          rewriter.getMultiDimIdentityMap(rank),
          rewriter.getMultiDimIdentityMap(rank)
      };
      SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

      // 4. 创建 GenericOp (直接使用外部的 quantizedInput)
      auto linalgOp = rewriter.create<linalg::GenericOp>(loc,
          /*resultTypes=*/outputType,
          /*inputs=*/quantizedInput,
          /*outputs=*/regionAlloc,
          indexingMaps, iteratorTypes,
          /*bodyBuilder=*/createUnaryBody);

      // 5. 设置属性
      linalgOp->setAttr("library_call", rewriter.getStringAttr(libCallName));
      linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
      linalgOp->setAttr("in_scale", rewriter.getF32FloatAttr(inScale));
      linalgOp->setAttr("in_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), inZp));
      linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
      linalgOp->setAttr("out_zp", rewriter.getIntegerAttr(rewriter.getI16Type(), outZp));

      if (attrHook) attrHook(linalgOp);

      // 6. Yield 结果
      rewriter.create<scf::YieldOp>(loc, linalgOp.getResults());
    }

    // 直接返回 Region 的结果
    return executeRegion.getResults()[0];
  }

  // ==========================================================
  // 路径 B: 4D 数据 (Spatial)
  // 需要 Pack -> Region -> Unpack
  // ==========================================================
  
  int64_t channelDimPos = 1;
  ArrayRef<int64_t> inShape = inputType.getShape();
  int64_t inputChannel = inShape[channelDimPos];
  
  // Tile Factor Calculation
  int64_t tileFactor = 32;
  bool isSmallChannel = (inputChannel != ShapedType::kDynamic) && (inputChannel < 32);
  if (isSmallChannel) {
    tileFactor = inputChannel;
  }

  SmallVector<OpFoldResult> innerTilesOFR = {rewriter.getIndexAttr(tileFactor)};
  SmallVector<int64_t> innerDimsPos = {channelDimPos};

  // 1. 计算 Packed Type
  SmallVector<int64_t> packedShape;
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
  
  auto packedType = RankedTensorType::get(packedShape, inputType.getElementType());

  // --- Pack (Outside Region) ---
  SmallVector<Value> packedDynamicSizes =
      getDynamicSizes(rewriter, loc, quantizedInput, inputType.getShape());
      
  Value packedInit = rewriter.create<tensor::EmptyOp>(loc, packedType, packedDynamicSizes);
  Value paddingVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(inputType.getElementType(), inZp));

  auto packOp = rewriter.create<linalg::PackOp>(loc, quantizedInput,
      packedInit, innerDimsPos, innerTilesOFR, paddingVal);


  // --- Execute Region (处理 Pack 后的数据) ---
  auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(loc, packedType);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&executeRegion.getRegion());

    // --- Compute (Generic) Inside Region ---
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
        /*bodyBuilder=*/createUnaryBody);

    // 设置属性
    linalgOp->setAttr("library_call", rewriter.getStringAttr(libCallName));
    linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    linalgOp->setAttr("in_scale", rewriter.getF32FloatAttr(inScale));
    linalgOp->setAttr("in_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), inZp));
    linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
    linalgOp->setAttr("out_zp", rewriter.getIntegerAttr(rewriter.getI16Type(), outZp));

    if (attrHook) attrHook(linalgOp);

    rewriter.create<scf::YieldOp>(loc, linalgOp.getResults());
  }

  // --- Unpack (Outside Region) ---
  SmallVector<Value> unpackDynamicSizes =
      getDynamicSizes(rewriter, loc, quantizedInput, outputType.getShape());
  Value unpackDestInit =
      rewriter.create<tensor::EmptyOp>(loc, outputType, unpackDynamicSizes);

  auto unpackOp = rewriter.create<linalg::UnPackOp>(loc,
      executeRegion.getResults()[0], unpackDestInit, innerDimsPos, innerTilesOFR);

  return unpackOp.getResult();
}

// ... (Patterns 代码保持不变，如下) ...

// ============================================================================
// 1. Gelu Pattern
// ============================================================================
struct GeluToLinalg : public OpConversionPattern<ONNXGeluOp> {
  using OpConversionPattern<ONNXGeluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGeluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value originInput = op.getX();
    auto dequantOp = originInput.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantOp) return failure();

    Value quantizedInput = dequantOp.getX();
    auto inputType = mlir::dyn_cast<RankedTensorType>(quantizedInput.getType());
    if (!inputType) return failure(); // 移除 Rank 检查

    if (!op.getResult().hasOneUse()) return failure();
    auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(
        *op.getResult().getUsers().begin());
    if (!quantOp) return failure();
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(quantOp.getResult().getType());

    auto inParams = getScalarQuantParams(dequantOp);
    auto outParams = getScalarQuantParams(quantOp);

    Value result = createPackedUnaryOp(rewriter, op.getLoc(), quantizedInput,
        inputType, outputType, inParams.scale, inParams.zeroPoint,
        outParams.scale, outParams.zeroPoint, "npu_gelu");

    rewriter.replaceOp(quantOp, result);
    rewriter.eraseOp(op);
    if (dequantOp->hasOneUse()) rewriter.eraseOp(dequantOp);
    return success();
  }
};

// ============================================================================
// 2. Softmax Pattern
// ============================================================================
struct SoftmaxToLinalg
    : public OpConversionPattern<ONNXSoftmaxOp> { 
  using OpConversionPattern<ONNXSoftmaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXSoftmaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value originInput = op.getInput(); 
    auto dequantOp = originInput.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantOp) return failure();

    Value quantizedInput = dequantOp.getX();
    auto inputType = mlir::dyn_cast<RankedTensorType>(quantizedInput.getType());
    if (!inputType) return failure(); // 移除 Rank 检查

    if (!op.getResult().hasOneUse()) return failure();
    auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(
        *op.getResult().getUsers().begin());
    if (!quantOp) return failure();
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(quantOp.getResult().getType());

    auto inParams = getScalarQuantParams(dequantOp);
    auto outParams = getScalarQuantParams(quantOp);

    int64_t axis = op.getAxis();

    Value result =
        createPackedUnaryOp(rewriter, op.getLoc(), quantizedInput, inputType,
            outputType, inParams.scale, inParams.zeroPoint, outParams.scale,
            outParams.zeroPoint, "npu_softmax", [&](Operation *genericOp) {
              genericOp->setAttr("axis", rewriter.getI64IntegerAttr(axis));
            });

    rewriter.replaceOp(quantOp, result);
    rewriter.eraseOp(op);
    if (dequantOp->hasOneUse()) rewriter.eraseOp(dequantOp);
    return success();
  }
};
}

void npux::populateLinalgUnaryPatterns(RewritePatternSet &patterns) {
  patterns.add<GeluToLinalg, SoftmaxToLinalg>(patterns.getContext());
}