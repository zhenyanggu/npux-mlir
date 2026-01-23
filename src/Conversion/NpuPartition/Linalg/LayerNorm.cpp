//==============================================================
// src/Conversion/NpuPartition/Linalg/LayerNorm.cpp
// This file implements the conversion of LayerNormalization
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

// 通用 Body 构建器 (占位逻辑，实际由 library_call 实现)
static void createLayerNormBody(OpBuilder &b, Location loc, ValueRange args) {
  Value input = args[0];
  Value result = input;
  
  if (mlir::isa<FloatType>(input.getType())) {
    result = b.create<arith::AddFOp>(loc, input, input);
  } else {
    result = b.create<arith::AddIOp>(loc, input, input);
  }
  b.create<linalg::YieldOp>(loc, result);
}

static Value createPackedLayerNormOp(
    ConversionPatternRewriter &rewriter, Location loc,
    Value quantizedInput,    // Int8 Input
    RankedTensorType inputType,
    RankedTensorType outputType,
    double inScale, int64_t inZp, double outScale, int64_t outZp,
    int64_t axis, float epsilon) {

  int64_t rank = inputType.getRank();
  bool isSpatial = (rank == 4);

  // ==========================================================
  // 路径 A: 非 4D 数据 (Flat/Seq) -> 仅 Region 包裹，不 Pack
  // ==========================================================
  if (!isSpatial) {
    // 1. 创建 Region
    auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(loc, outputType);
    
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&executeRegion.getRegion());

      // 2. Region 内部 Alloc
      SmallVector<Value> dynamicSizes =
          getDynamicSizes(rewriter, loc, quantizedInput, inputType.getShape());
      Value regionAlloc = rewriter.create<bufferization::AllocTensorOp>(
          loc, outputType, dynamicSizes);

      // 3. GenericOp (Identity Map)
      SmallVector<AffineMap, 2> indexingMaps = {
          rewriter.getMultiDimIdentityMap(rank),
          rewriter.getMultiDimIdentityMap(rank)
      };
      SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

      auto linalgOp = rewriter.create<linalg::GenericOp>(loc,
          /*resultTypes=*/outputType,
          /*inputs=*/quantizedInput, // 直接使用外部输入
          /*outputs=*/regionAlloc, 
          indexingMaps, iteratorTypes,
          /*bodyBuilder=*/createLayerNormBody);

      // 4. 设置属性
      linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_layernorm"));
      linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
      linalgOp->setAttr("in_scale", rewriter.getF32FloatAttr(inScale));
      linalgOp->setAttr("in_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), inZp));
      linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
      linalgOp->setAttr("out_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), outZp));
      linalgOp->setAttr("axis", rewriter.getI64IntegerAttr(axis));
      linalgOp->setAttr("epsilon", rewriter.getF32FloatAttr(epsilon));

      // 5. Yield
      rewriter.create<scf::YieldOp>(loc, linalgOp.getResults());
    }

    return executeRegion.getResults()[0];
  }

  // ==========================================================
  // 路径 B: 4D 数据 (Spatial) -> Pack + Region + Unpack
  // ==========================================================

  int64_t channelDimPos = 1; // 统一按照 Channel 维度 Pack
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

  // 1. 计算 Packed Shape
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
      
  Value packedInit =
      rewriter.create<tensor::EmptyOp>(loc, packedType, packedDynamicSizes);
  
  Value paddingVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(inputType.getElementType(), inZp));

  auto packOp = rewriter.create<linalg::PackOp>(loc, quantizedInput,
      packedInit, innerDimsPos, innerTilesOFR, paddingVal);

  // --- Execute Region ---
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
        /*bodyBuilder=*/createLayerNormBody);

    // 设置属性
    linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_layernorm"));
    linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    linalgOp->setAttr("in_scale", rewriter.getF32FloatAttr(inScale));
    linalgOp->setAttr("in_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), inZp));
    linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
    linalgOp->setAttr("out_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), outZp));
    linalgOp->setAttr("axis", rewriter.getI64IntegerAttr(axis));
    linalgOp->setAttr("epsilon", rewriter.getF32FloatAttr(epsilon));

    // Yield GenericOp 的结果
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

struct LayerNormToLinalg : public OpConversionPattern<ONNXLayerNormalizationOp> {
  using OpConversionPattern<ONNXLayerNormalizationOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXLayerNormalizationOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    
    // 1. 检查推理模式
    if (!op.getMean().use_empty() || !op.getInvStdDev().use_empty()) {
      return rewriter.notifyMatchFailure(op, "NPU LayerNorm only supports inference (Y output only).");
    }

    // 2. 获取输入 X
    Value originInput = op.getX();
    auto dequantOp = originInput.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantOp) return failure();

    Value quantizedInput = dequantOp.getX();
    auto inputType = mlir::dyn_cast<RankedTensorType>(quantizedInput.getType());
    
    // 修改点: 移除了 Rank!=4 的拦截，允许任意 Rank
    if (!inputType) return failure();

    // 3. 获取输出 Y
    if (!op.getY().hasOneUse()) return failure();
    auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(
        *op.getY().getUsers().begin());
    if (!quantOp) return failure();
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(quantOp.getResult().getType());

    // 4. 获取参数
    auto inParams = getScalarQuantParams(dequantOp);
    auto outParams = getScalarQuantParams(quantOp);
    int64_t axis = op.getAxis();
    float epsilon = op.getEpsilon().convertToFloat();

    // 5. 创建
    Value result = createPackedLayerNormOp(rewriter, op.getLoc(), 
        quantizedInput, inputType, outputType, 
        inParams.scale, inParams.zeroPoint, 
        outParams.scale, outParams.zeroPoint, 
        axis, epsilon);

    // 6. 替换
    rewriter.replaceOp(quantOp, result);
    rewriter.eraseOp(op);
    if (dequantOp->hasOneUse())
      rewriter.eraseOp(dequantOp);

    return success();
  }
};

} // namespace

void npux::populateLinalgLayerNormPattern(RewritePatternSet &patterns) {
  patterns.add<LayerNormToLinalg>(patterns.getContext());
}