//==============================================================
//src/Conversion/NpuPartition/Linalg/LayerNorm.cpp
// this file implements the conversion of LayerNormalization
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
using namespace npux; // 假设 helper 在此命名空间

namespace {

// 复用 Unary.cpp 中的逻辑结构，针对 LayerNorm 进行适配
// 保持与 Gelu/Softmax 一致的 Channel-First Packing 策略 (NCHW -> NCHwc)
static Value createPackedLayerNormOp(
    ConversionPatternRewriter &rewriter, Location loc,
    Value quantizedInput,    // Int8 Input
    RankedTensorType inputType,
    RankedTensorType outputType,
    double inScale, int64_t inZp, double outScale, int64_t outZp,
    int64_t axis, float epsilon) {

  int64_t tileFactor = 32;
  int64_t channelDimPos = 1; // 统一按照 Channel 维度 Pack，与 Gelu/Softmax 保持一致
  SmallVector<OpFoldResult> innerTilesOFR = {rewriter.getIndexAttr(tileFactor)};
  SmallVector<int64_t> innerDimsPos = {channelDimPos};

  // 1. 计算 Packed Shape
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
  auto packedType = RankedTensorType::get(packedShape, inputType.getElementType());

  // 2. 创建 SCF Region
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
    // LayerNorm 这里被视为单输入单输出算子 (忽略 Scale/Bias 输入，由库函数内部处理或忽略)
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
          // Body 仅做占位，实际计算由 library_call 定义
          if (mlir::isa<FloatType>(input.getType())) {
             result = b.create<arith::AddFOp>(loc, input, input); 
          } else {
             result = b.create<arith::AddIOp>(loc, input, input);
          }
          b.create<linalg::YieldOp>(loc, result);
        });

    // 设置通用 NPU 属性
    linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_layernorm"));
    linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    
    // 量化参数
    linalgOp->setAttr("in_scale", rewriter.getF32FloatAttr(inScale));
    linalgOp->setAttr("in_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), inZp));
    linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
    linalgOp->setAttr("out_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), outZp));

    // LayerNorm 特有属性
    linalgOp->setAttr("axis", rewriter.getI64IntegerAttr(axis));
    linalgOp->setAttr("epsilon", rewriter.getF32FloatAttr(epsilon));

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

struct LayerNormToLinalg : public OpConversionPattern<ONNXLayerNormalizationOp> {
  using OpConversionPattern<ONNXLayerNormalizationOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXLayerNormalizationOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    
    // 1. 检查推理模式 (Mean 和 InvStdDev 必须未被使用)
    if (!op.getMean().use_empty() || !op.getInvStdDev().use_empty()) {
      return rewriter.notifyMatchFailure(op, "NPU LayerNorm only supports inference (Y output only).");
    }

    // 2. 获取输入 X 并解包 Dequant (忽略 Scale 和 B 输入)
    Value originInput = op.getX();
    auto dequantOp = originInput.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantOp)
      return failure();

    Value quantizedInput = dequantOp.getX();
    auto inputType = mlir::dyn_cast<RankedTensorType>(quantizedInput.getType());
    if (!inputType || inputType.getRank() != 4)
      return failure();

    // 3. 获取输出 Y 并检查 Quant
    if (!op.getY().hasOneUse())
      return failure();
    auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(
        *op.getY().getUsers().begin());
    if (!quantOp)
      return failure();
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(quantOp.getResult().getType());

    // 4. 获取量化参数
    auto inParams = getScalarQuantParams(dequantOp);
    auto outParams = getScalarQuantParams(quantOp);

    // 5. 获取 LayerNorm 属性
    int64_t axis = op.getAxis();
    float epsilon = op.getEpsilon().convertToFloat(); // 获取 float 值

    // 6. 创建 Packed Op (类似于 Gelu/Softmax 的单输入结构)
    Value result = createPackedLayerNormOp(rewriter, op.getLoc(), 
        quantizedInput, inputType, outputType, 
        inParams.scale, inParams.zeroPoint, 
        outParams.scale, outParams.zeroPoint, 
        axis, epsilon);

    // 7. 替换和清理
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