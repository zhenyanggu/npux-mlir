//==============================================================
// src/Conversion/NpuPartition/Linalg/LayerNorm.cpp
// This file implements the conversion of LayerNormalization
// to linalg operations for NPU partitioning.
//==============================================================
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h" // 虽然移除了 scf.region，但保留以防其他地方使用
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/NpuPartition/LinalgConversionHelper.hpp"

using namespace mlir;
using namespace npux;

namespace {

// ==========================================================
// 辅助函数：为 RankedTensorType 添加 encoding=1
// ==========================================================
static RankedTensorType addEncoding1(RankedTensorType type, OpBuilder &b) {
  if (type.getEncoding()) {
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(type.getEncoding())) {
      if (intAttr.getInt() == 1) return type;
    }
  }
  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               b.getI64IntegerAttr(1));
}

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
    Operation *sourceOp,
    RankedTensorType inputType,
    RankedTensorType outputType,
    double inScale, int64_t inZp, double outScale, int64_t outZp,
    int64_t axis, float epsilon) {

  int64_t rank = inputType.getRank();

  // 1. 创建带有 encoding=1 的新类型
  RankedTensorType inputType1 = addEncoding1(inputType, rewriter);
  RankedTensorType outputType1 = addEncoding1(outputType, rewriter);

  // 2. 检查输入是否来自 onnx.constant，若是则插入 linalg.copy
  if (quantizedInput.getDefiningOp<ONNXConstantOp>()) {
    SmallVector<Value> copyDynamicSizes =
        getDynamicSizes(rewriter, loc, quantizedInput, inputType1.getShape());
    // 为 copy 的输出申请 encoding=1 的 Tensor
    Value copyAlloc = rewriter.create<bufferization::AllocTensorOp>(
        loc, inputType1, copyDynamicSizes);
    auto copyOp = rewriter.create<linalg::CopyOp>(loc, quantizedInput, copyAlloc);
    // 更新 quantizedInput 为 copy 之后带有 encoding=1 的输出
    quantizedInput = copyOp.getResult(0);
  } else {
    // 强制原地修改外部输入的类型
    quantizedInput.setType(inputType1);
  }

  // 3. 为 LayerNorm 的输出申请 encoding=1 的 Buffer
  SmallVector<Value> dynamicSizes =
      getDynamicSizes(rewriter, loc, quantizedInput, inputType1.getShape());
  Value regionAlloc = rewriter.create<bufferization::AllocTensorOp>(
      loc, outputType1, dynamicSizes);

  // 4. 构建 GenericOp (Identity Map)
  SmallVector<AffineMap, 2> indexingMaps = {
      rewriter.getMultiDimIdentityMap(rank),
      rewriter.getMultiDimIdentityMap(rank)
  };
  SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

  auto linalgOp = rewriter.create<linalg::GenericOp>(loc,
      /*resultTypes=*/outputType1,
      /*inputs=*/quantizedInput, 
      /*outputs=*/regionAlloc, 
      indexingMaps, iteratorTypes,
      /*bodyBuilder=*/createLayerNormBody);

  // 5. 设置属性
  linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_layernorm"));
  linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
  SmallVector<StringRef> fusedOps = {"LayerNormalization"};
  setNpuProfileAttrs(linalgOp, sourceOp, rewriter,
      getNpuProfileLayerName(sourceOp), "compute", fusedOps);
  linalgOp->setAttr("in_scale", rewriter.getF32FloatAttr(inScale));
  linalgOp->setAttr("in_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), inZp));
  linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
  linalgOp->setAttr("out_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), outZp));
  linalgOp->setAttr("axis", rewriter.getI64IntegerAttr(axis));
  linalgOp->setAttr("epsilon", rewriter.getF32FloatAttr(epsilon));

  // 6. 直接返回 LinalgOp 的结果 (移除了 scf.region 的逻辑)
  return linalgOp.getResult(0);
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
        quantizedInput, op, inputType, outputType, 
        inParams.scale, inParams.zeroPoint, 
        outParams.scale, outParams.zeroPoint, 
        axis, epsilon);

    // 6. 替换 (新增：确保下游 consumer 也继承 encoding=1 的 Type)
    quantOp.getResult().setType(result.getType());
    
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
