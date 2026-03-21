//==============================================================
// src/Conversion/NpuPartition/Linalg/Binary.cpp
// This file implements the conversion of binary operations (Add)
// to linalg operations for NPU partitioning.
//==============================================================

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

// ==========================================================
// Binary 专用的 Body 构建器，用于填充 linalg.generic 的 Region
// ==========================================================
static void createBinaryBody(OpBuilder &b, Location loc, ValueRange args) {
  Value lhs = args[0];
  Value rhs = args[1];
  Type elemType = lhs.getType();
  Value result;

  // 这里的计算逻辑通常被 library_call 覆盖，但在 IR 层面必须合法
  if (mlir::isa<FloatType>(elemType)) {
    result = b.create<arith::AddFOp>(loc, lhs, rhs);
  } else if (mlir::isa<IntegerType>(elemType)) {
    result = b.create<arith::AddIOp>(loc, lhs, rhs);
  } else {
    // 降级保护，默认直接 yield lhs (虽然实际会被 NPU lib 覆盖)
    result = lhs; 
  }

  b.create<linalg::YieldOp>(loc, result);
}

// ==========================================================
// Binary 专用的 GenericOp 包装器
// ==========================================================
static Value createPackedBinaryOp(
    ConversionPatternRewriter &rewriter, Location loc,
    Value quantizedInputA, RankedTensorType inputTypeA,
    Value quantizedInputB, RankedTensorType inputTypeB,
    RankedTensorType outputType,
    double in1Scale, int64_t in1Zp, 
    double in2Scale, int64_t in2Zp, 
    double outScale, int64_t outZp,
    StringRef libCallName, Operation *sourceOp, StringRef layerName,
    ArrayRef<StringRef> fusedOps,
    std::function<void(Operation *)> attrHook = nullptr) {
  
  int64_t rank = inputTypeA.getRank();
  
  // 直接创建带有 Encoding=1 的新类型
  RankedTensorType inputTypeA1 = addEncoding1(inputTypeA, rewriter);
  RankedTensorType inputTypeB1 = addEncoding1(inputTypeB, rewriter);
  RankedTensorType outputType1 = addEncoding1(outputType, rewriter);

  // --- 新增: 处理输入的 Lambda，如果是 Constant 则插入 linalg.copy ---
  auto processInput = [&](Value input, RankedTensorType targetType) -> Value {
    if (input.getDefiningOp<ONNXConstantOp>()) {
      SmallVector<Value> dynSizes = getDynamicSizes(rewriter, loc, input, targetType.getShape());
      Value alloc = rewriter.create<bufferization::AllocTensorOp>(loc, targetType, dynSizes);
      auto copyOp = rewriter.create<linalg::CopyOp>(loc, input, alloc);
      return copyOp.getResult(0);
    }
    // 非 Constant 情况保持原有逻辑，强制原地修改类型
    input.setType(targetType);
    return input;
  };

  // 应用修改逻辑
  quantizedInputA = processInput(quantizedInputA, inputTypeA1);
  quantizedInputB = processInput(quantizedInputB, inputTypeB1);

  // 1. 创建带有 encoding=1 的输出 Buffer
  SmallVector<Value> dynamicSizes =
      getDynamicSizes(rewriter, loc, quantizedInputA, inputTypeA1.getShape());
  Value regionAlloc = rewriter.create<bufferization::AllocTensorOp>(
      loc, outputType1, dynamicSizes);

  // 2. 构建 Indexing Maps (两个输入 + 一个输出，全部为 Identity)
  SmallVector<AffineMap, 3> indexingMaps = {
      rewriter.getMultiDimIdentityMap(rank), // lhs
      rewriter.getMultiDimIdentityMap(rank), // rhs
      rewriter.getMultiDimIdentityMap(rank)  // result
  };
  SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

  // 3. 创建 GenericOp
  auto linalgOp = rewriter.create<linalg::GenericOp>(loc,
      /*resultTypes=*/outputType1,
      /*inputs=*/ValueRange{quantizedInputA, quantizedInputB},
      /*outputs=*/regionAlloc,
      indexingMaps, iteratorTypes,
      /*bodyBuilder=*/createBinaryBody);

  // 4. 设置属性
  linalgOp->setAttr("library_call", rewriter.getStringAttr(libCallName));
  linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
  setNpuProfileAttrs(
      linalgOp, sourceOp, rewriter, layerName, "compute", fusedOps);
  // 针对双输入，分别记录 scale 和 zp
  linalgOp->setAttr("in1_scale", rewriter.getF32FloatAttr(in1Scale));
  linalgOp->setAttr("in1_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), in1Zp));
  linalgOp->setAttr("in2_scale", rewriter.getF32FloatAttr(in2Scale));
  linalgOp->setAttr("in2_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), in2Zp));
  linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
  linalgOp->setAttr("out_zp", rewriter.getIntegerAttr(rewriter.getI16Type(), outZp));

  if (attrHook) attrHook(linalgOp);

  return linalgOp.getResult(0);
}

// ============================================================================
// Add Pattern (No Broadcasting)
// ============================================================================
struct AddToLinalg : public OpConversionPattern<ONNXAddOp> {
  using OpConversionPattern<ONNXAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    // 1. 获取两端输入并寻找量化 / 反量化节点
    Value originInputA = op.getA();
    Value originInputB = op.getB();
    auto dequantOpA = originInputA.getDefiningOp<ONNXDequantizeLinearOp>();
    auto dequantOpB = originInputB.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantOpA || !dequantOpB) return failure();

    Value quantizedInputA = dequantOpA.getX();
    Value quantizedInputB = dequantOpB.getX();
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(quantizedInputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(quantizedInputB.getType());
    if (!inputTypeA || !inputTypeB) return failure();

    // 2. 严格检查广播：如果 shape 不完全一致，拒绝转换
    if (inputTypeA.getShape() != inputTypeB.getShape()) {
        return failure(); 
    }

    // 3. 获取输出端并寻找量化节点
    if (!op.getResult().hasOneUse()) return failure();
    auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(
        *op.getResult().getUsers().begin());
    if (!quantOp) return failure();
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(quantOp.getResult().getType());

    // 4. 提取量化参数
    auto inParamsA = getScalarQuantParams(dequantOpA);
    auto inParamsB = getScalarQuantParams(dequantOpB);
    auto outParams = getScalarQuantParams(quantOp);

    // 5. 创建 linalgOp，传入 "npu_matadd"
    SmallVector<StringRef> fusedOps = {"Add"};
    Value result = createPackedBinaryOp(rewriter, op.getLoc(), 
        quantizedInputA, inputTypeA, 
        quantizedInputB, inputTypeB, 
        outputType, 
        inParamsA.scale, inParamsA.zeroPoint, 
        inParamsB.scale, inParamsB.zeroPoint,
        outParams.scale, outParams.zeroPoint, 
        "npu_matadd", op, getNpuProfileLayerName(op), fusedOps);

    quantOp.getResult().setType(result.getType());

    // 6. 替换和清理旧 Ops
    rewriter.replaceOp(quantOp, result);
    rewriter.eraseOp(op);
    if (dequantOpA->hasOneUse()) rewriter.eraseOp(dequantOpA);
    if (dequantOpB->hasOneUse()) rewriter.eraseOp(dequantOpB);
    
    return success();
  }
};
} // namespace

void npux::populateLinalgBinaryPatterns(RewritePatternSet &patterns) {
  patterns.add<AddToLinalg>(patterns.getContext());
}
