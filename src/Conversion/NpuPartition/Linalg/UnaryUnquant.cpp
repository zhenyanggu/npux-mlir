//==============================================================
// src/Conversion/NpuPartition/Lianlg/UnaryUnquant.cpp
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
#include "src/Conversion/NpuPartition/LinalgConversionHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace npux;

namespace {

// ============================================================================
// 1. 辅助逻辑：量化上下文处理 (已极简优化)
// ============================================================================

struct QuantizedContext {
  Value finalInput;             // 最终传给 NPU Op 的输入 (Int8)
  RankedTensorType finalOutputType; // NPU Op 的输出类型 (Int8)
  
  // 回调函数：处理输出替换
  std::function<void(Value npuResult, PatternRewriter &rewriter)> handleOutputReplacement;
};

// 核心逻辑：严格匹配上下游 qdq，没有就不处理，有就直接吸收转 Int8
static LogicalResult handleQuantizationContext(
    Operation *op, 
    Value originalInput, 
    RankedTensorType originalOutputType,
    PatternRewriter &rewriter,
    QuantizedContext &ctx) 
{
  // 1. 检查上游是否有 DequantizeLinearOp
  auto dequantOp = originalInput.getDefiningOp<ONNXDequantizeLinearOp>();
  if (!dequantOp) {
    return failure(); // 没有上游 dq，不处理
  }

  // 2. 检查下游是否有且仅有一个 QuantizeLinearOp
  if (!op->getResult(0).hasOneUse()) {
    return failure(); // 简单起见，如果下游有多个 use 且不全是 q，不处理
  }
  auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(*op->getResult(0).getUsers().begin());
  if (!quantOp) {
    return failure(); // 没有下游 q，不处理
  }

  // 3. 匹配成功 (int8 -> dq -> op -> q -> int8)
  // 吸收上游 dq：直接拿 dequant 之前的 Int8 作为输入
  ctx.finalInput = dequantOp.getX(); 
  
  // 吸收下游 q：直接拿 quant 之后的类型作为 NPU Op 的输出类型
  ctx.finalOutputType = cast<RankedTensorType>(quantOp.getResult().getType());
  
  ctx.handleOutputReplacement = [=](Value npuResult, PatternRewriter &b) {
    // 替换下游的 QuantOp 为 NPU 生成的直通 Int8 结果
    b.replaceOp(quantOp, npuResult);
    // 删除当前的 Resample/Transpose Op
    b.eraseOp(op);
    // 尝试清理上游的 DequantOp（如果它没有其他使用者了）
    if (dequantOp->use_empty()) {
      b.eraseOp(dequantOp);
    }
  };

  return success();
}

// ============================================================================
// 2. Linalg 构建逻辑 (保持不变)
// ============================================================================

static void createLinalgBody(OpBuilder &b, Location loc, ValueRange args) {
  Value input = args[0];
  Value result = input; 
  Type elemType = input.getType();

  if (mlir::isa<FloatType>(elemType)) {
    result = b.create<arith::AddFOp>(loc, input, input);
  } else if (mlir::isa<IntegerType>(elemType)) {
    result = b.create<arith::AddIOp>(loc, input, input);
  }

  b.create<linalg::YieldOp>(loc, result);
}

static Value createResampleOp(
    ConversionPatternRewriter &rewriter, Location loc,
    Value input,                 
    RankedTensorType inputType,  
    RankedTensorType outputType, 
    StringRef libCallName,       
    AffineMap inputMap,          
    AffineMap outputMap,         
    std::function<void(Operation *)> attrHook = nullptr 
) {
  auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(loc, outputType);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&executeRegion.getRegion());

    SmallVector<Value> dynamicSizes =
        getDynamicSizes(rewriter, loc, input, outputType.getShape());
    Value regionAlloc = rewriter.create<bufferization::AllocTensorOp>(
        loc, outputType, dynamicSizes);

    SmallVector<AffineMap, 2> indexingMaps = {inputMap, outputMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        outputType.getRank(), utils::IteratorType::parallel);

    auto linalgOp = rewriter.create<linalg::GenericOp>(loc,
        outputType, input, regionAlloc, indexingMaps, iteratorTypes,
        createLinalgBody);

    linalgOp->setAttr("library_call", rewriter.getStringAttr(libCallName));
    linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    if (attrHook) attrHook(linalgOp);

    rewriter.create<scf::YieldOp>(loc, linalgOp.getResults());
  }
  return executeRegion.getResults()[0];
}

// ============================================================================
// 3. MaxPool Pattern (保持不变，依赖新的 context 逻辑)
// ============================================================================
struct MaxPoolToLinalg : public OpConversionPattern<ONNXMaxPoolSingleOutOp> {
  using OpConversionPattern<ONNXMaxPoolSingleOutOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto kernelShape = op.getKernelShape();
    auto strides = op.getStrides();
    if (!kernelShape || !strides) return failure();

    Value input = op.getX();
    auto outputType = mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!outputType) return failure();
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() != 4 || outputType.getRank() != 4) {
      op.emitWarning() << "ONNXMaxPool lowering to NPU resample requires NCHW (4D) tensors.";
      return failure();
    }

    QuantizedContext ctx;
    if (failed(handleQuantizationContext(op, input, outputType, rewriter, ctx))) {
        return failure();
    }

    int64_t rank = outputType.getRank();
    SmallVector<AffineExpr> inputExprs;
    for (int i = 0; i < rank; ++i) {
      auto expr = rewriter.getAffineDimExpr(i);
      if (i >= 2) {
        expr = expr * 2;
      }
      inputExprs.push_back(expr);
    }

    auto inputMap = AffineMap::get(rank, 0, inputExprs, rewriter.getContext());
    auto outputMap = rewriter.getMultiDimIdentityMap(rank);

    Value result = createResampleOp(rewriter, op.getLoc(), 
        ctx.finalInput, 
        mlir::cast<RankedTensorType>(ctx.finalInput.getType()), 
        ctx.finalOutputType, 
        "npu_maxpool", inputMap, outputMap);

    ctx.handleOutputReplacement(result, rewriter);
    return success();
  }
};

// ============================================================================
// 4. AveragePool Pattern (保持不变，依赖新的 context 逻辑)
// ============================================================================
struct AveragePoolToLinalg : public OpConversionPattern<ONNXAveragePoolOp> {
  using OpConversionPattern<ONNXAveragePoolOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXAveragePoolOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    op.emitWarning() << "NPU backend does not support AveragePool currently. Skip NPU lowering for this op.";
    return failure();

    Value input = op.getX();
    auto outputType = mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!outputType) return failure();
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() != 4 || outputType.getRank() != 4) {
      op.emitWarning() << "ONNXAveragePool lowering to NPU resample requires NCHW (4D) tensors.";
      return failure();
    }

    QuantizedContext ctx;
    if (failed(handleQuantizationContext(op, input, outputType, rewriter, ctx))) {
        return failure();
    }

    int64_t rank = outputType.getRank();
    SmallVector<AffineExpr> inputExprs;
    for (int i = 0; i < rank; ++i) {
      auto expr = rewriter.getAffineDimExpr(i);
      if (i >= 2) { 
        expr = expr * 2;
      }
      inputExprs.push_back(expr);
    }

    auto inputMap = AffineMap::get(rank, 0, inputExprs, rewriter.getContext());
    auto outputMap = rewriter.getMultiDimIdentityMap(rank);

    Value result = createResampleOp(rewriter, op.getLoc(), 
        ctx.finalInput, 
        mlir::cast<RankedTensorType>(ctx.finalInput.getType()), 
        ctx.finalOutputType, 
        "npu_avgpool", inputMap, outputMap);

    ctx.handleOutputReplacement(result, rewriter);
    return success();
  }
};

// ============================================================================
// 5. Resize Pattern (保持不变，依赖新的 context 逻辑)
// ============================================================================
struct ResizeToLinalg : public OpConversionPattern<ONNXResizeOp> {
  using OpConversionPattern<ONNXResizeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXResizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    StringRef mode = op.getMode();
    if (mode != "nearest") return failure();

    Value input = op.getX();
    auto outputType = mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!outputType) return failure();
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() != 4 || outputType.getRank() != 4) {
      op.emitWarning() << "ONNXResize lowering to NPU resample requires NCHW (4D) tensors.";
      return failure();
    }

    QuantizedContext ctx;
    if (failed(handleQuantizationContext(op, input, outputType, rewriter, ctx))) {
        return failure();
    }

    int64_t rank = outputType.getRank();
    SmallVector<AffineExpr> inputExprs;
    for (int i = 0; i < rank; ++i) {
      auto expr = rewriter.getAffineDimExpr(i);
      if (i >= 2) { 
        expr = expr.floorDiv(2);
      }
      inputExprs.push_back(expr);
    }

    auto inputMap = AffineMap::get(rank, 0, inputExprs, rewriter.getContext());
    auto outputMap = rewriter.getMultiDimIdentityMap(rank);

    Value result = createResampleOp(rewriter, op.getLoc(), 
        ctx.finalInput, 
        mlir::cast<RankedTensorType>(ctx.finalInput.getType()), 
        ctx.finalOutputType, 
        "npu_upsample", inputMap, outputMap);

    ctx.handleOutputReplacement(result, rewriter);
    return success();
  }
};

// ============================================================================
// 6. Transpose Pattern (保持不变，依赖新的 context 逻辑)
// ============================================================================
struct TransposeToLinalg : public OpConversionPattern<ONNXTransposeOp> {
  using OpConversionPattern<ONNXTransposeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXTransposeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Value input = op.getData();
    auto outputType = mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!outputType) return failure();

    QuantizedContext ctx;
    if (failed(handleQuantizationContext(op, input, outputType, rewriter, ctx))) {
        return failure();
    }

    int64_t rank = outputType.getRank();
    SmallVector<int64_t> perm;
    if (auto permAttr = op.getPermAttr()) {
      for (auto p : permAttr) perm.push_back(mlir::cast<IntegerAttr>(p).getInt());
    } else {
      for (int i = 0; i < rank; ++i) perm.push_back(rank - 1 - i);
    }

    auto inputMap = rewriter.getMultiDimIdentityMap(rank);
    auto outputMap = AffineMap::getPermutationMap(perm, rewriter.getContext());

    SmallVector<AffineMap, 2> indexingMaps = {inputMap, outputMap};
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(op.getLoc(), ctx.finalOutputType);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&executeRegion.getRegion());

      SmallVector<Value> dynSizes = getDynamicSizes(rewriter, op.getLoc(), ctx.finalInput, ctx.finalOutputType.getShape());
      Value resultInit = rewriter.create<bufferization::AllocTensorOp>(op.getLoc(), ctx.finalOutputType, dynSizes);

      auto linalgOp = rewriter.create<linalg::GenericOp>(op.getLoc(),
          ctx.finalOutputType, ctx.finalInput, resultInit, indexingMaps, iteratorTypes,
          createLinalgBody); 

      linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_transpose"));
      linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

      rewriter.create<scf::YieldOp>(op.getLoc(), linalgOp.getResults());
    }

    ctx.handleOutputReplacement(executeRegion.getResults()[0], rewriter);
    return success();
  }
};

} // namespace

void npux::populateLinalgResamplePatterns(RewritePatternSet &patterns) {
  patterns.add<MaxPoolToLinalg>(patterns.getContext());
}

void npux::populateLinalgTransposePattern(RewritePatternSet &patterns) {
  patterns.add<TransposeToLinalg>(patterns.getContext());
}