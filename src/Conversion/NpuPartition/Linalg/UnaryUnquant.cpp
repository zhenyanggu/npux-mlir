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
// 0. 辅助函数：为 RankedTensorType 添加 encoding=1
// ============================================================================
static RankedTensorType addEncoding1(RankedTensorType type, OpBuilder &b) {
  if (type.getEncoding()) {
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(type.getEncoding())) {
      if (intAttr.getInt() == 1) return type;
    }
  }
  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               b.getI64IntegerAttr(1));
}

// ============================================================================
// 1. 辅助逻辑：量化上下文处理
// ============================================================================

struct QuantizedContext {
  Value finalInput;             // 最终传给 NPU Op 的输入 (Int8, 带有 encoding=1)
  RankedTensorType finalOutputType; // NPU Op 的输出类型 (Int8, 带有 encoding=1)
  
  // 回调函数：处理输出替换
  std::function<void(Value npuResult, PatternRewriter &rewriter)> handleOutputReplacement;
};

// 核心逻辑：严格匹配上下游 qdq，没有就不处理，有就直接吸收转 Int8，并加上 encoding=1
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
  
  // 吸收上游 dq：直接拿 dequant 之前的 Int8 作为输入，并强制覆盖 Type (加上 encoding=1)
  ctx.finalInput = dequantOp.getX(); 
  auto inputType = mlir::cast<RankedTensorType>(ctx.finalInput.getType());
  ctx.finalInput.setType(addEncoding1(inputType, rewriter));

  // 吸收下游 q：直接拿 quant 之后的类型作为 NPU Op 的输出类型，并加上 encoding=1
  auto outputType = mlir::cast<RankedTensorType>(quantOp.getResult().getType());
  ctx.finalOutputType = addEncoding1(outputType, rewriter);
  
  ctx.handleOutputReplacement = [=](Value npuResult, PatternRewriter &b) mutable {
    // 顺延 encoding=1 到外部消费节点的类型上
    quantOp.getResult().setType(npuResult.getType());
    
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
// 2. Linalg 构建逻辑
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
  // 移除 scf::ExecuteRegionOp，直接在当前块分配 Buffer 并执行 linalg
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

  // 直接返回 genericOp 的结果
  return linalgOp.getResult(0);
}

// ============================================================================
// 3. MaxPool Pattern
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
      op.emitWarning() << "ONNXMaxPool lowering requires NCHW (4D) tensors.";
      return failure();
    }

    QuantizedContext ctx;
    if (failed(handleQuantizationContext(op, input, outputType, rewriter, ctx))) {
        return failure();
    }

    int64_t strideH = mlir::cast<IntegerAttr>((*strides)[0]).getInt();
    int64_t strideW = mlir::cast<IntegerAttr>((*strides)[1]).getInt();
    int64_t kH = mlir::cast<IntegerAttr>((kernelShape)[0]).getInt();
    int64_t kW = mlir::cast<IntegerAttr>((kernelShape)[1]).getInt();

    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel,  // d0: N
        utils::IteratorType::parallel,  // d1: C
        utils::IteratorType::parallel,  // d2: H_out
        utils::IteratorType::parallel,  // d3: W_out
        utils::IteratorType::reduction, // d4: Kernel_H
        utils::IteratorType::reduction  // d5: Kernel_W
    };

    int64_t loopRank = 6;
    auto n  = rewriter.getAffineDimExpr(0);
    auto c  = rewriter.getAffineDimExpr(1);
    auto oh = rewriter.getAffineDimExpr(2);
    auto ow = rewriter.getAffineDimExpr(3);
    auto kh = rewriter.getAffineDimExpr(4);
    auto kw = rewriter.getAffineDimExpr(5);

    SmallVector<AffineExpr> inputExprs = {n, c, oh * strideH + kh, ow * strideW + kw};
    auto inputMap = AffineMap::get(loopRank, 0, inputExprs, rewriter.getContext());

    SmallVector<AffineExpr> windowExprs = {kh, kw};
    auto windowMap = AffineMap::get(loopRank, 0, windowExprs, rewriter.getContext());

    SmallVector<AffineExpr> outputExprs = {n, c, oh, ow};
    auto outputMap = AffineMap::get(loopRank, 0, outputExprs, rewriter.getContext());

    // 1. 创建带有 encoding=1 的 Dummy Window Tensor
    auto dummyWindowType = RankedTensorType::get({kH, kW}, ctx.finalOutputType.getElementType(), rewriter.getI64IntegerAttr(1));
    Value dummyWindow = rewriter.create<bufferization::AllocTensorOp>(op.getLoc(), dummyWindowType, ValueRange{});

    // 2. 创建 Result Tensor
    SmallVector<Value> dynSizes = getDynamicSizes(rewriter, op.getLoc(), ctx.finalInput, ctx.finalOutputType.getShape());
    Value resultInit = rewriter.create<bufferization::AllocTensorOp>(op.getLoc(), ctx.finalOutputType, dynSizes);

    SmallVector<AffineMap, 3> indexingMaps = {inputMap, windowMap, outputMap};
    
    auto linalgOp = rewriter.create<linalg::GenericOp>(op.getLoc(),
        ctx.finalOutputType, 
        ValueRange{ctx.finalInput, dummyWindow}, 
        resultInit, 
        indexingMaps, 
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value in = args[0];
            Value acc = args[2];
            
            Value res;
            if (mlir::isa<FloatType>(in.getType())) {
                res = b.create<arith::MaximumFOp>(loc, in, acc);
            } else {
                res = b.create<arith::MaxSIOp>(loc, in, acc);
            }
            b.create<linalg::YieldOp>(loc, res);
        }); 

    linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_maxpool"));
    linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

    ctx.handleOutputReplacement(linalgOp.getResult(0), rewriter);
    return success();
  }
};

// ============================================================================
// 4. AveragePool Pattern
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
// 5. Resize Pattern
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
// 6. Transpose Pattern
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

    // 移除了 scf 域相关的构造逻辑
    SmallVector<Value> dynSizes = getDynamicSizes(rewriter, op.getLoc(), ctx.finalInput, ctx.finalOutputType.getShape());
    Value resultInit = rewriter.create<bufferization::AllocTensorOp>(op.getLoc(), ctx.finalOutputType, dynSizes);

    auto linalgOp = rewriter.create<linalg::GenericOp>(op.getLoc(),
        ctx.finalOutputType, ctx.finalInput, resultInit, indexingMaps, iteratorTypes,
        createLinalgBody); 

    linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_transpose"));
    linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

    ctx.handleOutputReplacement(linalgOp.getResult(0), rewriter);
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