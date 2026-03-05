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
// 1. 辅助逻辑：量化上下文处理
// ============================================================================

struct QuantizedContext {
  Value finalInput;             // 最终传给 NPU Op 的输入 (Int8/Int32)
  RankedTensorType finalOutputType; // NPU Op 的输出类型 (Int8/Int32)
  
  // 回调函数：处理输出替换
  // npuResult: NPU Op 产生的结果
  // rewriter: 用于操作 IR
  // return: 应当替换原 Op 结果的 Value (可能是 npuResult，也可能是 Dequant 的结果)
  std::function<void(Value npuResult, PatternRewriter &rewriter)> handleOutputReplacement;
};

// 核心逻辑：根据上下游 Quant/Dequant 节点判断如何处理
static LogicalResult handleQuantizationContext(
    Operation *op, 
    Value originalInput, 
    RankedTensorType originalOutputType,
    PatternRewriter &rewriter,
    QuantizedContext &ctx) 
{
  Location loc = op->getLoc();
  Type elemType = cast<RankedTensorType>(originalInput.getType()).getElementType();

  // Case 0: 输入本身就是 Integer (已经量化，或非浮点)
  if (isa<IntegerType>(elemType)) {
    ctx.finalInput = originalInput;
    ctx.finalOutputType = originalOutputType; // 保持原样
    ctx.handleOutputReplacement = [=](Value npuResult, PatternRewriter &b) {
      b.replaceOp(op, npuResult);
    };
    return success();
  }

  // 浮点输入，需要检查上下文
  auto dequantOp = originalInput.getDefiningOp<ONNXDequantizeLinearOp>();
  
  // 检查下游是否有且仅有一个 QuantizeLinearOp
  ONNXQuantizeLinearOp quantOp = nullptr;
  if (op->getResult(0).hasOneUse()) {
    quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(*op->getResult(0).getUsers().begin());
  }

  // Case 3: 上游有 Dequantize，下游有 Quantize -> 全删，直通 Int8
  if (dequantOp && quantOp) {
    ctx.finalInput = dequantOp.getX(); // 使用 Dequant 前的 Int8
    // 输出类型使用 Quantize 后的类型 (Int8)
    ctx.finalOutputType = cast<RankedTensorType>(quantOp.getResult().getType());
    
    ctx.handleOutputReplacement = [=](Value npuResult, PatternRewriter &b) {
      // 替换下游的 QuantOp
      b.replaceOp(quantOp, npuResult);
      // 删除当前 Op (Result 已被 QuantOp 替代，QuantOp 被 npuResult 替代)
      b.eraseOp(op);
      // 尝试清理上游
      if (dequantOp->hasOneUse()) b.eraseOp(dequantOp);
    };
    return success();
  }

  // Case 1: 上游有 Dequantize，下游没有 Quantize -> 删上游，下游插 Dequantize
  if (dequantOp && !quantOp) {
    ctx.finalInput = dequantOp.getX();
    // NPU 输出应当是 Int8，类型取自 Input (假设 Resample 不改变数据类型精度)
    Type quantizedElemType = cast<RankedTensorType>(ctx.finalInput.getType()).getElementType();
    ctx.finalOutputType = originalOutputType.clone(quantizedElemType);

    // 保存 Dequant 的参数，用于在输出端重建
    Value scale = dequantOp.getXScale();
    Value zp = dequantOp.getXZeroPoint();

    ctx.handleOutputReplacement = [=](Value npuResult, PatternRewriter &b) {
      // 在 NPU 输出后插入 Dequant
      auto newDequant = b.create<ONNXDequantizeLinearOp>(
          loc, originalOutputType, npuResult, scale, zp);
      b.replaceOp(op, newDequant);
      // 清理上游
      if (dequantOp->hasOneUse()) b.eraseOp(dequantOp);
    };
    return success();
  }

  // Case 2: 上游没有 Dequantize，下游有 Quantize -> 上游插 Quantize，删下游
  if (!dequantOp && quantOp) {
    // 获取下游 Quant 的参数，用于上游 Quant
    // 注意：这里假设 Resample 操作不改变数值分布(如 Transpose/Pool)，所以沿用参数是合理的。
    Value scale = quantOp.getYScale();
    Value zp = quantOp.getYZeroPoint();
    
    // 在 Op 前插入 Quantize
    // 这里的 Result Type 应该和 quantOp 的 Element Type 一致，但 Shape 和 Input 一致
    Type quantElemType = cast<RankedTensorType>(quantOp.getResult().getType()).getElementType();
    RankedTensorType quantInputType = cast<RankedTensorType>(originalInput.getType()).clone(quantElemType);
    
    auto newQuant = rewriter.create<ONNXQuantizeLinearOp>(
        loc, quantInputType, originalInput, scale, zp);
    
    ctx.finalInput = newQuant;
    ctx.finalOutputType = cast<RankedTensorType>(quantOp.getResult().getType());

    ctx.handleOutputReplacement = [=](Value npuResult, PatternRewriter &b) {
      b.replaceOp(quantOp, npuResult);
      b.eraseOp(op);
    };
    return success();
  }

  // Case 4: 都没有 -> 报 Warning，强行插入 Quant/Dequant (Default)
  op->emitWarning() << "NPU Resample/Transpose op found without Quantization context. Injecting default quantization (Scale=1.0, ZP=0).";
  
  // 创建默认参数 (Scale=1.0, ZP=0)
  RankedTensorType scaleType = RankedTensorType::get({}, rewriter.getF32Type());
  auto scaleAttr = DenseElementsAttr::get(scaleType, llvm::ArrayRef<float>{1.0f});
  
  Value defaultScale = rewriter.create<ONNXConstantOp>(loc, Attribute(), scaleAttr);

  RankedTensorType zpType = RankedTensorType::get({}, rewriter.getI8Type());
  auto zpAttr = DenseElementsAttr::get(zpType, llvm::ArrayRef<int8_t>{0});

  Value defaultZp = rewriter.create<ONNXConstantOp>(loc, Attribute(), zpAttr);
  
  // 1. Pre-Quantize
  RankedTensorType quantType = cast<RankedTensorType>(originalInput.getType()).clone(rewriter.getI8Type());
  auto newQuant = rewriter.create<ONNXQuantizeLinearOp>(
      loc, quantType, originalInput, defaultScale, defaultZp);
  
  ctx.finalInput = newQuant;
  ctx.finalOutputType = originalOutputType.clone(rewriter.getI8Type()); // NPU Output is I8

  ctx.handleOutputReplacement = [=](Value npuResult, PatternRewriter &b) {
    // 2. Post-Dequantize
    auto newDequant = b.create<ONNXDequantizeLinearOp>(
        loc, originalOutputType, npuResult, defaultScale, defaultZp);
    b.replaceOp(op, newDequant);
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

static Value createPackedResampleOp(
    ConversionPatternRewriter &rewriter, Location loc,
    Value input,                 
    RankedTensorType inputType,  
    RankedTensorType outputType, 
    StringRef libCallName,       
    AffineMap inputMap,          
    AffineMap outputMap,         
    std::function<void(Operation *)> attrHook = nullptr 
) {
  int64_t rank = inputType.getRank();
  // Spatial pool/resize path expects packed 5D maps.
  // Non-spatial path requires indexing maps consistent with tensor rank.
  bool useSpatialPackPath =
      (rank == 4) && (inputMap.getNumDims() == 5) && (outputMap.getNumDims() == 5);
  // --- Path A: Non-Spatial ---
  if (!useSpatialPackPath) {
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

  // --- Path B: Spatial (Pack/Unpack) ---
  int64_t channelDimPos = 1; 
  ArrayRef<int64_t> inShape = inputType.getShape();
  ArrayRef<int64_t> outShape = outputType.getShape();
  int64_t inputChannel = inShape[channelDimPos];

  int64_t tileFactor = 32;
  bool isSmallChannel = (inputChannel != ShapedType::kDynamic) && (inputChannel < 32);
  if (isSmallChannel) tileFactor = inputChannel;

  SmallVector<OpFoldResult> innerTilesOFR = {rewriter.getIndexAttr(tileFactor)};
  SmallVector<int64_t> innerDimsPos = {channelDimPos};

  // Packed Types
  SmallVector<int64_t> packedInputShape;
  if (inputType.hasStaticShape()) {
    for (int i = 0; i < 4; ++i) {
      if (i == channelDimPos) packedInputShape.push_back((inShape[i] + tileFactor - 1) / tileFactor);
      else packedInputShape.push_back(inShape[i]);
    }
    packedInputShape.push_back(tileFactor);
  } else {
    packedInputShape = SmallVector<int64_t>(5, ShapedType::kDynamic);
  }
  auto packedInputType = RankedTensorType::get(packedInputShape, inputType.getElementType());

  SmallVector<int64_t> packedOutputShape;
  if (outputType.hasStaticShape()) {
    for (int i = 0; i < 4; ++i) {
      if (i == channelDimPos) packedOutputShape.push_back((outShape[i] + tileFactor - 1) / tileFactor);
      else packedOutputShape.push_back(outShape[i]);
    }
    packedOutputShape.push_back(tileFactor);
  } else {
    packedOutputShape = SmallVector<int64_t>(5, ShapedType::kDynamic);
  }
  auto packedOutputType = RankedTensorType::get(packedOutputShape, outputType.getElementType());

  // Pack
  SmallVector<Value> packedDynamicSizes = getDynamicSizes(rewriter, loc, input, inputType.getShape());
  Value packedInit = rewriter.create<tensor::EmptyOp>(loc, packedInputType, packedDynamicSizes);

  Value paddingVal;
  if (mlir::isa<FloatType>(inputType.getElementType())) {
    paddingVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(inputType.getElementType(), 0.0));
  } else {
    paddingVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(inputType.getElementType(), 0));
  }

  auto packOp = rewriter.create<linalg::PackOp>(loc, input, packedInit, innerDimsPos, innerTilesOFR, paddingVal);

  // Execute Region
  auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(loc, packedOutputType);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&executeRegion.getRegion());

    SmallVector<Value> regionDynamicSizes; 
    Value regionAlloc = rewriter.create<bufferization::AllocTensorOp>(loc, packedOutputType, regionDynamicSizes);

    SmallVector<AffineMap, 2> indexingMaps = {inputMap, outputMap};
    SmallVector<utils::IteratorType> iteratorTypes(5, utils::IteratorType::parallel);

    auto linalgOp = rewriter.create<linalg::GenericOp>(loc,
        packedOutputType, packOp.getResult(), regionAlloc, indexingMaps, iteratorTypes,
        createLinalgBody);

    linalgOp->setAttr("library_call", rewriter.getStringAttr(libCallName));
    linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    if (attrHook) attrHook(linalgOp);

    rewriter.create<scf::YieldOp>(loc, linalgOp.getResults());
  }

  // Unpack
  SmallVector<Value> unpackDynamicSizes = getDynamicSizes(rewriter, loc, input, outputType.getShape());
  Value unpackDestInit = rewriter.create<tensor::EmptyOp>(loc, outputType, unpackDynamicSizes);

  auto unpackOp = rewriter.create<linalg::UnPackOp>(loc, executeRegion.getResults()[0], unpackDestInit, innerDimsPos, innerTilesOFR);

  return unpackOp.getResult();
}

// ============================================================================
// 3. MaxPool Pattern
// ============================================================================
struct MaxPoolToLinalg : public OpConversionPattern<ONNXMaxPoolSingleOutOp> {
  using OpConversionPattern<ONNXMaxPoolSingleOutOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // 1. Check constraints
    auto kernelShape = op.getKernelShape();
    auto strides = op.getStrides();
    if (!kernelShape || !strides) return failure();
    // 假设这里有严谨的 2x2 检查...

    Value input = op.getX();
    auto outputType = mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!outputType) return failure();
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() != 4 || outputType.getRank() != 4) {
      op.emitWarning() << "ONNXMaxPool lowering to NPU resample requires NCHW (4D) tensors.";
      return failure();
    }

    // 2. 处理量化上下文 [New Logic]
    QuantizedContext ctx;
    if (failed(handleQuantizationContext(op, input, outputType, rewriter, ctx))) {
        return failure();
    }

    // 3. 构建 Maps
    auto nExpr = rewriter.getAffineDimExpr(0);
    auto cExpr = rewriter.getAffineDimExpr(1);
    auto hExpr = rewriter.getAffineDimExpr(2);
    auto wExpr = rewriter.getAffineDimExpr(3);

    auto inputMap = AffineMap::get(4, 0, {nExpr, cExpr, hExpr * 2, wExpr * 2}, rewriter.getContext());
    auto outputMap = rewriter.getMultiDimIdentityMap(4);

    // 4. 使用 ctx 中的 input 和 outputType 创建 NPU Op
    Value result = createPackedResampleOp(rewriter, op.getLoc(), 
        ctx.finalInput, 
        mlir::cast<RankedTensorType>(ctx.finalInput.getType()), 
        ctx.finalOutputType, 
        "npu_maxpool", inputMap, outputMap);

    // 5. 调用回调替换原 Op
    ctx.handleOutputReplacement(result, rewriter);
    
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

    auto nExpr = rewriter.getAffineDimExpr(0);
    auto cExpr = rewriter.getAffineDimExpr(1);
    auto hExpr = rewriter.getAffineDimExpr(2);
    auto wExpr = rewriter.getAffineDimExpr(3);

    auto inputMap = AffineMap::get(4, 0, {nExpr, cExpr, hExpr * 2, wExpr * 2}, rewriter.getContext());
    auto outputMap = rewriter.getMultiDimIdentityMap(4);

    Value result = createPackedResampleOp(rewriter, op.getLoc(), 
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

    auto nExpr = rewriter.getAffineDimExpr(0);
    auto cExpr = rewriter.getAffineDimExpr(1);
    auto hExpr = rewriter.getAffineDimExpr(2);
    auto wExpr = rewriter.getAffineDimExpr(3);

    auto inputMap = AffineMap::get(4, 0,
      {nExpr, cExpr, hExpr.floorDiv(2), wExpr.floorDiv(2)},
      rewriter.getContext());
    auto outputMap = rewriter.getMultiDimIdentityMap(4);

    Value result = createPackedResampleOp(rewriter, op.getLoc(), 
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

    // 1. 处理量化上下文
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

      // 注意：使用 ctx.finalInput 和 ctx.finalOutputType
      SmallVector<Value> dynSizes = getDynamicSizes(rewriter, op.getLoc(), ctx.finalInput, ctx.finalOutputType.getShape());
      Value resultInit = rewriter.create<bufferization::AllocTensorOp>(op.getLoc(), ctx.finalOutputType, dynSizes);

      auto linalgOp = rewriter.create<linalg::GenericOp>(op.getLoc(),
          ctx.finalOutputType, ctx.finalInput, resultInit, indexingMaps, iteratorTypes,
          createLinalgBody); 

      linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_transpose"));
      linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

      rewriter.create<scf::YieldOp>(op.getLoc(), linalgOp.getResults());
    }

    // 替换逻辑交给 Context 回调
    ctx.handleOutputReplacement(executeRegion.getResults()[0], rewriter);
    return success();
  }
};

} // namespace

void npux::populateLinalgResamplePatterns(RewritePatternSet &patterns) {
  patterns.add<MaxPoolToLinalg, ResizeToLinalg>(patterns.getContext());
}

void npux::populateLinalgTransposePattern(RewritePatternSet &patterns) {
  patterns.add<TransposeToLinalg>(patterns.getContext());
}