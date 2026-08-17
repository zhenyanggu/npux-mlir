//==============================================================
//src/Conversion/NpuPartition/Lianlg/Unary.cpp
// this file implements the conversion of unary operations (Gelu, Softmax)
// to linalg operations for NPU partitioning.
//==============================================================
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
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

// 通用的 Body 构建器，用于填充 linalg.generic 的 Region
static void createUnaryBody(OpBuilder &b, Location loc, ValueRange args) {
  Value input = args[0];
  Value result = input;
  Type elemType = input.getType();

  // 这里的计算逻辑通常被 library_call 覆盖，但在 IR 层面必须合法
  if (mlir::isa<FloatType>(elemType)) {
    result = b.create<arith::AddFOp>(loc, input, input);
  } else if (mlir::isa<IntegerType>(elemType)) {
    if (elemType.isSignlessInteger()) {
      result = b.create<arith::AddIOp>(loc, input, input);
    } else {
      Type signlessType = b.getIntegerType(elemType.getIntOrFloatBitWidth());
      Value signlessInput = b.create<UnrealizedConversionCastOp>(
          loc, signlessType, input).getResult(0);
      Value sum = b.create<arith::AddIOp>(loc, signlessInput, signlessInput);
      result = b.create<UnrealizedConversionCastOp>(loc, elemType, sum)
                   .getResult(0);
    }
  }

  b.create<linalg::YieldOp>(loc, result);
}

static Value createPackedUnaryOp(
    ConversionPatternRewriter &rewriter, Location loc,
    Value quantizedInput,            // 已是 Int8 的输入
    RankedTensorType inputType,      // 输入类型
    RankedTensorType outputType,     // 输出类型
    double inScale, int64_t inZp, double outScale, int64_t outZp,
    StringRef libCallName, Operation *sourceOp, StringRef layerName,
    ArrayRef<StringRef> fusedOps,
    std::function<void(Operation *)> attrHook = nullptr
) {
  int64_t rank = inputType.getRank();
  // ==========================================================
  // 直接创建带有 Encoding=1 的新类型
  // ==========================================================
  RankedTensorType inputType1 = addEncoding1(inputType, rewriter);
  RankedTensorType outputType1 = addEncoding1(outputType, rewriter);

  // 强制原地修改外部输入的类型 (无 Cast 读写同一块内存)
  quantizedInput.setType(inputType1);
    // 1. 创建带有 encoding=1 的输出 Buffer
    SmallVector<Value> dynamicSizes =
        getDynamicSizes(rewriter, loc, quantizedInput, inputType1.getShape());
    Value regionAlloc = rewriter.create<bufferization::AllocTensorOp>(
        loc, outputType1, dynamicSizes);

    // 2. 构建 Indexing Maps (Identity)
    SmallVector<AffineMap, 2> indexingMaps = {
        rewriter.getMultiDimIdentityMap(rank),
        rewriter.getMultiDimIdentityMap(rank)
    };
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    // 3. 创建 GenericOp
    auto linalgOp = rewriter.create<linalg::GenericOp>(loc,
        /*resultTypes=*/outputType1,
        /*inputs=*/quantizedInput,
        /*outputs=*/regionAlloc,
        indexingMaps, iteratorTypes,
        /*bodyBuilder=*/createUnaryBody);

    // 4. 设置属性
    linalgOp->setAttr("library_call", rewriter.getStringAttr(libCallName));
    linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    setNpuProfileAttrs(
        linalgOp, sourceOp, rewriter, layerName, "compute", fusedOps);
    linalgOp->setAttr("in_scale", rewriter.getF32FloatAttr(inScale));
    linalgOp->setAttr("in_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), inZp));
    linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
    linalgOp->setAttr("out_zp", rewriter.getIntegerAttr(rewriter.getI16Type(), outZp));

    if (attrHook) attrHook(linalgOp);

    // 直接返回 LinalgOp 的结果
    return linalgOp.getResult(0);
}

struct GeluQuantizedChain {
  ONNXDequantizeLinearOp dequantOp;
  ONNXReshapeOp inputReshapeOp;
  ONNXReshapeOp outputReshapeOp;
  UnrealizedConversionCastOp outputCastOp;
  ONNXQuantizeLinearOp quantOp;
};

static RankedTensorType getQuantizedTensorTypeLike(
    RankedTensorType shapedType, RankedTensorType referenceType) {
  return RankedTensorType::get(shapedType.getShape(),
      referenceType.getElementType(), referenceType.getEncoding());
}

static Value createQuantizedReshapeLike(ConversionPatternRewriter &rewriter,
    Location loc, Value input, ONNXReshapeOp reshapeOp) {
  auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
  auto reshapeType =
      mlir::dyn_cast<RankedTensorType>(reshapeOp.getResult().getType());
  if (!inputType || !reshapeType)
    return Value();

  RankedTensorType outputType =
      getQuantizedTensorTypeLike(reshapeType, inputType);
  onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(rewriter, loc);
  return create.onnx.reshape(
      outputType, input, reshapeOp.getShape(), reshapeOp.getAllowzeroAttr());
}

static bool matchGeluQuantizedChain(ONNXGeluOp op, GeluQuantizedChain &chain) {
  Value input = op.getX();
  if (auto dequantOp = input.getDefiningOp<ONNXDequantizeLinearOp>()) {
    chain.dequantOp = dequantOp;
  } else if (auto reshapeOp = input.getDefiningOp<ONNXReshapeOp>()) {
    if (!input.hasOneUse())
      return false;
    auto dequantOp = reshapeOp.getData().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantOp)
      return false;
    chain.dequantOp = dequantOp;
    chain.inputReshapeOp = reshapeOp;
  } else {
    return false;
  }

  Value result = op.getResult();
  if (!result.hasOneUse())
    return false;

  Operation *user = *result.getUsers().begin();
  if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
    if (castOp.getNumResults() != 1 || !castOp.getResult(0).hasOneUse())
      return false;
    chain.outputCastOp = castOp;
    user = *castOp.getResult(0).getUsers().begin();
  }

  if (auto quantOp = dyn_cast<ONNXQuantizeLinearOp>(user)) {
    chain.quantOp = quantOp;
    return true;
  }

  auto reshapeOp = dyn_cast<ONNXReshapeOp>(user);
  if (!reshapeOp || !reshapeOp.getResult().hasOneUse())
    return false;

  auto quantOp =
      dyn_cast<ONNXQuantizeLinearOp>(*reshapeOp.getResult().getUsers().begin());
  if (!quantOp)
    return false;

  chain.outputReshapeOp = reshapeOp;
  chain.quantOp = quantOp;
  return true;
}

// ============================================================================
// 1. Gelu Pattern
// ============================================================================
struct GeluToLinalg : public OpConversionPattern<ONNXGeluOp> {
  using OpConversionPattern<ONNXGeluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGeluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    GeluQuantizedChain chain;
    if (!matchGeluQuantizedChain(op, chain))
      return failure();

    Value quantizedInput = chain.dequantOp.getX();
    auto quantizedInputType =
        mlir::dyn_cast<RankedTensorType>(quantizedInput.getType());
    auto geluResultType =
        mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
    auto finalOutputType =
        mlir::dyn_cast<RankedTensorType>(chain.quantOp.getResult().getType());
    if (!quantizedInputType || !geluResultType || !finalOutputType)
      return failure();

    RankedTensorType packedInputType = quantizedInputType;
    if (chain.inputReshapeOp) {
      quantizedInput = createQuantizedReshapeLike(
          rewriter, op.getLoc(), quantizedInput, chain.inputReshapeOp);
      if (!quantizedInput)
        return failure();
      auto reshapedInputType =
          mlir::dyn_cast<RankedTensorType>(quantizedInput.getType());
      if (!reshapedInputType)
        return failure();
      packedInputType = reshapedInputType;
    }

    RankedTensorType packedOutputType = finalOutputType;
    if (chain.outputReshapeOp)
      packedOutputType =
          getQuantizedTensorTypeLike(geluResultType, finalOutputType);

    auto inParams = getScalarQuantParams(chain.dequantOp);
    auto outParams = getScalarQuantParams(chain.quantOp);
    SmallVector<StringRef> fusedOps = {"Gelu"};

    Value packedResult = createPackedUnaryOp(rewriter, op.getLoc(),
        quantizedInput, packedInputType, packedOutputType, inParams.scale,
        inParams.zeroPoint,
        outParams.scale, outParams.zeroPoint, "npu_gelu", op,
        getNpuProfileLayerName(op), fusedOps);

    Value finalResult = packedResult;
    if (chain.outputReshapeOp) {
      finalResult = createQuantizedReshapeLike(
          rewriter, op.getLoc(), packedResult, chain.outputReshapeOp);
      if (!finalResult)
        return failure();
    }

    chain.quantOp.getResult().setType(finalResult.getType());

    // result 现在带有 encoding=1，这会向外顺延影响下游的 Consumer
    rewriter.replaceAllUsesWith(chain.quantOp.getResult(), finalResult);
    rewriter.eraseOp(chain.quantOp);
    if (chain.outputReshapeOp)
      rewriter.eraseOp(chain.outputReshapeOp);
    if (chain.outputCastOp && chain.outputCastOp->use_empty())
      rewriter.eraseOp(chain.outputCastOp);
    rewriter.eraseOp(op);
    if (chain.inputReshapeOp)
      rewriter.eraseOp(chain.inputReshapeOp);
    if (chain.dequantOp->use_empty())
      rewriter.eraseOp(chain.dequantOp);
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
    SmallVector<StringRef> fusedOps = {"Softmax"};

    Value result =
        createPackedUnaryOp(rewriter, op.getLoc(), quantizedInput, inputType,
            outputType, inParams.scale, inParams.zeroPoint, outParams.scale,
            outParams.zeroPoint, "npu_softmax", op,
            getNpuProfileLayerName(op), fusedOps, [&](Operation *genericOp) {
              genericOp->setAttr("axis", rewriter.getI64IntegerAttr(axis));
            });

    quantOp.getResult().setType(result.getType());

    rewriter.replaceOp(quantOp, result);
    rewriter.eraseOp(op);
    if (dequantOp->hasOneUse()) rewriter.eraseOp(dequantOp);
    return success();
  }
};
}

void npux::populateLinalgUnaryPatterns(RewritePatternSet &patterns) {
  patterns.add<GeluToLinalg, SoftmaxToLinalg>(
      patterns.getContext());
}
