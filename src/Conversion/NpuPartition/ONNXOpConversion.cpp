//=============================================================================
// /src/Conversion/NpuPartition/ONNXOpConversion.cpp
// this file implements the conversion of ONNX operations to linalg operations
// for NPU partitioning.
//=============================================================================

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/NPUConversionRegistry.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "src/Conversion/NpuPartition/NpuQuantHelper.hpp"

using namespace mlir;

namespace npux {

struct GeluToLinalg : public OpConversionPattern<ONNXGeluOp> {
  using OpConversionPattern<ONNXGeluOp>::OpConversionPattern;

  static bool isHardwareSupported(ONNXGeluOp op) {
    return true;
  }


  LogicalResult matchAndRewrite(
      ONNXGeluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    Location loc = op.getLoc();

    // 1. 向上匹配 Dequantize
    Value originInput = op.getX();
    auto dequantOp = originInput.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantOp) return failure();

    Value quantizedInput = dequantOp.getX(); // int8 Input
    auto inScaleOpt = getScalarQuantParams(dequantOp).scale;
    auto inZpOpt = getScalarQuantParams(dequantOp).zeroPoint;
    if (!inScaleOpt || !inZpOpt) return failure();

    // 2. 向下匹配 Quantize
    if (!op.getResult().hasOneUse()) return failure();
    Operation *userOp = *op.getResult().getUsers().begin();
    auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(userOp);
    if (!quantOp) return failure();

    auto outputType = mlir::dyn_cast<RankedTensorType>(quantOp.getResult().getType());
    if (!outputType) return failure();

    auto outScaleOpt = getScalarQuantParams(quantOp).scale;
    auto outZpOpt = getScalarQuantParams(quantOp).zeroPoint;
    if (!outScaleOpt || !outZpOpt) return failure();

    // ============================================================
    // 3. 创建包裹层 (scf.execute_region)
    // ============================================================
    auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(
        loc, outputType /* Result Type: int8 */
    );

    // === 【进入 Region 内部】 ===

    OpBuilder::InsertionGuard guard(rewriter);
    
    rewriter.createBlock(&executeRegion.getRegion());


    Value emptyTensor = rewriter.create<bufferization::AllocTensorOp>(
            loc, outputType, /*dynamicSizes=*/ValueRange{});

    // B. Maps & Iterators
    SmallVector<AffineMap, 2> indexingMaps = {
        rewriter.getMultiDimIdentityMap(outputType.getRank()), 
        rewriter.getMultiDimIdentityMap(outputType.getRank())};
    SmallVector<utils::IteratorType> iteratorTypes(
        outputType.getRank(), utils::IteratorType::parallel);

    // C. 创建 Linalg Generic
    auto linalgOp = rewriter.create<linalg::GenericOp>(loc,
        /*resultTypes=*/outputType,
        /*inputs=*/quantizedInput, 
        /*outputs=*/emptyTensor,
        indexingMaps, iteratorTypes,
        /*bodyBuilder=*/[&](OpBuilder &b, Location loc, ValueRange args) {
            Value in = args[0];
            Value dummyResult = b.create<arith::AddIOp>(loc, in, in);
            b.create<linalg::YieldOp>(loc, dummyResult);
        });

    // D. 设置 Attr
    linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_gelu")); 
    linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    linalgOp->setAttr("in_scale", rewriter.getF32FloatAttr(inScaleOpt));
    linalgOp->setAttr("in_zp", rewriter.getIntegerAttr(
        rewriter.getI32Type(), static_cast<int64_t>(inZpOpt)));
    linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScaleOpt));
    linalgOp->setAttr("out_zp", rewriter.getIntegerAttr(
        rewriter.getI16Type(), static_cast<int64_t>(outZpOpt)));

    // E. Region 结束，返回 Linalg 的结果
    rewriter.create<scf::YieldOp>(loc, linalgOp.getResults());

    // === 退出 Region (guard 析构) ===

    // ============================================================
    // 4. 替换外部 Op
    // ============================================================
    rewriter.replaceOp(quantOp, executeRegion.getResults());

    rewriter.eraseOp(op);
    if (dequantOp->hasOneUse()) {
        rewriter.eraseOp(dequantOp);
    }

    return success();
  }
};

struct ConvToLinalg : public OpConversionPattern<ONNXConvOp> {
  using OpConversionPattern<ONNXConvOp>::OpConversionPattern;

  static bool isHardwareSupported(ONNXConvOp op) { return true; }

  // 辅助函数：报告错误
  LogicalResult reportError(Operation *op, ConversionPatternRewriter &rewriter,
                            const std::string &msg) const {
    llvm::errs() << "[[ConvToLinalg FAIL]] " << msg << "\n";
    return rewriter.notifyMatchFailure(op, msg);
  }


  LogicalResult matchAndRewrite(ONNXConvOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // 1. Input Side
    Value originInput = op.getX();
    auto dequantX = originInput.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantX) return reportError(op, rewriter, "Input not from Dequantize");

    Value quantizedX = dequantX.getX();
    auto xScale = getScalarQuantParams(dequantX).scale;
    auto xZp = getScalarQuantParams(dequantX).zeroPoint;
    if (!xScale || !xZp) return reportError(op, rewriter, "Input params missing");

    // 2. Weight Side
    Value originWeight = op.getW();
    auto dequantW = originWeight.getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dequantW) return reportError(op, rewriter, "Weight not from Dequantize");

    Value quantizedW = dequantW.getX();
    

    // 3. Bias Side
    Value bias = op.getB();
    bool hasBias = !mlir::isa<NoneType>(bias.getType());
    
    // [Fix 1] 专门定义一个变量来记录 Bias 的 Dequant Op，以便稍后删除
    ONNXDequantizeLinearOp dequantBias = nullptr; 

    if (hasBias) {
      if (auto db = bias.getDefiningOp<ONNXDequantizeLinearOp>()) {
        dequantBias = db; // 记录下来！
        bias = db.getX(); // 获取 Int32 的 Bias 输入
      }
    }

    // 4. Output Side
    if (!op.getResult().hasOneUse()) return failure();
    auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(*op.getResult().getUsers().begin());
    if (!quantOp) return failure();
    auto outputType = mlir::dyn_cast<RankedTensorType>(quantOp.getResult().getType());

    auto outScale = getScalarQuantParams(quantOp).scale;
    auto outZp = getScalarQuantParams(quantOp).zeroPoint;
    if (!outScale || !outZp) return failure();

    // ============================================================
    // Linalg Generation
    // ============================================================
    auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(loc, outputType);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&executeRegion.getRegion());

      // === Explicit Padding ===
      Value inputToLinalg = quantizedX; 
      SmallVector<int64_t> padsVal;
      if (auto padsAttr = op.getPadsAttr()) {
         for(auto p : padsAttr) padsVal.push_back(mlir::cast<IntegerAttr>(p).getInt());
      }
      
      bool needPadding = !padsVal.empty() && llvm::any_of(padsVal, [](int64_t v){ return v > 0; });
      
      if (needPadding && padsVal.size() == 4) {
          int64_t pH_begin = padsVal[0];
          int64_t pW_begin = padsVal[1];
          int64_t pH_end = padsVal[2];
          int64_t pW_end = padsVal[3];
          
          // [Fix 1] 准备填充值 (i8)
          Value padVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getI8IntegerAttr((int8_t)xZp));
          
          // [Fix 2] 准备 Low/High 参数
          // 使用 OpFoldResult 存储 Attribute (静态值)，完全匹配 build 签名
          SmallVector<OpFoldResult> low, high;
          
          // Low: [0, 0, pH_begin, pW_begin]
          low.push_back(rewriter.getIndexAttr(0));
          low.push_back(rewriter.getIndexAttr(0));
          low.push_back(rewriter.getIndexAttr(pH_begin));
          low.push_back(rewriter.getIndexAttr(pW_begin));

          // High: [0, 0, pH_end, pW_end]
          high.push_back(rewriter.getIndexAttr(0));
          high.push_back(rewriter.getIndexAttr(0));
          high.push_back(rewriter.getIndexAttr(pH_end));
          high.push_back(rewriter.getIndexAttr(pW_end));
          
          // [Fix 3] 计算 Result Type (Static Shape)
          auto srcType = mlir::cast<RankedTensorType>(quantizedX.getType());
          auto inputShape = srcType.getShape();
          SmallVector<int64_t> paddedShape = {
              inputShape[0], inputShape[1], 
              inputShape[2] + pH_begin + pH_end, 
              inputShape[3] + pW_begin + pW_end
          };
          auto paddedType = RankedTensorType::get(paddedShape, srcType.getElementType());

          // [Fix 4] 调用 create，直接传入 padVal，不需要 lambda
          // 签名匹配: (Type resultType, Value source, ArrayRef<OpFoldResult> low, ArrayRef<OpFoldResult> high, Value constantPadValue, bool nofold)
          auto padOp = rewriter.create<tensor::PadOp>(
              loc, 
              paddedType,     // resultType
              quantizedX,     // source
              low,            // low
              high,           // high
              padVal,         // constantPadValue <--- 关键！直接传值
              /*nofold=*/false 
          );
          
          inputToLinalg = padOp.getResult();
      }

      Value emptyTensor = rewriter.create<bufferization::AllocTensorOp>(loc, outputType, ValueRange{});

      MLIRContext *ctx = rewriter.getContext();
      auto d0 = getAffineDimExpr(0, ctx); 
      auto d1 = getAffineDimExpr(1, ctx); 
      auto d2 = getAffineDimExpr(2, ctx); 
      auto d3 = getAffineDimExpr(3, ctx); 
      auto d4 = getAffineDimExpr(4, ctx); 
      auto d5 = getAffineDimExpr(5, ctx); 
      auto d6 = getAffineDimExpr(6, ctx); 

      // Input Map logic: OH + KH -> IH
      auto inputMap = AffineMap::get(7, 0, {d0, d4, d2 + d5, d3 + d6}, ctx);
      auto weightMap = AffineMap::get(7, 0, {d1, d4, d5, d6}, ctx);
      auto outputMap = AffineMap::get(7, 0, {d0, d1, d2, d3}, ctx);

      SmallVector<AffineMap> maps = {inputMap, weightMap};
      SmallVector<Value> inputs = {inputToLinalg, quantizedW}; 

      if (hasBias) {
        maps.push_back(AffineMap::get(7, 0, {d1}, ctx));
        inputs.push_back(bias);
      }
      maps.push_back(outputMap);

      SmallVector<utils::IteratorType> iterators = {
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::reduction, utils::IteratorType::reduction,
          utils::IteratorType::reduction};

      auto linalgOp = rewriter.create<linalg::GenericOp>(
          loc, outputType, inputs, emptyTensor, maps, iterators,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value in = args[0];
            Value w = args[1];
            Value res = b.create<arith::MulIOp>(loc, in, w);
            b.create<linalg::YieldOp>(loc, res);
          });
      
      linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_conv_nchwc32"));
      linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
      linalgOp->setAttr("in_scale", rewriter.getF32FloatAttr(xScale));
      linalgOp->setAttr("in_zp", rewriter.getI32IntegerAttr((int32_t)xZp));
      linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
      linalgOp->setAttr("out_zp", rewriter.getI32IntegerAttr((int32_t)outZp));
      
      if (auto strides = op.getStridesAttr()) linalgOp->setAttr("strides", strides);
      if (auto dilations = op.getDilationsAttr()) linalgOp->setAttr("dilations", dilations);

      rewriter.create<scf::YieldOp>(loc, linalgOp.getResults());
    }

    rewriter.replaceOp(quantOp, executeRegion.getResults());

    rewriter.eraseOp(op);


    if (dequantW && dequantW->hasOneUse()) {
        rewriter.eraseOp(dequantW);
    }
    if (dequantX && dequantX->hasOneUse()) {
        rewriter.eraseOp(dequantX);
    }

    if (dequantBias && dequantBias->hasOneUse()) {
        rewriter.eraseOp(dequantBias);
    }

    return success();
  }
};



static npux::NPUOpRegistration<GeluToLinalg, ONNXGeluOp> registerGelu;
//static npux::NPUOpRegistration<ConvToLinalg, ONNXConvOp> registerConv;

void registerNpuOpConversions() {};

} // namespace npux