//==============================================================
// src/Conversion/NpuPartition/Linalg/MatMul.cpp
// This file implements the conversion of Gemm and QLinearMatMul
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
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/NpuPartition/LinalgConversionHelper.hpp"

using namespace mlir;
using namespace npux;

namespace {

// ==========================================================
// 辅助函数
// ==========================================================

static float getScalarFloat(Value v, float defaultVal = 1.0f) {
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto floatAttr = mlir::dyn_cast<FloatAttr>(constOp.getValue())) {
      return (float)floatAttr.getValueAsDouble();
    }
  }
  if (auto constOp = v.getDefiningOp<ONNXConstantOp>()) {
    if (auto dense = mlir::dyn_cast<DenseElementsAttr>(constOp.getValueAttr())) {
      return dense.getValues<float>()[0];
    }
  }
  return defaultVal;
}

static int64_t getScalarInt(Value v, int64_t defaultVal = 0) {
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
  }
  if (auto constOp = v.getDefiningOp<ONNXConstantOp>()) {
    if (auto dense = mlir::dyn_cast<DenseElementsAttr>(constOp.getValueAttr())) {
       if (dense.getElementType().isInteger(8)) return dense.getValues<int8_t>()[0];
       if (dense.getElementType().isInteger(32)) return dense.getValues<int32_t>()[0];
       if (dense.getElementType().isInteger(64)) return dense.getValues<int64_t>()[0];
    }
  }
  return defaultVal;
}

// 创建显式的 ONNXTransposeOp，交换最后两维
static Value createExplicitTranspose(OpBuilder &b, Location loc, Value input) {
  auto type = mlir::dyn_cast<RankedTensorType>(input.getType());
  if (!type) return input;

  int64_t rank = type.getRank();
  if (rank < 2) return input;

  SmallVector<int64_t> perm;
  for (int64_t i = 0; i < rank; ++i) perm.push_back(i);
  std::swap(perm[rank - 1], perm[rank - 2]);

  auto permArrayAttr = b.getI64ArrayAttr(perm);
  
  auto transOp = b.create<ONNXTransposeOp>(loc, 
      /*resultType=*/UnrankedTensorType::get(type.getElementType()), 
      /*data=*/input,
      /*perm=*/permArrayAttr);
  
  // 简单的 Shape Inference
  SmallVector<int64_t> newShape(type.getShape());
  std::swap(newShape[rank - 1], newShape[rank - 2]);
  auto newType = RankedTensorType::get(newShape, type.getElementType());
  transOp.getResult().setType(newType);
  
  return transOp.getResult();
}

// MatMul i32 Body 构建器
// 确保使用 Integer 运算并且仅输出 i32 的累加结果，精度转换留给 SPM 阶段
static void createI32MatMulBody(OpBuilder &b, Location loc, ValueRange args) {
  // args: [A, B, (Optional C), OutAcc]
  Value lhs = args[0];
  Value rhs = args[1];
  Value outAcc = args.back();
  bool hasBias = (args.size() == 4);

  // Helper: 统一提升到 i32 进行计算
  auto castToI32 = [&](Value v) -> Value {
    Type t = v.getType();
    if (t.isInteger(32)) return v;
    if (t.isInteger(8) || t.isInteger(1) || t.isInteger(16)) {
      return b.create<arith::ExtSIOp>(loc, b.getI32Type(), v);
    }
    // Fallback for float
    if (mlir::isa<FloatType>(t)) {
        return b.create<arith::FPToSIOp>(loc, b.getI32Type(), v);
    }
    return v; 
  };

  Value lhsI32 = castToI32(lhs);
  Value rhsI32 = castToI32(rhs);
  Value outI32 = castToI32(outAcc); // i32 accum

  // 1. Mul: i32 = i32 * i32
  Value mul = b.create<arith::MulIOp>(loc, lhsI32, rhsI32);

  // 2. Add Bias: i32 = i32 + i32
  if (hasBias) {
    Value biasI32 = castToI32(args[2]);
    mul = b.create<arith::AddIOp>(loc, biasI32, mul);
  }

  // 3. Accumulate: i32 = i32 + i32
  Value resI32 = b.create<arith::AddIOp>(loc, outI32, mul);

  // 4. Yield pure i32 (no truncation here!)
  b.create<linalg::YieldOp>(loc, resI32);
}

// 通用的 MatMul 构建器
static Value createGenericMatMulOp(
    ConversionPatternRewriter &rewriter, Location loc,
    SmallVector<Value> inputs,          // [A, B] 或 [A, B, C]
    RankedTensorType outType,            // Final Output Type (e.g., i8)
    // Quant Params
    float lhsScale, int64_t lhsZp,
    float rhsScale, int64_t rhsZp,
    float outScale, int64_t outZp,
    StringRef libCallName
) {
  int64_t outRank = outType.getRank();
  bool hasBias = (inputs.size() == 3);
  int32_t withBiasAttr = hasBias ? 1 : 0;

  auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(loc, outType);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&executeRegion.getRegion());

    // 1. 提取动态尺寸，供内存分配使用
    SmallVector<Value> dynamicSizes =
        getDynamicSizes(rewriter, loc, inputs[0], outType.getShape());

    // 2. 分配中间 i32 累加器 Buffer (NPU Accumulator)
    auto i32Type = RankedTensorType::get(outType.getShape(), rewriter.getI32Type());
    Value i32Alloc = rewriter.create<bufferization::AllocTensorOp>(
        loc, i32Type, dynamicSizes);

    // 3. 构建 GEMM Indexing Maps
    SmallVector<utils::IteratorType> iteratorTypes(outRank + 1, utils::IteratorType::parallel);
    iteratorTypes[outRank] = utils::IteratorType::reduction; 
    
    // Map 构建 Helper
    auto getMap = [&](bool isA, bool isB, bool isC, Value val = Value()) -> AffineMap {
      SmallVector<AffineExpr> exprs;
      for (int i = 0; i < outRank - 2; ++i) 
        exprs.push_back(rewriter.getAffineDimExpr(i));
        
      AffineExpr m = rewriter.getAffineDimExpr(outRank - 2);
      AffineExpr n = rewriter.getAffineDimExpr(outRank - 1);
      AffineExpr k = rewriter.getAffineDimExpr(outRank);
      
      if (isA) { 
        exprs.push_back(m);
        exprs.push_back(k);
      } else if (isB) { 
        exprs.push_back(k);
        exprs.push_back(n);
      } else if (isC) {
        int64_t biasRank = mlir::cast<RankedTensorType>(val.getType()).getRank();
        if (biasRank == 1) {
             exprs.clear(); 
             exprs.push_back(n);
             return AffineMap::get(outRank + 1, 0, exprs, rewriter.getContext());
        } else {
             exprs.push_back(m);
             exprs.push_back(n);
        }
      } else { 
        exprs.push_back(m);
        exprs.push_back(n);
      }
      return AffineMap::get(outRank + 1, 0, exprs, rewriter.getContext());
    };

    SmallVector<AffineMap> gemmIndexingMaps;
    gemmIndexingMaps.push_back(getMap(true, false, false));  // A
    gemmIndexingMaps.push_back(getMap(false, true, false));  // B
    
    if (hasBias) {
      gemmIndexingMaps.push_back(getMap(false, false, true, inputs[2])); // C
    }
    
    gemmIndexingMaps.push_back(getMap(false, false, false)); // Out (i32)

    // 4. 创建 GenericOp (GEMM -> i32)
    auto gemmOp = rewriter.create<linalg::GenericOp>(loc,
        /*resultTypes=*/i32Type,
        /*inputs=*/inputs,
        /*outputs=*/i32Alloc,
        gemmIndexingMaps, iteratorTypes,
        /*bodyBuilder=*/createI32MatMulBody);

    gemmOp->setAttr("library_call", rewriter.getStringAttr(libCallName));
    gemmOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    gemmOp->setAttr("lhs_scale", rewriter.getF32FloatAttr(lhsScale));
    gemmOp->setAttr("lhs_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), lhsZp));
    gemmOp->setAttr("rhs_scale", rewriter.getF32FloatAttr(rhsScale));
    gemmOp->setAttr("rhs_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), rhsZp));
    gemmOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
    gemmOp->setAttr("out_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), outZp));
    gemmOp->setAttr("with_bias", rewriter.getI32IntegerAttr(withBiasAttr));

    // ==========================================
    // 5. SPM 阶段：i32 -> 最终类型 (通常为 i8)
    // ==========================================

    // 分配最终类型的 Output Buffer (SPM)
    Value finalAlloc = rewriter.create<bufferization::AllocTensorOp>(
        loc, outType, dynamicSizes);

    SmallVector<AffineMap> identityMaps(2, rewriter.getMultiDimIdentityMap(outRank));
    SmallVector<utils::IteratorType> parallelIters(outRank, utils::IteratorType::parallel);

    auto quantOp = rewriter.create<linalg::GenericOp>(loc, 
        /*resultTypes=*/outType,
        /*inputs=*/ValueRange{gemmOp.getResult(0)}, 
        /*outputs=*/ValueRange{finalAlloc},
        identityMaps, parallelIters,
        [&](OpBuilder &nestedB, Location nestedLoc, ValueRange args) {
          Value inI32 = args[0];
          Type targetType = outType.getElementType();
          Value res;

          if (targetType.isInteger(32)) {
            res = inI32;
          } else if (targetType.isInteger(8)) {
            // i32 -> i8 截断
            res = nestedB.create<arith::TruncIOp>(nestedLoc, targetType, inI32);
          } else if (mlir::isa<FloatType>(targetType)) {
            res = nestedB.create<arith::SIToFPOp>(nestedLoc, targetType, inI32);
          } else {
            res = inI32;
          }
          nestedB.create<linalg::YieldOp>(nestedLoc, res);
        });

    quantOp->setAttr("library_call", rewriter.getStringAttr("mv_acc_to_spm"));
    quantOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

    rewriter.create<scf::YieldOp>(loc, quantOp.getResults());
  }
  return executeRegion.getResults()[0];
}

// ============================================================================
// 1. Gemm Pattern
// ============================================================================
struct GemmToLinalg : public OpConversionPattern<ONNXGemmOp> {
  using OpConversionPattern<ONNXGemmOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGemmOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    Value inputA = op.getA();
    Value inputB = op.getB();
    Value inputC = op.getC(); // Bias
    
    auto dequantA = inputA.getDefiningOp<ONNXDequantizeLinearOp>();
    auto dequantB = inputB.getDefiningOp<ONNXDequantizeLinearOp>();
    
    if (!dequantA || !dequantB) return failure();
    
    Value quantInputA = dequantA.getX();
    Value quantInputB = dequantB.getX();
    
    if (!op.getResult().hasOneUse()) return failure();
    auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(
        *op.getResult().getUsers().begin());
    if (!quantOp) return failure();
    
    auto outputType = mlir::cast<RankedTensorType>(quantOp.getResult().getType());

    auto paramsA = getScalarQuantParams(dequantA);
    auto paramsB = getScalarQuantParams(dequantB);
    auto paramsOut = getScalarQuantParams(quantOp);
    
    // Handle Transpose
    if (op.getTransA()) {
      quantInputA = createExplicitTranspose(rewriter, op.getLoc(), quantInputA);
    }
    if (op.getTransB()) {
      quantInputB = createExplicitTranspose(rewriter, op.getLoc(), quantInputB);
    }

    SmallVector<Value> inputs;
    inputs.push_back(quantInputA);
    inputs.push_back(quantInputB);
    
    // Handle Bias Type Backtracking
    bool hasBias = !mlir::isa<NoneType>(inputC.getType());
    if (hasBias) {
      if (auto dequantC = inputC.getDefiningOp<ONNXDequantizeLinearOp>()) {
          inputs.push_back(dequantC.getX()); // Use the i32 input
      } else {
          inputs.push_back(inputC);
      }
    }

    Value result = createGenericMatMulOp(rewriter, op.getLoc(),
        inputs, outputType,
        paramsA.scale, paramsA.zeroPoint,
        paramsB.scale, paramsB.zeroPoint,
        paramsOut.scale, paramsOut.zeroPoint,
        "npu_gemm");

    rewriter.replaceOp(quantOp, result);
    rewriter.eraseOp(op);
    
    if (dequantA->hasOneUse()) rewriter.eraseOp(dequantA);
    if (dequantB->hasOneUse()) rewriter.eraseOp(dequantB);
    // 尝试清理 Bias 的 Dequant Op
    if (hasBias) {
       if (auto dequantC = inputC.getDefiningOp<ONNXDequantizeLinearOp>()) {
           if (dequantC->hasOneUse()) rewriter.eraseOp(dequantC);
       }
    }
    
    return success();
  }
};

// ============================================================================
// 2. QLinearMatMul Pattern
// ============================================================================
struct QLinearMatMulToLinalg : public OpConversionPattern<ONNXQLinearMatMulOp> {
  using OpConversionPattern<ONNXQLinearMatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXQLinearMatMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    Value inputA = op.getA();
    Value inputB = op.getB();
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    
    float scaleA = getScalarFloat(op.getAScale());
    int64_t zpA = getScalarInt(op.getAZeroPoint());
    float scaleB = getScalarFloat(op.getBScale());
    int64_t zpB = getScalarInt(op.getBZeroPoint());
    float scaleY = getScalarFloat(op.getYScale());
    int64_t zpY = getScalarInt(op.getYZeroPoint());
    
    SmallVector<Value> inputs = {inputA, inputB};
    
    Value result = createGenericMatMulOp(rewriter, op.getLoc(),
        inputs, outputType,
        scaleA, zpA, scaleB, zpB, scaleY, zpY,
        "npu_matmul");
        
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void npux::populateLinalgGemmPattern(RewritePatternSet &patterns) {
  patterns.add<GemmToLinalg, QLinearMatMulToLinalg>(patterns.getContext());
}