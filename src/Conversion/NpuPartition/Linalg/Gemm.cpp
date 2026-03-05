//==============================================================
// src/Conversion/NpuPartition/Linalg/MatMul.cpp
// This file implements the conversion of Gemm and QLinearMatMul
// to linalg operations for NPU partitioning.
//==============================================================

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
#include "src/Dialect/ONNX/ONNXOps.hpp"

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
    if (auto dense =
            mlir::dyn_cast<DenseElementsAttr>(constOp.getValueAttr())) {
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
    if (auto dense =
            mlir::dyn_cast<DenseElementsAttr>(constOp.getValueAttr())) {
      if (dense.getElementType().isInteger(8))
        return dense.getValues<int8_t>()[0];
      if (dense.getElementType().isInteger(32))
        return dense.getValues<int32_t>()[0];
      if (dense.getElementType().isInteger(64))
        return dense.getValues<int64_t>()[0];
    }
  }
  return defaultVal;
}

// ==========================================================
// 生成 Linalg 转置 (在 Region 内部)
// ==========================================================
static Value buildLinalgTranspose(OpBuilder &b, Location loc, Value input) {
  auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || inputType.getRank() < 2)
    return input;

  int64_t rank = inputType.getRank();
  SmallVector<int64_t> outShape(inputType.getShape());
  std::swap(outShape[rank - 1], outShape[rank - 2]);
  auto outType = RankedTensorType::get(outShape, inputType.getElementType());

  SmallVector<Value> dynSizes;
  for (int i = 0; i < rank; ++i) {
    int64_t origDim = (i == rank - 1) ? rank - 2 : (i == rank - 2) ? rank - 1 : i;
    if (outShape[i] == ShapedType::kDynamic) {
      dynSizes.push_back(b.create<tensor::DimOp>(loc, input, origDim).getResult());
    }
  }

  Value alloc = b.create<bufferization::AllocTensorOp>(loc, outType, dynSizes);

  SmallVector<AffineExpr> inExprs, outExprs;
  for (int i = 0; i < rank; ++i) {
    inExprs.push_back(b.getAffineDimExpr(i));
    outExprs.push_back(b.getAffineDimExpr(i));
  }
  std::swap(outExprs[rank - 1], outExprs[rank - 2]);

  SmallVector<AffineMap> maps = {
      AffineMap::get(rank, 0, inExprs, b.getContext()),
      AffineMap::get(rank, 0, outExprs, b.getContext())};

  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);

  auto genericOp = b.create<linalg::GenericOp>(loc, outType,
      /*inputs=*/input,
      /*outputs=*/alloc, maps, iters,
      [&](OpBuilder &nestedB, Location nestedLoc, ValueRange args) {
        Value in = args[0];
        Type elemType = in.getType();
        Value res = in;
        // 按照用户需求，用 addi 构造 dummy body 
        if (elemType.isIntOrIndex()) {
          res = nestedB.create<arith::AddIOp>(nestedLoc, in, in);
        } else if (mlir::isa<FloatType>(elemType)) {
          res = nestedB.create<arith::AddFOp>(nestedLoc, in, in);
        }
        nestedB.create<linalg::YieldOp>(nestedLoc, res);
      });

  genericOp->setAttr("library_call", b.getStringAttr("npu_transpose"));
  genericOp->setAttr("npu.target", b.getStringAttr("npu"));

  return genericOp.getResult(0);
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
    if (t.isInteger(32))
      return v;
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
static Value createGenericMatMulOp(ConversionPatternRewriter &rewriter,
    Location loc,
    SmallVector<Value> inputs, // [A, B] 或 [A, B, C]
    RankedTensorType outType,  // Final Output Type (e.g., i8)
    // Quant Params
    float lhsScale, int64_t lhsZp, float rhsScale, int64_t rhsZp,
    float outScale, int64_t outZp, StringRef libCallName,
    int64_t do_relu = 0, int64_t relu_type = 0,
    bool transA = false, bool transB = false) {
  int64_t outRank = outType.getRank();
  assert(outRank >= 2 && "MatMul output rank must be >= 2");
  bool hasBias = (inputs.size() == 3);
  int32_t withBiasAttr = hasBias ? 1 : 0;

  // ==============================================================================
  // Helper Lambda: 负责构建纯 2D 的 GEMM (Linalg Generic) 及后续的 SPM 转换
  // 参数接收的 lhs2D, rhs2D 必须是已经被降维到 2D 的 Tensor (或者 1D Bias)
  // ==============================================================================
  auto build2DGemm = [&](OpBuilder &b, Location loc, Value lhs2D, Value rhs2D,
                         Value bias2D) -> Value {
    auto lhsType = mlir::cast<RankedTensorType>(lhs2D.getType());
    auto rhsType = mlir::cast<RankedTensorType>(rhs2D.getType());

    int64_t M = lhsType.getShape()[0];
    int64_t N = rhsType.getShape()[1];

    // 1. 获取动态尺寸（处理形状为 ? 的情况）
    SmallVector<Value> dynSizesI32;
    if (M == ShapedType::kDynamic)
      dynSizesI32.push_back(b.create<tensor::DimOp>(loc, lhs2D, 0).getResult());
    if (N == ShapedType::kDynamic)
      dynSizesI32.push_back(b.create<tensor::DimOp>(loc, rhs2D, 1).getResult());

    // 2. 分配中间 i32 累加器 Buffer (NPU Accumulator)
    auto i32Type = RankedTensorType::get({M, N}, b.getI32Type());
    Value i32Alloc =
        b.create<bufferization::AllocTensorOp>(loc, i32Type, dynSizesI32);

    // 3. 构建 2D GEMM Indexing Maps
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel, // M
        utils::IteratorType::parallel, // N
        utils::IteratorType::reduction // K
    };

    auto get2DMap = [&](bool isA, bool isB, bool isC,
                        Value val = Value()) -> AffineMap {
      auto m = b.getAffineDimExpr(0);
      auto n = b.getAffineDimExpr(1);
      auto k = b.getAffineDimExpr(2);
      if (isA)
        return AffineMap::get(3, 0, {m, k}, b.getContext());
      if (isB)
        return AffineMap::get(3, 0, {k, n}, b.getContext());
      if (isC) {
        int64_t biasRank =
            mlir::cast<RankedTensorType>(val.getType()).getRank();
        // 如果 Bias 是 1D，映射到列方向 {n}，否则映射到 {m, n}
        if (biasRank == 1)
          return AffineMap::get(3, 0, {n}, b.getContext());
        return AffineMap::get(3, 0, {m, n}, b.getContext());
      }
      return AffineMap::get(3, 0, {m, n}, b.getContext());
    };

    SmallVector<AffineMap> gemmMaps;
    gemmMaps.push_back(get2DMap(true, false, false)); // A
    gemmMaps.push_back(get2DMap(false, true, false)); // B
    if (bias2D)
      gemmMaps.push_back(get2DMap(false, false, true, bias2D)); // C
    gemmMaps.push_back(get2DMap(false, false, false));          // Out (i32)

    SmallVector<Value> gemmInputs = {lhs2D, rhs2D};
    if (bias2D)
      gemmInputs.push_back(bias2D);

    // 4. 创建 GenericOp (2D GEMM -> i32)
    auto gemmOp = b.create<linalg::GenericOp>(loc,
        /*resultTypes=*/i32Type,
        /*inputs=*/gemmInputs,
        /*outputs=*/i32Alloc, gemmMaps, iteratorTypes,
        /*bodyBuilder=*/createI32MatMulBody);

    gemmOp->setAttr("library_call", b.getStringAttr(libCallName));
    gemmOp->setAttr("npu.target", b.getStringAttr("npu"));
    gemmOp->setAttr("lhs_scale", b.getF32FloatAttr(lhsScale));
    gemmOp->setAttr("lhs_zp", b.getIntegerAttr(b.getI32Type(), lhsZp));
    gemmOp->setAttr("rhs_scale", b.getF32FloatAttr(rhsScale));
    gemmOp->setAttr("rhs_zp", b.getIntegerAttr(b.getI32Type(), rhsZp));
    gemmOp->setAttr("out_scale", b.getF32FloatAttr(outScale));
    gemmOp->setAttr("out_zp", b.getIntegerAttr(b.getI32Type(), outZp));
    gemmOp->setAttr("with_bias", b.getI32IntegerAttr(withBiasAttr));

    // 写 Relu Fusion 属性
    gemmOp->setAttr("do_relu", b.getI32IntegerAttr(do_relu));
    if (do_relu == 1) {
      gemmOp->setAttr("relu_type", b.getI32IntegerAttr(relu_type));
    }

    // 5. SPM 阶段：i32 -> 最终类型 (通常为 i8)
    Type finalElemType = outType.getElementType();
    auto out2DType = RankedTensorType::get({M, N}, finalElemType);
    Value out2DAlloc =
        b.create<bufferization::AllocTensorOp>(loc, out2DType, dynSizesI32);

    SmallVector<AffineMap> identityMaps(2, b.getMultiDimIdentityMap(2));
    SmallVector<utils::IteratorType> parallelIters(
        2, utils::IteratorType::parallel);

    auto quantOp = b.create<linalg::GenericOp>(loc,
        /*resultTypes=*/out2DType,
        /*inputs=*/ValueRange{gemmOp.getResult(0)},
        /*outputs=*/ValueRange{out2DAlloc}, identityMaps, parallelIters,
        [&](OpBuilder &nestedB, Location nestedLoc, ValueRange args) {
          Value inI32 = args[0];
          Value res;
          if (finalElemType.isInteger(32)) {
            res = inI32;
          } else if (finalElemType.isInteger(8)) {
            res = nestedB.create<arith::TruncIOp>(
                nestedLoc, finalElemType, inI32);
          } else if (mlir::isa<FloatType>(finalElemType)) {
            res = nestedB.create<arith::SIToFPOp>(
                nestedLoc, finalElemType, inI32);
          } else {
            res = inI32;
          }
          nestedB.create<linalg::YieldOp>(nestedLoc, res);
        });

    quantOp->setAttr("library_call", b.getStringAttr("mv_acc_to_spm"));
    quantOp->setAttr("npu.target", b.getStringAttr("npu"));

    return quantOp.getResult(0);
  };

  // ==============================================================================
  // 主干逻辑：根据 Rank 选择是否包裹 scf.for
  // ==============================================================================
  auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(loc, outType);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Block *body = rewriter.createBlock(&executeRegion.getRegion());

    // --- 0. 在 Region 内部执行输入端 Transpose (Linalg Generic) ---
    Value actualA = inputs[0];
    if (transA) {
      actualA = buildLinalgTranspose(rewriter, loc, actualA);
    }
    Value actualB = inputs[1];
    if (transB) {
      actualB = buildLinalgTranspose(rewriter, loc, actualB);
    }

    SmallVector<Value> processedInputs = {actualA, actualB};
    if (inputs.size() > 2) {
      processedInputs.push_back(inputs[2]);
    }

    SmallVector<Value> dynamicSizes =
        getDynamicSizes(rewriter, loc, processedInputs[0], outType.getShape());

    if (outRank == 2) {
      // --- 纯 2D MatMul，直接调用 2D 构建器 ---
      Value biasVal = hasBias ? processedInputs[2] : nullptr;
      Value res2D = build2DGemm(rewriter, loc, processedInputs[0], processedInputs[1], biasVal);
      rewriter.create<scf::YieldOp>(loc, res2D);
    } else if (outRank == 3) {
      // --- 3D Batched MatMul，在外层 (Batch) 循环并做 Slice ---
      int64_t B = outType.getShape()[0];
      Value finalAlloc = rewriter.create<bufferization::AllocTensorOp>(
          loc, outType, dynamicSizes);

      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      Value bUpper;
      if (B == ShapedType::kDynamic) {
        bUpper = rewriter.create<tensor::DimOp>(loc, finalAlloc, 0).getResult();
      } else {
        bUpper = rewriter.create<arith::ConstantIndexOp>(loc, B);
      }

      // 获取 OpFoldResult 类型的尺寸 (兼顾静态与动态)
      auto getDimOpFoldResult = [&](OpBuilder &builder, Value tensor,
                                    int dim) -> OpFoldResult {
        auto shape = mlir::cast<RankedTensorType>(tensor.getType()).getShape();
        if (shape[dim] == ShapedType::kDynamic)
          return builder.create<tensor::DimOp>(loc, tensor, dim).getResult();
        return builder.getIndexAttr(shape[dim]);
      };

      auto bLoop = rewriter.create<scf::ForOp>(loc, zero, bUpper, one,
          ValueRange{finalAlloc},
           [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            Value outAccum = iterArgs[0];

            OpFoldResult ivAttr = iv;
            OpFoldResult zeroAttr = b.getIndexAttr(0);
            OpFoldResult oneAttr = b.getIndexAttr(1);

            // 1. 提取 LHS 的 2D 切片 (Rank-Reduction: 3D -> 2D)
            Value lhs2D = processedInputs[0];
            auto lhsType = mlir::cast<RankedTensorType>(lhs2D.getType());
            if (lhsType.getRank() == 3) {
              OpFoldResult mSize = getDimOpFoldResult(b, lhs2D, 1);
              OpFoldResult kSize = getDimOpFoldResult(b, lhs2D, 2);
              SmallVector<OpFoldResult> offsets = {ivAttr, zeroAttr, zeroAttr};
              SmallVector<OpFoldResult> sizes = {oneAttr, mSize, kSize};
              SmallVector<OpFoldResult> strides = {oneAttr, oneAttr, oneAttr};

              auto lhs2DType = RankedTensorType::get(
                  {lhsType.getShape()[1], lhsType.getShape()[2]},
                  lhsType.getElementType());
              lhs2D = b.create<tensor::ExtractSliceOp>(
                  loc, lhs2DType, processedInputs[0], offsets, sizes, strides);
            }

            // 2. 提取 RHS 的 2D 切片 (兼容 Broadcast, 例如 RHS 本来就是 2D)
            Value rhs2D = processedInputs[1];
            auto rhsType = mlir::cast<RankedTensorType>(rhs2D.getType());
            if (rhsType.getRank() == 3) {
              OpFoldResult kSize = getDimOpFoldResult(b, rhs2D, 1);
              OpFoldResult nSize = getDimOpFoldResult(b, rhs2D, 2);
              SmallVector<OpFoldResult> offsets = {ivAttr, zeroAttr, zeroAttr};
              SmallVector<OpFoldResult> sizes = {oneAttr, kSize, nSize};
              SmallVector<OpFoldResult> strides = {oneAttr, oneAttr, oneAttr};

              auto rhs2DType = RankedTensorType::get(
                  {rhsType.getShape()[1], rhsType.getShape()[2]},
                  rhsType.getElementType());
              rhs2D = b.create<tensor::ExtractSliceOp>(
                  loc, rhs2DType, processedInputs[1], offsets, sizes, strides);
            }

            // 3. 提取 Bias (如果有，且恰好也是 Batched 的 3D)
            Value biasSlice = nullptr;
            if (hasBias) {
              biasSlice = processedInputs[2];
              auto biasType = mlir::cast<RankedTensorType>(biasSlice.getType());
              if (biasType.getRank() == 3) {
                OpFoldResult mSize = getDimOpFoldResult(b, biasSlice, 1);
                OpFoldResult nSize = getDimOpFoldResult(b, biasSlice, 2);
                SmallVector<OpFoldResult> offsets = {
                    ivAttr, zeroAttr, zeroAttr};
                SmallVector<OpFoldResult> sizes = {oneAttr, mSize, nSize};
                SmallVector<OpFoldResult> strides = {oneAttr, oneAttr, oneAttr};

                auto bias2DType = RankedTensorType::get(
                    {biasType.getShape()[1], biasType.getShape()[2]},
                    biasType.getElementType());
                biasSlice = b.create<tensor::ExtractSliceOp>(
                    loc, bias2DType, processedInputs[2], offsets, sizes, strides);
              }
            }

            // 4. 计算 2D GEMM (调用 Lambda 返回 2D)
            Value res2D = build2DGemm(b, loc, lhs2D, rhs2D, biasSlice);

            // 5. 塞回 3D 结果容器 (Rank-Up: 2D 插入回 3D 的某个切片)
            OpFoldResult mSizeOut = getDimOpFoldResult(b, outAccum, 1);
            OpFoldResult nSizeOut = getDimOpFoldResult(b, outAccum, 2);
            SmallVector<OpFoldResult> offsetsOut = {ivAttr, zeroAttr, zeroAttr};
            SmallVector<OpFoldResult> sizesOut = {oneAttr, mSizeOut, nSizeOut};
            SmallVector<OpFoldResult> stridesOut = {oneAttr, oneAttr, oneAttr};

            Value updatedAccum = b.create<tensor::InsertSliceOp>(
                loc, res2D, outAccum, offsetsOut, sizesOut, stridesOut);
            b.create<scf::YieldOp>(loc, updatedAccum);
          });

      rewriter.create<scf::YieldOp>(loc, bLoop.getResult(0));
    }
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

    if (!dequantA || !dequantB)
      return failure();

    Value quantInputA = dequantA.getX();
    Value quantInputB = dequantB.getX();

    if (!op.getResult().hasOneUse())
      return failure();
    auto quantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(
        *op.getResult().getUsers().begin());
    if (!quantOp)
      return failure();

    // =========================================================================
    // 前向搜索 Relu 融合 (Gemm -> Q -> DQ -> Act -> Q)
    // =========================================================================
    int64_t do_relu = 0;
    int64_t relu_type = 0;
    ONNXQuantizeLinearOp finalQuantOp = quantOp;
    SmallVector<Operation *> fusionOpsToErase;

    if (quantOp.getResult().hasOneUse()) {
      if (auto dqOp = mlir::dyn_cast<ONNXDequantizeLinearOp>(
              *quantOp.getResult().getUsers().begin())) {
        if (dqOp.getResult().hasOneUse()) {
          Operation *actOp = *dqOp.getResult().getUsers().begin();
          bool is_relu = mlir::isa<ONNXReluOp>(actOp);
          bool is_leaky = mlir::isa<ONNXLeakyReluOp>(actOp);

          if ((is_relu || is_leaky) && actOp->getResult(0).hasOneUse()) {
            if (auto finalQOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(
                    *actOp->getResult(0).getUsers().begin())) {
              bool can_fuse = true;
              if (is_leaky) {
                auto leakyOp = mlir::cast<ONNXLeakyReluOp>(actOp);
                float alpha = 0.01f;
                if (auto alphaAttr = leakyOp.getAlphaAttr()) {
                  alpha = alphaAttr.getValueAsDouble();
                }
                if (std::abs(alpha - 0.1) < 1e-5) relu_type = 2;
                else if (std::abs(alpha - 0.2) < 1e-5) relu_type = 3;
                else if (std::abs(alpha - 0.01) < 1e-5) relu_type = 4;
                else {
                  llvm::errs() << "Warning: unsupported LeakyRelu alpha " << alpha
                               << ", skipping fusion.\n";
                  can_fuse = false;
                }
              } else {
                relu_type = 0; // relu
              }

              if (can_fuse) {
                do_relu = 1;
                finalQuantOp = finalQOp;
                fusionOpsToErase.push_back(quantOp);
                fusionOpsToErase.push_back(dqOp);
                fusionOpsToErase.push_back(actOp);
              }
            }
          }
        }
      }
    }

    auto outputType =
        mlir::cast<RankedTensorType>(finalQuantOp.getResult().getType());

    auto paramsA = getScalarQuantParams(dequantA);
    auto paramsB = getScalarQuantParams(dequantB);
    auto paramsOut = getScalarQuantParams(finalQuantOp);

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

    Value result = createGenericMatMulOp(rewriter, op.getLoc(), inputs,
        outputType, paramsA.scale, paramsA.zeroPoint, paramsB.scale,
        paramsB.zeroPoint, paramsOut.scale, paramsOut.zeroPoint, "npu_gemm",
        do_relu, relu_type, op.getTransA(), op.getTransB());

    // =========================================================================
    // 安全擦除
    // =========================================================================
    llvm::SmallPtrSet<Operation*, 4> opsErased;
    auto safeErase = [&](Operation* opToErase) {
      if (opToErase && opsErased.insert(opToErase).second) {
        rewriter.eraseOp(opToErase);
      }
    };

    rewriter.replaceOp(finalQuantOp, result);
    opsErased.insert(finalQuantOp);

    safeErase(op); // GemmOp

    for (auto *fuseOp : fusionOpsToErase) {
      safeErase(fuseOp);
    }

    if (dequantA->hasOneUse()) safeErase(dequantA);
    if (dequantB->hasOneUse()) safeErase(dequantB);

    if (hasBias) {
      if (auto dequantC = inputC.getDefiningOp<ONNXDequantizeLinearOp>()) {
        if (dequantC->hasOneUse()) safeErase(dequantC);
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

    // =========================================================================
    // 前向搜索 Relu 融合 (QLinearMatMul -> DQ -> Act -> Q)
    // =========================================================================
    int64_t do_relu = 0;
    int64_t relu_type = 0;
    Operation* replaceTarget = op;
    SmallVector<Operation*> fusionOpsToErase;

    if (op.getResult().hasOneUse()) {
      if (auto dqOp = mlir::dyn_cast<ONNXDequantizeLinearOp>(
              *op.getResult().getUsers().begin())) {
        if (dqOp.getResult().hasOneUse()) {
          Operation *actOp = *dqOp.getResult().getUsers().begin();
          bool is_relu = mlir::isa<ONNXReluOp>(actOp);
          bool is_leaky = mlir::isa<ONNXLeakyReluOp>(actOp);

          if ((is_relu || is_leaky) && actOp->getResult(0).hasOneUse()) {
            if (auto finalQOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(
                    *actOp->getResult(0).getUsers().begin())) {
              bool can_fuse = true;
              if (is_leaky) {
                auto leakyOp = mlir::cast<ONNXLeakyReluOp>(actOp);
                float alpha = 0.01f;
                if (auto alphaAttr = leakyOp.getAlphaAttr()) {
                  alpha = alphaAttr.getValueAsDouble();
                }
                if (std::abs(alpha - 0.1) < 1e-5) relu_type = 2;
                else if (std::abs(alpha - 0.2) < 1e-5) relu_type = 3;
                else if (std::abs(alpha - 0.01) < 1e-5) relu_type = 4;
                else {
                  llvm::errs() << "Warning: unsupported LeakyRelu alpha " << alpha
                               << ", skipping fusion.\n";
                  can_fuse = false;
                }
              } else {
                relu_type = 0; // relu
              }

              if (can_fuse) {
                do_relu = 1;
                replaceTarget = finalQOp;

                // 使用最后 Quantize 的 Params 和 Type 作为整个 fused block 的输出
                auto qParams = getScalarQuantParams(finalQOp);
                scaleY = qParams.scale;
                zpY = qParams.zeroPoint;
                outputType = mlir::cast<RankedTensorType>(finalQOp.getResult().getType());

                fusionOpsToErase.push_back(dqOp);
                fusionOpsToErase.push_back(actOp);
              }
            }
          }
        }
      }
    }

    SmallVector<Value> inputs = {inputA, inputB};

    Value result = createGenericMatMulOp(rewriter, op.getLoc(), inputs,
        outputType, scaleA, zpA, scaleB, zpB, scaleY, zpY, "npu_matmul",
        do_relu, relu_type, false, false); // QLinearMatMul 无 transA/transB

    // =========================================================================
    // 安全擦除
    // =========================================================================
    llvm::SmallPtrSet<Operation*, 4> opsErased;
    auto safeErase = [&](Operation* opToErase) {
      if (opToErase && opsErased.insert(opToErase).second) {
        rewriter.eraseOp(opToErase);
      }
    };

    rewriter.replaceOp(replaceTarget, result);
    opsErased.insert(replaceTarget);

    safeErase(op); // erase original QLinearMatMul

    for (auto *fuseOp : fusionOpsToErase) {
      safeErase(fuseOp);
    }

    return success();
  }
};

} // namespace

void npux::populateLinalgGemmPattern(RewritePatternSet &patterns) {
  patterns.add<GemmToLinalg, QLinearMatMulToLinalg>(patterns.getContext());
}