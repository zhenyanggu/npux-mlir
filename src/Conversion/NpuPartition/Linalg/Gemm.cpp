//==============================================================
// src/Conversion/NpuPartition/Linalg/Gemm.cpp
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
  // Helper Lambda
  // ==============================================================================
  auto buildGemm = [&](OpBuilder &b, Location loc, Value lhs, Value rhs,
                       Value bias, SmallVector<Value> dynSizes) -> Value {
    int64_t outRank = outType.getRank();
    assert((outRank == 2 || outRank == 3) && "Only 2D and 3D MatMul are supported here");

    // 1. 分配中间 i32 累加器 Buffer (NPU Accumulator)
    auto i32Type = RankedTensorType::get(outType.getShape(), b.getI32Type());
    Value i32Alloc = b.create<bufferization::AllocTensorOp>(loc, i32Type, dynSizes);

    // 2. 构建 Iterators (保证 N, M, K 顺序，3D 时最外层为 B)
    SmallVector<utils::IteratorType> iteratorTypes;
    if (outRank == 2) {
      iteratorTypes = {
          utils::IteratorType::parallel, // N (对应 d0)
          utils::IteratorType::parallel, // M (对应 d1)
          utils::IteratorType::reduction // K (对应 d2)
      };
    } else {
      iteratorTypes = {
          utils::IteratorType::parallel, // B (对应 d0)
          utils::IteratorType::parallel, // N (对应 d1)
          utils::IteratorType::parallel, // M (对应 d2)
          utils::IteratorType::reduction // K (对应 d3)
      };
    }

    // 3. 构建 Indexing Maps
    auto getMap = [&](Value val, bool isA, bool isB, bool isOut) -> AffineMap {
      int64_t rank = val ? mlir::cast<RankedTensorType>(val.getType()).getRank() : outRank;
      SmallVector<AffineExpr> exprs;
      
      if (outRank == 2) {
        auto n = b.getAffineDimExpr(0);
        auto m = b.getAffineDimExpr(1);
        auto k = b.getAffineDimExpr(2);
        if (isA) exprs = {m, k};
        else if (isB) exprs = {k, n};
        else if (isOut) exprs = {m, n};
        else { // Bias (C)
          if (rank == 1) exprs = {n};
          else exprs = {m, n};
        }
      } else { // outRank == 3
        auto b_dim = b.getAffineDimExpr(0);
        auto n = b.getAffineDimExpr(1);
        auto m = b.getAffineDimExpr(2);
        auto k = b.getAffineDimExpr(3);
        
        if (isA) {
          // 兼容 A 可能是 2D Broadcast 的情况
          if (rank == 3) exprs = {b_dim, m, k};
          else exprs = {m, k};
        } else if (isB) {
          // 兼容 B 可能是 2D Broadcast 的情况
          if (rank == 3) exprs = {b_dim, k, n};
          else exprs = {k, n};
        } else if (isOut) {
          exprs = {b_dim, m, n};
        } else { // Bias (C)
          if (rank == 1) exprs = {n};
          else if (rank == 2) exprs = {m, n};
          else exprs = {b_dim, m, n};
        }
      }
      return AffineMap::get(outRank + 1, 0, exprs, b.getContext());
    };

    SmallVector<AffineMap> gemmMaps;
    gemmMaps.push_back(getMap(lhs, true, false, false)); // A
    gemmMaps.push_back(getMap(rhs, false, true, false)); // B
    if (bias) gemmMaps.push_back(getMap(bias, false, false, false)); // C
    gemmMaps.push_back(getMap(nullptr, false, false, true)); // Out (i32)

    SmallVector<Value> gemmInputs = {lhs, rhs};
    if (bias) gemmInputs.push_back(bias);

    // 4. 创建 GenericOp (GEMM -> i32)
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

    gemmOp->setAttr("do_relu", b.getI32IntegerAttr(do_relu));
    if (do_relu == 1) {
      gemmOp->setAttr("relu_type", b.getI32IntegerAttr(relu_type));
    }

    // 5. SPM 阶段：i32 -> 最终类型 (通常为 i8)
    Type finalElemType = outType.getElementType();
    Value outAlloc = b.create<bufferization::AllocTensorOp>(loc, outType, dynSizes);

    SmallVector<AffineMap> quantMaps;
    if (outRank == 2) {
      // 迭代空间: [N, M] (即 d0=N, d1=M)
      // Tensor 物理形状: [M, N]
      // 映射关系: (d0, d1) -> (d1, d0)
      auto d0 = b.getAffineDimExpr(0);
      auto d1 = b.getAffineDimExpr(1);
      auto map = AffineMap::get(2, 0, {d1, d0}, b.getContext());
      quantMaps = {map, map}; // input (i32) 和 output (i8) 的形状相同，Map 也相同
    } else {
      // 迭代空间: [B, N, M] (即 d0=B, d1=N, d2=M)
      // Tensor 物理形状: [B, M, N]
      // 映射关系: (d0, d1, d2) -> (d0, d2, d1)
      auto d0 = b.getAffineDimExpr(0);
      auto d1 = b.getAffineDimExpr(1);
      auto d2 = b.getAffineDimExpr(2);
      auto map = AffineMap::get(3, 0, {d0, d2, d1}, b.getContext());
      quantMaps = {map, map};
    }
    SmallVector<utils::IteratorType> parallelIters(outRank, utils::IteratorType::parallel);

    auto quantOp = b.create<linalg::GenericOp>(loc,
        /*resultTypes=*/outType,
        /*inputs=*/ValueRange{gemmOp.getResult(0)},
        /*outputs=*/ValueRange{outAlloc}, quantMaps, parallelIters,
        [&](OpBuilder &nestedB, Location nestedLoc, ValueRange args) {
          Value inI32 = args[0];
          Value res;
          if (finalElemType.isInteger(32)) {
            res = inI32;
          } else if (finalElemType.isInteger(8)) {
            res = nestedB.create<arith::TruncIOp>(nestedLoc, finalElemType, inI32);
          } else if (mlir::isa<FloatType>(finalElemType)) {
            res = nestedB.create<arith::SIToFPOp>(nestedLoc, finalElemType, inI32);
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
  // 主干逻辑
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

    // 获取输出的动态尺寸
    SmallVector<Value> dynamicSizes =
        getDynamicSizes(rewriter, loc, processedInputs[0], outType.getShape());

    // --- 1. 直接调用通用构建器生成 Linalg 算子 ---
    Value biasVal = hasBias ? processedInputs[2] : nullptr;
    Value result = buildGemm(rewriter, loc, processedInputs[0], processedInputs[1], biasVal, dynamicSizes);
    
    rewriter.create<scf::YieldOp>(loc, result);
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