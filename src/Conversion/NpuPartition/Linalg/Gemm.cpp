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
  if (!v)
    return defaultVal;

  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
  }

  if (auto constOp = v.getDefiningOp<ONNXConstantOp>()) {
    if (auto dense =
            mlir::dyn_cast<DenseElementsAttr>(constOp.getValueAttr())) {
      auto elemType = dense.getElementType();

      // 必须先检查无符号类型，因为 isInteger(8) 会包含 isUnsignedInteger(8)
      if (elemType.isUnsignedInteger(8))
        return dense.getValues<uint8_t>()[0];
      if (elemType.isInteger(8))
        return dense.getValues<int8_t>()[0];

      if (elemType.isUnsignedInteger(16))
        return dense.getValues<uint16_t>()[0];
      if (elemType.isInteger(16))
        return dense.getValues<int16_t>()[0];

      if (elemType.isUnsignedInteger(32))
        return dense.getValues<uint32_t>()[0];
      if (elemType.isInteger(32))
        return dense.getValues<int32_t>()[0];

      if (elemType.isUnsignedInteger(64))
        return dense.getValues<uint64_t>()[0];
      if (elemType.isInteger(64))
        return dense.getValues<int64_t>()[0];
    }
  }
  return defaultVal;
}

// 辅助函数：为 RankedTensorType 添加 encoding=1
static RankedTensorType addEncoding1(RankedTensorType type, OpBuilder &b) {
  if (type.getEncoding()) {
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(type.getEncoding())) {
      if (intAttr.getInt() == 1)
        return type;
    }
  }
  return RankedTensorType::get(
      type.getShape(), type.getElementType(), b.getI64IntegerAttr(1));
}

// 辅助函数：处理 linalg 输入的 encoding 和 constant 的拷贝
static Value processNpuLinalgInput(OpBuilder &b, Location loc, Value v) {
  if (!v)
    return v;
  auto type = mlir::dyn_cast<RankedTensorType>(v.getType());
  if (!type)
    return v;

  if (isa<BlockArgument>(v)) {
    return v;
  }

  auto encType = addEncoding1(type, b);

  // 如果是常数，插入 linalg.copy
  if (v.getDefiningOp<ONNXConstantOp>()) {
    SmallVector<Value> dynSizes;
    for (int i = 0; i < type.getRank(); ++i) {
      if (type.isDynamicDim(i)) {
        dynSizes.push_back(b.create<tensor::DimOp>(loc, v, i).getResult());
      }
    }
    Value alloc =
        b.create<bufferization::AllocTensorOp>(loc, encType, dynSizes);
    auto copyOp = b.create<linalg::CopyOp>(loc, v, alloc);
    return copyOp.getResult(0);
  } else {
    // 否则直接强制类型转换 encoding=1
    v.setType(encType);
    return v;
  }
}

// ==========================================================
// 生成 Linalg 转置
// ==========================================================
static Value buildLinalgTranspose(OpBuilder &b, Location loc, Value input,
    Operation *sourceOp, StringRef layerName, ArrayRef<StringRef> fusedOps) {
  auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || inputType.getRank() < 2)
    return input;

  int64_t rank = inputType.getRank();
  SmallVector<int64_t> outShape(inputType.getShape());
  std::swap(outShape[rank - 1], outShape[rank - 2]);

  // 输出带有 encoding=1
  auto outType = RankedTensorType::get(outShape, inputType.getElementType());
  auto encOutType = addEncoding1(outType, b);

  SmallVector<Value> dynSizes;
  for (int i = 0; i < rank; ++i) {
    int64_t origDim = (i == rank - 1)   ? rank - 2
                      : (i == rank - 2) ? rank - 1
                                        : i;
    if (outShape[i] == ShapedType::kDynamic) {
      dynSizes.push_back(
          b.create<tensor::DimOp>(loc, input, origDim).getResult());
    }
  }

  Value alloc =
      b.create<bufferization::AllocTensorOp>(loc, encOutType, dynSizes);

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

  auto genericOp = b.create<linalg::GenericOp>(loc, encOutType,
      /*inputs=*/input,
      /*outputs=*/alloc, maps, iters,
      [&](OpBuilder &nestedB, Location nestedLoc, ValueRange args) {
        Value in = args[0];
        Type elemType = in.getType();
        Value res = in;
        // 按照用户需求，用 addi 构造 dummy body
        if (elemType.isIntOrIndex()) {
          if (elemType.isSignlessInteger()) {
            res = nestedB.create<arith::AddIOp>(nestedLoc, in, in);
          } else {
            Type signlessType = nestedB.getIntegerType(
                elemType.getIntOrFloatBitWidth());
            Value signlessInput = nestedB
                                      .create<UnrealizedConversionCastOp>(
                                          nestedLoc, signlessType, in)
                                      .getResult(0);
            Value sum = nestedB.create<arith::AddIOp>(
                nestedLoc, signlessInput, signlessInput);
            res = nestedB
                      .create<UnrealizedConversionCastOp>(
                          nestedLoc, elemType, sum)
                      .getResult(0);
          }
        } else if (mlir::isa<FloatType>(elemType)) {
          res = nestedB.create<arith::AddFOp>(nestedLoc, in, in);
        }
        nestedB.create<linalg::YieldOp>(nestedLoc, res);
      });

  genericOp->setAttr("library_call", b.getStringAttr("npu_transpose"));
  genericOp->setAttr("npu.target", b.getStringAttr("npu"));
  setNpuProfileAttrs(genericOp, sourceOp, b, layerName, "layout_in", fusedOps);

  return genericOp.getResult(0);
}

// MatMul i32 Body 构建器
static void createI32MatMulBody(OpBuilder &b, Location loc, ValueRange args) {
  Value lhs = args[0];
  Value rhs = args[1];
  Value outAcc = args.back();
  bool hasBias = (args.size() == 4);

  auto castToI32 = [&](Value v) -> Value {
    Type t = v.getType();

    // 1. 处理浮点数
    if (mlir::isa<FloatType>(t)) {
      return b.create<arith::FPToSIOp>(loc, b.getI32Type(), v);
    }

    // 2. 处理所有整数类型
    if (t.isIntOrIndex()) {
      unsigned bitwidth = t.getIntOrFloatBitWidth();
      Value signlessVal = v;

      // 步骤 A：如果类型带有符号 (ui8, si8, ui32 等)，强制剥离为 Signless (i8,
      // i32)
      if (!t.isSignlessInteger()) {
        Type signlessType = b.getIntegerType(bitwidth);
        signlessVal = b.create<UnrealizedConversionCastOp>(loc, signlessType, v)
                          .getResult(0);
      }

      // 步骤 B：如果是 32 位，剥离符号后直接返回
      if (bitwidth == 32) {
        return signlessVal;
      }

      // 步骤 C：根据原始类型是有符号还是无符号，选择对应的扩展指令
      if (t.isUnsignedInteger()) {
        // 原始类型是 ui8/ui16，对 signlessVal 执行无符号零扩展 (Zero Extension)
        return b.create<arith::ExtUIOp>(loc, b.getI32Type(), signlessVal);
      } else {
        // 原始类型是 si8/i8，对 signlessVal 执行有符号扩展 (Sign Extension)
        return b.create<arith::ExtSIOp>(loc, b.getI32Type(), signlessVal);
      }
    }

    return v;
  };

  Value lhsI32 = castToI32(lhs);
  Value rhsI32 = castToI32(rhs);
  Value outI32 = castToI32(outAcc);

  Value mul = b.create<arith::MulIOp>(loc, lhsI32, rhsI32);

  if (hasBias) {
    Value biasI32 = castToI32(args[2]);
    mul = b.create<arith::AddIOp>(loc, biasI32, mul);
  }

  Value resI32 = b.create<arith::AddIOp>(loc, outI32, mul);
  b.create<linalg::YieldOp>(loc, resI32);
}
// 通用的 MatMul 构建器
static Value createGenericMatMulOp(ConversionPatternRewriter &rewriter,
    Location loc,
    SmallVector<Value> inputs, // [A, B] 或 [A, B, C]
    RankedTensorType outType,  // Final Output Type
    float lhsScale, int64_t lhsZp, float rhsScale, int64_t rhsZp,
    float outScale, int64_t outZp, StringRef libCallName, int64_t do_relu = 0,
    int64_t relu_type = 0, bool transA = false, bool transB = false,
    Operation *sourceOp = nullptr, StringRef layerName = {},
    ArrayRef<StringRef> fusedOps = {}, bool skipSPM = false,
    bool isGraphOutput = false) {
  int64_t outRank = outType.getRank();
  assert(outRank >= 2 && "MatMul output rank must be >= 2");
  bool hasBias = (inputs.size() == 3);
  int32_t withBiasAttr = hasBias ? 1 : 0;

  auto buildGemm = [&](OpBuilder &b, Location loc, Value lhs, Value rhs,
                       Value bias, SmallVector<Value> dynSizes) -> Value {
    int64_t outRank = outType.getRank();
    assert((outRank >= 2 && outRank <= 4) &&
           "Only 2D, 3D and 4D MatMul are supported here");

    // 1. 分配中间带有 encoding=1 的 i32 累加器 Buffer
    auto i32Type = RankedTensorType::get(outType.getShape(), b.getI32Type());
    RankedTensorType encI32Type = (skipSPM && isGraphOutput) ? 
                                   i32Type : addEncoding1(i32Type, b);
    Value i32Alloc =
        b.create<bufferization::AllocTensorOp>(loc, encI32Type, dynSizes);

    // 2. 构建 Iterators
    SmallVector<utils::IteratorType> iteratorTypes;
    if (outRank == 2) {
      iteratorTypes = {utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::reduction};
    } else if (outRank == 3) {
      iteratorTypes = {utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::reduction};
    } else { // outRank == 4
      iteratorTypes = {utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::reduction};
    }

    // 3. 构建 Indexing Maps
    auto getMap = [&](Value val, bool isA, bool isB, bool isOut) -> AffineMap {
      int64_t rank =
          val ? mlir::cast<RankedTensorType>(val.getType()).getRank() : outRank;
      SmallVector<AffineExpr> exprs;

      if (outRank == 2) {
        auto n = b.getAffineDimExpr(0);
        auto m = b.getAffineDimExpr(1);
        auto k = b.getAffineDimExpr(2);
        if (isA)
          exprs = {m, k};
        else if (isB)
          exprs = {k, n};
        else if (isOut)
          exprs = {m, n};
        else {
          if (rank == 1)
            exprs = {n};
          else
            exprs = {m, n};
        }
      } else if (outRank == 3) {
        auto b_dim = b.getAffineDimExpr(0);
        auto n = b.getAffineDimExpr(1);
        auto m = b.getAffineDimExpr(2);
        auto k = b.getAffineDimExpr(3);

        if (isA) {
          if (rank == 3)
            exprs = {b_dim, m, k};
          else
            exprs = {m, k};
        } else if (isB) {
          if (rank == 3)
            exprs = {b_dim, k, n};
          else
            exprs = {k, n};
        } else if (isOut) {
          exprs = {b_dim, m, n};
        } else {
          if (rank == 1)
            exprs = {n};
          else if (rank == 2)
            exprs = {m, n};
          else
            exprs = {b_dim, m, n};
        }
      } else {
        auto b1_dim = b.getAffineDimExpr(0);
        auto b2_dim = b.getAffineDimExpr(1);
        auto n = b.getAffineDimExpr(2); // 沿用原代码的 N, M, K 相对序号
        auto m = b.getAffineDimExpr(3);
        auto k = b.getAffineDimExpr(4);

        if (isA) {
          if (rank == 4)
            exprs = {b1_dim, b2_dim, m, k};
          else if (rank == 3)
            exprs = {b2_dim, m, k};
          else
            exprs = {m, k};
        } else if (isB) {
          if (rank == 4)
            exprs = {b1_dim, b2_dim, k, n};
          else if (rank == 3)
            exprs = {b2_dim, k, n};
          else
            exprs = {k, n};
        } else if (isOut) {
          exprs = {b1_dim, b2_dim, m, n};
        } else { // Bias fallback
          if (rank == 1)
            exprs = {n};
          else if (rank == 2)
            exprs = {m, n};
          else if (rank == 3)
            exprs = {b2_dim, m, n};
          else
            exprs = {b1_dim, b2_dim, m, n};
        }
      }
      return AffineMap::get(outRank + 1, 0, exprs, b.getContext());
    };

    SmallVector<AffineMap> gemmMaps;
    gemmMaps.push_back(getMap(lhs, true, false, false));
    gemmMaps.push_back(getMap(rhs, false, true, false));
    if (bias)
      gemmMaps.push_back(getMap(bias, false, false, false));
    gemmMaps.push_back(getMap(nullptr, false, false, true));

    SmallVector<Value> gemmInputs = {lhs, rhs};
    if (bias)
      gemmInputs.push_back(bias);

    // 4. 创建 GenericOp (GEMM -> i32 encoding=1)
    auto gemmOp = b.create<linalg::GenericOp>(loc,
        /*resultTypes=*/encI32Type,
        /*inputs=*/gemmInputs,
        /*outputs=*/i32Alloc, gemmMaps, iteratorTypes,
        /*bodyBuilder=*/createI32MatMulBody);

    gemmOp->setAttr("library_call", b.getStringAttr(libCallName));
    gemmOp->setAttr("npu.target", b.getStringAttr("npu"));
    setNpuProfileAttrs(gemmOp, sourceOp, b, layerName, "compute", fusedOps);
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

    if (skipSPM) {
      return gemmOp.getResult(0);
    }

    // 5. SPM 阶段：i32 -> 最终带有 encoding=1 的输出类型
    Type finalElemType = outType.getElementType();
    RankedTensorType encOutType =
        isGraphOutput ? outType : addEncoding1(outType, b);
    Value outAlloc =
        b.create<bufferization::AllocTensorOp>(loc, encOutType, dynSizes);

    SmallVector<AffineMap> quantMaps;
    if (outRank == 2) {
      auto d0 = b.getAffineDimExpr(0);
      auto d1 = b.getAffineDimExpr(1);
      auto map = AffineMap::get(2, 0, {d1, d0}, b.getContext());
      quantMaps = {map, map};
    } else if (outRank == 3) {
      auto d0 = b.getAffineDimExpr(0);
      auto d1 = b.getAffineDimExpr(1);
      auto d2 = b.getAffineDimExpr(2);
      auto map = AffineMap::get(3, 0, {d0, d2, d1}, b.getContext());
      quantMaps = {map, map};
    } else { // 新增的 4D 逻辑
      auto d0 = b.getAffineDimExpr(0);
      auto d1 = b.getAffineDimExpr(1);
      auto d2 = b.getAffineDimExpr(2);
      auto d3 = b.getAffineDimExpr(3);
      // 保持尾部维度交错的 NPU 特性: (b1, b2, m, n) -> (b1, b2, n, m)
      auto map = AffineMap::get(4, 0, {d0, d1, d3, d2}, b.getContext());
      quantMaps = {map, map};
    }
    SmallVector<utils::IteratorType> parallelIters(
        outRank, utils::IteratorType::parallel);

    auto quantOp = b.create<linalg::GenericOp>(loc,
        /*resultTypes=*/encOutType,
        /*inputs=*/ValueRange{gemmOp.getResult(0)},
        /*outputs=*/ValueRange{outAlloc}, quantMaps, parallelIters,
        [&](OpBuilder &nestedB, Location nestedLoc, ValueRange args) {
          Value inI32 = args[0];
          Value res;
          if (finalElemType.isInteger(32)) {
            res = inI32;
          } else if (finalElemType.isInteger(8)) {
            // Arithmetic integer ops require a signless result type. Preserve
            // ONNX's i8/ui8 element type with a cast at the linalg boundary.
            Type signlessI8 = nestedB.getIntegerType(8);
            Value truncated = nestedB.create<arith::TruncIOp>(
                nestedLoc, signlessI8, inI32);
            res = finalElemType == signlessI8
                      ? truncated
                      : nestedB
                            .create<UnrealizedConversionCastOp>(
                                nestedLoc, finalElemType, truncated)
                            .getResult(0);
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
    setNpuProfileAttrs(quantOp, sourceOp, b, layerName, "quant", fusedOps);

    return quantOp.getResult(0);
  };

  // --- 0. 预处理 inputs (设置 Encode=1, 插入 linalg.copy 处理 Constant) ---
  SmallVector<Value> processedInputs;
  for (Value in : inputs) {
    processedInputs.push_back(processNpuLinalgInput(rewriter, loc, in));
  }

  // --- 1. 执行输入端 Transpose ---
  Value actualA = processedInputs[0];
  if (transA) {
    actualA = buildLinalgTranspose(
        rewriter, loc, actualA, sourceOp, layerName, fusedOps);
  }
  Value actualB = processedInputs[1];
  if (transB) {
    actualB = buildLinalgTranspose(
        rewriter, loc, actualB, sourceOp, layerName, fusedOps);
  }

  // --- 2. 准备动态尺寸 ---
  SmallVector<Value> dynamicSizes =
      getDynamicSizes(rewriter, loc, processedInputs[0], outType.getShape());

  // --- 3. 调用核心构建器生成 Gemm ---
  Value biasVal = hasBias ? processedInputs[2] : nullptr;
  Value result =
      buildGemm(rewriter, loc, actualA, actualB, biasVal, dynamicSizes);

  return result;
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
                if (std::abs(alpha - 0.1) < 1e-5)
                  relu_type = 2;
                else if (std::abs(alpha - 0.2) < 1e-5)
                  relu_type = 3;
                else if (std::abs(alpha - 0.01) < 1e-5)
                  relu_type = 4;
                else {
                  llvm::errs() << "Warning: unsupported LeakyRelu alpha "
                               << alpha << ", skipping fusion.\n";
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
    SmallVector<StringRef> fusedOps = {"Gemm"};
    if (do_relu == 1)
      fusedOps.push_back("Relu");

    SmallVector<Value> inputs;
    inputs.push_back(quantInputA);
    inputs.push_back(quantInputB);

    bool hasBias = !mlir::isa<NoneType>(inputC.getType());
    if (hasBias) {
      if (auto dequantC = inputC.getDefiningOp<ONNXDequantizeLinearOp>()) {
        inputs.push_back(dequantC.getX());
      } else {
        inputs.push_back(inputC);
      }
    }

    Value result = createGenericMatMulOp(rewriter, op.getLoc(), inputs,
        outputType, paramsA.scale, paramsA.zeroPoint, paramsB.scale,
        paramsB.zeroPoint, paramsOut.scale, paramsOut.zeroPoint, "npu_gemm",
        do_relu, relu_type, op.getTransA(), op.getTransB(), op,
        getNpuProfileLayerName(op), fusedOps);

    // =========================================================================
    // 安全擦除
    // =========================================================================
    llvm::SmallPtrSet<Operation *, 4> opsErased;
    auto safeErase = [&](Operation *opToErase) {
      if (opToErase && opsErased.insert(opToErase).second) {
        rewriter.eraseOp(opToErase);
      }
    };

    // 顺延类型 encoding=1 向下传递
    finalQuantOp.getResult().setType(result.getType());

    rewriter.replaceOp(finalQuantOp, result);
    opsErased.insert(finalQuantOp);

    safeErase(op);

    for (auto *fuseOp : fusionOpsToErase) {
      safeErase(fuseOp);
    }

    if (dequantA->hasOneUse())
      safeErase(dequantA);
    if (dequantB->hasOneUse())
      safeErase(dequantB);

    if (hasBias) {
      if (auto dequantC = inputC.getDefiningOp<ONNXDequantizeLinearOp>()) {
        if (dequantC->hasOneUse())
          safeErase(dequantC);
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

    int64_t do_relu = 0;
    int64_t relu_type = 0;
    Operation *replaceTarget = op;
    SmallVector<Operation *> fusionOpsToErase;

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
                if (std::abs(alpha - 0.1) < 1e-5)
                  relu_type = 2;
                else if (std::abs(alpha - 0.2) < 1e-5)
                  relu_type = 3;
                else if (std::abs(alpha - 0.01) < 1e-5)
                  relu_type = 4;
                else {
                  llvm::errs() << "Warning: unsupported LeakyRelu alpha "
                               << alpha << ", skipping fusion.\n";
                  can_fuse = false;
                }
              } else {
                relu_type = 0; // relu
              }

              if (can_fuse) {
                do_relu = 1;
                replaceTarget = finalQOp;

                auto qParams = getScalarQuantParams(finalQOp);
                scaleY = qParams.scale;
                zpY = qParams.zeroPoint;
                outputType = mlir::cast<RankedTensorType>(
                    finalQOp.getResult().getType());

                fusionOpsToErase.push_back(dqOp);
                fusionOpsToErase.push_back(actOp);
              }
            }
          }
        }
      }
    }

    SmallVector<Value> inputs = {inputA, inputB};
    SmallVector<StringRef> fusedOps = {"QLinearMatMul"};
    if (do_relu == 1)
      fusedOps.push_back("Relu");

    bool isGraphOutput = llvm::any_of(replaceTarget->getUsers(),
        [](Operation *user) {
          return user->getName().getStringRef() == "func.return";
        });

    Value result =
        createGenericMatMulOp(rewriter, op.getLoc(), inputs, outputType, scaleA,
            zpA, scaleB, zpB, scaleY, zpY, "npu_matmul", do_relu, relu_type,
            false, false, op, getNpuProfileLayerName(op), fusedOps,
            /*skipSPM=*/false, /*isGraphOutput=*/isGraphOutput);

    // =========================================================================
    // 安全擦除
    // =========================================================================
    llvm::SmallPtrSet<Operation *, 4> opsErased;
    auto safeErase = [&](Operation *opToErase) {
      if (opToErase && opsErased.insert(opToErase).second) {
        rewriter.eraseOp(opToErase);
      }
    };

    // 顺延类型 encoding=1 向下传递
    replaceTarget->getResult(0).setType(result.getType());

    rewriter.replaceOp(replaceTarget, result);
    opsErased.insert(replaceTarget);

    safeErase(op);

    for (auto *fuseOp : fusionOpsToErase) {
      safeErase(fuseOp);
    }

    return success();
  }
};


struct MatMulIntegerToLinalg : public OpConversionPattern<ONNXMatMulIntegerOp> {
  using OpConversionPattern<ONNXMatMulIntegerOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXMatMulIntegerOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Value inputA = op.getA();
    Value inputB = op.getB();
    Value aZp = op.getAZeroPoint();
    Value bZp = op.getBZeroPoint();
    Location loc = op.getLoc();

    int64_t zpA = getScalarInt(aZp, 0);
    int64_t zpB = getScalarInt(bZp, 0);

    if (zpB != 0) {
      llvm::errs() << "Warning: MatMulInteger with non-zero weight zero-point "
                   << "will cause dynamic row offsets. Optimization assumes "
                      "symmetric weights.\n";
    }

    auto bType = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    if (!bType || bType.getRank() != 2) {
      return failure(); // 仅支持 2D 静态权重解析
    }
    int64_t K = bType.getShape()[0];
    int64_t N = bType.getShape()[1];

    // =========================================================
    // 1. 图融合探测：寻找紧跟其后的 onnx.Add
    // =========================================================
    ONNXAddOp fusedAddOp = nullptr;
    Value addConstVal = nullptr;

    if (op.getResult().hasOneUse()) {
      if (auto addOp =
              mlir::dyn_cast<ONNXAddOp>(*op.getResult().getUsers().begin())) {
        // 确定哪一边是 MatMulInteger 的输出，哪一边是 Bias 常量
        Value otherOperand =
            (addOp.getA() == op.getResult()) ? addOp.getB() : addOp.getA();
        if (otherOperand.getDefiningOp<ONNXConstantOp>()) {
          fusedAddOp = addOp;
          addConstVal = otherOperand;
        }
      }
    }

    SmallVector<StringRef> fusedOps = {"MatMulInteger"};
    SmallVector<int32_t> combinedBiasVals(N, 0);
    bool needBiasInput = false;

    // =========================================================
    // 2. 解析后续的 Add Bias 常量并并入 combinedBiasVals
    // =========================================================
    bool addFusedSuccessfully = false;

    if (fusedAddOp && addConstVal) {
      auto addConstOp = addConstVal.getDefiningOp<ONNXConstantOp>();
      if (auto addDense =
              mlir::dyn_cast<DenseElementsAttr>(addConstOp.getValueAttr())) {
        if (addDense.getElementType().isInteger(32)) {
          auto shape = cast<ShapedType>(addDense.getType()).getShape();
          
          // 支持 Scalar (空 shape) 或 [1] 的 broadcasting
          if (shape.empty() || (shape.size() == 1 && shape[0] == 1)) {
            int32_t scalarBias = addDense.getValues<int32_t>()[0];
            for (int n = 0; n < N; ++n)
              combinedBiasVals[n] = scalarBias;
            addFusedSuccessfully = true;
          } 
          // 支持 [N] 的 1D Tensor 或 [1, N] 的 2D Tensor
          else if ((shape.size() == 1 && shape[0] == N) || 
                   (shape.size() == 2 && shape[0] == 1 && shape[1] == N)) {
            auto values = addDense.getValues<int32_t>();
            int n = 0;
            for (auto val : values) {
              combinedBiasVals[n++] = val;
            }
            addFusedSuccessfully = true;
          }
        }
      }
    }

    // 根据解析结果决定是否提交融合状态
    if (fusedAddOp && addFusedSuccessfully) {
      needBiasInput = true;
      fusedOps.push_back("Add");
    } else {
      // 如果不满足支持的条件，则取消融合探测，保留原始 Add 节点
      fusedAddOp = nullptr;
      addConstVal = nullptr;
    }
    // =========================================================
    // 3. 计算由于激活值非对称量化引入的列偏移 Bias
    // =========================================================
    if (zpA != 0) {
      auto bConstOp = inputB.getDefiningOp<ONNXConstantOp>();
      if (bConstOp) {
        if (auto denseAttr =
                mlir::dyn_cast<DenseElementsAttr>(bConstOp.getValueAttr())) {
          if (denseAttr.getElementType().isInteger(8)) {
            auto values = denseAttr.getValues<int8_t>();
            auto it = values.begin();
            // sum_k (B_k,j) * (-Z_A)
            for (int k = 0; k < K; ++k) {
              for (int n = 0; n < N; ++n) {
                combinedBiasVals[n] += (128 - zpA) * static_cast<int32_t>(*it);
                ++it;
              }
            }
            needBiasInput = true;
          }
        }
      }
    }

    // =========================================================
    // 4. 生成最终的 3输入 / 2输入 参数列表
    // =========================================================
    SmallVector<Value> inputs = {inputA, inputB};
    if (needBiasInput) {
      auto biasType = RankedTensorType::get({N}, rewriter.getI32Type());
      auto newDenseAttr = DenseElementsAttr::get(
          biasType, llvm::ArrayRef<int32_t>(combinedBiasVals));
      Value precomputedBias =
          rewriter.create<ONNXConstantOp>(loc, mlir::Attribute(), newDenseAttr);

      inputs.push_back(precomputedBias);
    }

    // 确定最后被替换的目标算子和类型
    Operation *replaceTarget =
        fusedAddOp ? fusedAddOp.getOperation() : op.getOperation();
    auto outputType =
        mlir::cast<RankedTensorType>(replaceTarget->getResult(0).getType());

    // =========================================================
    // 5. 调用核心接口生成纯粹的 INT32 GEMM（跳过 SPM）
    // =========================================================
    bool isGraphOutput = false;
    for (Operation *user : replaceTarget->getUsers()) {
      if (user->getName().getStringRef() == "func.return") {
        isGraphOutput = true;
        break;
      }
    }

    // 5. 调用核心接口，传入 isGraphOutput
    Value result = createGenericMatMulOp(rewriter, loc, inputs, outputType,
        1.0f, zpA, 1.0f, zpB, 1.0f, 0, "npu_matmul_integer", 0, 0, false, false,
        op, getNpuProfileLayerName(op), fusedOps,
        /*skipSPM=*/true,
        /*isGraphOutput=*/isGraphOutput);

    // =========================================================
    // 6. 安全擦除与图更新
    // =========================================================
    replaceTarget->getResult(0).setType(result.getType());
    rewriter.replaceOp(replaceTarget, result);

    if (fusedAddOp) {
      rewriter.eraseOp(op); // 擦除原 MatMulInteger
      if (addConstVal.hasOneUse()) {
        rewriter.eraseOp(addConstVal.getDefiningOp());
      }
    }

    return success();
  }
};

} // namespace

void npux::populateLinalgGemmPattern(RewritePatternSet &patterns) {
  patterns.add<GemmToLinalg, QLinearMatMulToLinalg, MatMulIntegerToLinalg>(
      patterns.getContext());
}
