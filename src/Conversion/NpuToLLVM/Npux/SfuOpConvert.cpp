//=======================================
//src/Conversion/NpuToLLVM/SfuOpConvert.cpp
//this file implements convert linalg sfu op
// to custom npux sfu run op
//=======================================

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "src/Dialect/Npux/NpuxOps.hpp"
#include "src/Conversion/NpuToLLVM/NpuxConversionHelper.hpp"

#include <algorithm>
#include <cmath> 

using namespace mlir;
using namespace npux;

namespace {

// ============================================================================
// Helper: Quant Params (Existing)
// ============================================================================
struct FixedPointParams {
  int16_t multiplier;
  int16_t shift;
};

FixedPointParams getFixedPointParams(double scale) {
  if (std::abs(scale) < 1e-8) return {0, 0};

  int exponent;
  double mantissa = std::frexp(scale, &exponent); 

  double mantissa_scaled = std::round(mantissa * 32768.0);

  if (mantissa_scaled >= 32768.0) {
    mantissa_scaled /= 2.0;
    exponent += 1;
  }

  return {
    static_cast<int16_t>(mantissa_scaled),
    static_cast<int16_t>(exponent - 15)
  };
}

static bool hasOnlyStaticPositiveShape(MemRefType type) {
  for (int64_t dim : type.getShape()) {
    if (dim == ShapedType::kDynamic || dim <= 0)
      return false;
  }
  return true;
}

static Value createIndexConstant(
    PatternRewriter &rewriter, Location loc, int64_t value) {
  return rewriter.create<arith::ConstantIndexOp>(loc, value);
}

static Value createI16Constant(
    PatternRewriter &rewriter, Location loc, int64_t value) {
  return rewriter.create<arith::ConstantIntOp>(loc, value, 16);
}

static SmallVector<int64_t> getPermutation(linalg::GenericOp op) {
  SmallVector<int64_t> perm;
  if (op.getIndexingMapsArray().size() < 2)
    return perm;

  AffineMap outputMap = op.getIndexingMapsArray()[1];
  if (!outputMap.isPermutation())
    return {};

  perm.reserve(outputMap.getNumResults());
  for (AffineExpr expr : outputMap.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return {};
    perm.push_back(dimExpr.getPosition());
  }
  return perm;
}

static bool matchesPermutation(
    ArrayRef<int64_t> perm, ArrayRef<int64_t> expected) {
  return perm.size() == expected.size() &&
         std::equal(perm.begin(), perm.end(), expected.begin());
}

static bool matchesRank2TransposeShape(
    ArrayRef<int64_t> inShape, ArrayRef<int64_t> outShape) {
  return inShape.size() == 2 && outShape.size() == 2 &&
         outShape[0] == inShape[1] && outShape[1] == inShape[0];
}

static bool matchesRank3BatchLast2TransposeShape(
    ArrayRef<int64_t> inShape, ArrayRef<int64_t> outShape) {
  return inShape.size() == 3 && outShape.size() == 3 &&
         outShape[0] == inShape[0] &&
         outShape[1] == inShape[2] &&
         outShape[2] == inShape[1];
}

static bool matchesRank4Perm0231Shape(
    ArrayRef<int64_t> inShape, ArrayRef<int64_t> outShape) {
  return inShape.size() == 4 && outShape.size() == 4 &&
         outShape[0] == inShape[0] &&
         outShape[1] == inShape[2] &&
         outShape[2] == inShape[3] &&
         outShape[3] == inShape[1];
}

static bool matchesRank4Perm0213Shape(
    ArrayRef<int64_t> inShape, ArrayRef<int64_t> outShape) {
  return inShape.size() == 4 && outShape.size() == 4 &&
         outShape[0] == inShape[0] &&
         outShape[1] == inShape[2] &&
         outShape[2] == inShape[1] &&
         outShape[3] == inShape[3];
}

static Value createNpuSubview(
    PatternRewriter &rewriter, Location loc, Value source,
    ArrayRef<Value> offsets, ArrayRef<int64_t> resultShape) {
  auto sourceType = cast<MemRefType>(source.getType());
  MemRefLayoutAttrInterface layout;
  auto resultType = MemRefType::get(
      resultShape, sourceType.getElementType(), layout,
      sourceType.getMemorySpace());
  return rewriter
      .create<npux::SubviewOp>(loc, resultType, source, offsets)
      .getResult();
}

static Value createStaticSramAlloc(
    PatternRewriter &rewriter, Location loc, ArrayRef<int64_t> shape,
    Type elemType) {
  MemRefLayoutAttrInterface layout;
  auto type = MemRefType::get(shape, elemType, layout,
      IntegerAttr::get(IntegerType::get(rewriter.getContext(), 64), 2));
  return rewriter.create<npux::SramAllocOp>(loc, type, ValueRange{}).getResult();
}

static void emitTransposeRun(PatternRewriter &rewriter, Location loc,
    Value inputMemRef, Value outputMemRef, int64_t colsM1, int64_t rowsM1) {
  rewriter.create<TransposeOp>(loc, inputMemRef, outputMemRef,
      createI16Constant(rewriter, loc, colsM1),
      createI16Constant(rewriter, loc, rowsM1));
}

// ============================================================================
// Pattern 1: SFU Ops (Gelu, Softmax, LayerNorm) -> SfuRunOp
// ============================================================================
class LinalgSfuToNpuxPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr) return failure();
    StringRef opName = libCallAttr.getValue();

    // 仅匹配 SFU 相关算子
    npux::SFUOpType sfuOpEnum;
    if (opName == "npu_gelu") sfuOpEnum = npux::SFUOpType::gelu;
    else if (opName == "npu_softmax") sfuOpEnum = npux::SFUOpType::softmax;
    else if (opName == "npu_layernorm") sfuOpEnum = npux::SFUOpType::layernorm;
    else return failure(); 

    Location loc = op.getLoc();

    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) return failure();
    
    Value inputMemRef = op.getInputs()[0];
    Value outputMemRef = op.getOutputs()[0];

    auto inType = mlir::dyn_cast<MemRefType>(inputMemRef.getType());
    if (!inType || inType.getMemorySpaceAsInt() != 2) return failure();

    ArrayRef<int64_t> shape = inType.getShape();
    int rank = shape.size();
    if (rank < 2) return failure();
    
    // SFU 硬件接口需要的是 col_num - 1 和 row_num - 1
    int64_t rows = shape[rank - 2] - 1;
    int64_t cols = shape[rank - 1] - 1;

    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows, 16);

    // ... (Quant Params Calculation 保持不变) ...
    // 1. Input Quant
    double inScale = 1.0;
    if (auto attr = op->getAttrOfType<FloatAttr>("in_scale")) inScale = attr.getValueAsDouble();
    auto inFP = getFixedPointParams(inScale);
    
    Value vInScale = rewriter.create<arith::ConstantIntOp>(loc, inFP.multiplier, 16);
    Value vInShift = rewriter.create<arith::ConstantIntOp>(loc, inFP.shift, 16);
    
    int64_t inZp = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("in_zp")) inZp = attr.getInt();
    Value vInZp = rewriter.create<arith::ConstantIntOp>(loc, inZp, 32); 

    // 2. Output Quant
    double outScale = 1.0;
    if (auto attr = op->getAttrOfType<FloatAttr>("out_scale")) outScale = attr.getValueAsDouble();
    // 硬件要求 out_scale 是倒数形式 (inv_scale)
    double invOutScale = (std::abs(outScale) > 1e-9) ? (1.0 / outScale) : 1.0;
    auto outFP = getFixedPointParams(invOutScale);

    Value vOutScale = rewriter.create<arith::ConstantIntOp>(loc, outFP.multiplier, 16);
    Value vOutShift = rewriter.create<arith::ConstantIntOp>(loc, outFP.shift, 16);

    int64_t outZp = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("out_zp")) outZp = attr.getInt();
    Value vOutZp = rewriter.create<arith::ConstantIntOp>(loc, outZp, 16); 

    // F. 其他配置
    int8_t intTypeVal = 0; 
    if (inType.getElementType().isInteger(16)) intTypeVal = 1;
    else if (inType.getElementType().isInteger(32)) intTypeVal = 2;
    Value vIntType = rewriter.create<arith::ConstantIntOp>(loc, intTypeVal, 8);

    bool isQuant = op->hasAttr("in_zp");
    Value vIsQuant = rewriter.create<arith::ConstantIntOp>(loc, isQuant, 1);

    auto opTypeAttr = SFUOpTypeAttr::get(rewriter.getContext(), sfuOpEnum);

    rewriter.replaceOpWithNewOp<SfuRunOp>(op,
        opTypeAttr,
        vIntType,
        vIsQuant,
        inputMemRef, 
        vCol, 
        vRow,
        outputMemRef, 
        vInZp,
        vOutZp,
        vInScale,
        vInShift,
        vOutScale,
        vOutShift
    );

    return success();
  }
};

// ============================================================================
// Pattern 2: Transpose Op -> TransposeOp
// ============================================================================
class LinalgTransposeToNpuxPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr || libCallAttr.getValue() != "npu_transpose") return failure();

    Location loc = op.getLoc();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) return failure();

    Value inputMemRef = op.getInputs()[0];
    Value outputMemRef = op.getOutputs()[0];

    // 检查 Memory Space (SRAM=2)
    auto inType = mlir::dyn_cast<MemRefType>(inputMemRef.getType());
    auto outType = mlir::dyn_cast<MemRefType>(outputMemRef.getType());
    if (!inType || !outType || inType.getMemorySpaceAsInt() != 2 ||
        outType.getMemorySpaceAsInt() != 2)
      return failure();
    if (!hasOnlyStaticPositiveShape(inType) || !hasOnlyStaticPositiveShape(outType))
      return failure();

    SmallVector<int64_t> perm = getPermutation(op);
    if (perm.empty())
      return failure();

    ArrayRef<int64_t> inShape = inType.getShape();
    ArrayRef<int64_t> outShape = outType.getShape();
    int64_t rank = static_cast<int64_t>(inShape.size());

    Value c0 = createIndexConstant(rewriter, loc, 0);
    Value c1 = createIndexConstant(rewriter, loc, 1);

    auto eraseOriginalOp = [&]() -> LogicalResult {
      rewriter.eraseOp(op);
      return success();
    };

    auto emitRank2Transpose = [&]() -> LogicalResult {
      static constexpr int64_t kPerm2D[] = {1, 0};
      if (rank != 2 ||
          !(matchesPermutation(perm, kPerm2D) ||
            matchesRank2TransposeShape(inShape, outShape)))
        return failure();
      emitTransposeRun(rewriter, loc, inputMemRef, outputMemRef,
          inShape[1] - 1, inShape[0] - 1);
      return eraseOriginalOp();
    };

    auto emitRank3BatchLast2Transpose = [&]() -> LogicalResult {
      static constexpr int64_t kPerm3D[] = {0, 2, 1};
      if (rank != 3 ||
          !(matchesPermutation(perm, kPerm3D) ||
            matchesRank3BatchLast2TransposeShape(inShape, outShape)))
        return failure();

      Value batchUpper = createIndexConstant(rewriter, loc, inShape[0]);
      auto batchLoop = rewriter.create<scf::ForOp>(loc, c0, batchUpper, c1);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(batchLoop.getBody()->getTerminator());
        Value batch = batchLoop.getInductionVar();

        SmallVector<Value> inOffsets = {batch, c0, c0};
        SmallVector<Value> outOffsets = {batch, c0, c0};

        Value inSubview = createNpuSubview(
            rewriter, loc, inputMemRef, inOffsets, {inShape[1], inShape[2]});
        Value outSubview = createNpuSubview(
            rewriter, loc, outputMemRef, outOffsets, {inShape[2], inShape[1]});

        emitTransposeRun(rewriter, loc, inSubview, outSubview,
            inShape[2] - 1, inShape[1] - 1);
      }
      return eraseOriginalOp();
    };

    auto emitRank4Perm0231 = [&]() -> LogicalResult {
      static constexpr int64_t kPerm0231[] = {0, 2, 3, 1};
      if (rank != 4 ||
          !(matchesPermutation(perm, kPerm0231) ||
            matchesRank4Perm0231Shape(inShape, outShape)))
        return failure();

      const int64_t batchSize = inShape[0];
      const int64_t dimA = inShape[1];
      const int64_t dimB = inShape[2];
      const int64_t dimC = inShape[3];

      Value batchUpper = createIndexConstant(rewriter, loc, batchSize);
      auto batchLoop = rewriter.create<scf::ForOp>(loc, c0, batchUpper, c1);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(batchLoop.getBody()->getTerminator());
        Value batch = batchLoop.getInductionVar();

        SmallVector<Value> inOffsets = {batch, c0, c0, c0};
        SmallVector<Value> outOffsets = {batch, c0, c0, c0};

        Value inSubview = createNpuSubview(
            rewriter, loc, inputMemRef, inOffsets, {dimA, dimB, dimC});
        Value outSubview = createNpuSubview(
            rewriter, loc, outputMemRef, outOffsets, {dimB, dimC, dimA});

        emitTransposeRun(rewriter, loc, inSubview, outSubview,
            dimB * dimC - 1, dimA - 1);
      }
      return eraseOriginalOp();
    };

    auto emitRank4Perm0213 = [&]() -> LogicalResult {
      static constexpr int64_t kPerm0213[] = {0, 2, 1, 3};
      if (rank != 4 ||
          !(matchesPermutation(perm, kPerm0213) ||
            matchesRank4Perm0213Shape(inShape, outShape)))
        return failure();

      const int64_t batchSize = inShape[0];
      const int64_t dimA = inShape[1];
      const int64_t dimB = inShape[2];
      const int64_t dimC = inShape[3];

      Value tempBuffer = createStaticSramAlloc(
          rewriter, loc, {batchSize, dimB, dimC, dimA}, inType.getElementType());
      Value batchUpper = createIndexConstant(rewriter, loc, batchSize);
      Value headUpper = createIndexConstant(rewriter, loc, dimB);

      auto stage1Loop = rewriter.create<scf::ForOp>(loc, c0, batchUpper, c1);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(stage1Loop.getBody()->getTerminator());
        Value batch = stage1Loop.getInductionVar();

        SmallVector<Value> inOffsets = {batch, c0, c0, c0};
        SmallVector<Value> tempOffsets = {batch, c0, c0, c0};

        Value inSubview = createNpuSubview(
            rewriter, loc, inputMemRef, inOffsets, {dimA, dimB, dimC});
        Value tempSubview = createNpuSubview(
            rewriter, loc, tempBuffer, tempOffsets, {dimB, dimC, dimA});

        emitTransposeRun(rewriter, loc, inSubview, tempSubview,
            dimB * dimC - 1, dimA - 1);
      }

      auto stage2BatchLoop = rewriter.create<scf::ForOp>(loc, c0, batchUpper, c1);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(stage2BatchLoop.getBody()->getTerminator());
        Value batch = stage2BatchLoop.getInductionVar();

        auto headLoop = rewriter.create<scf::ForOp>(loc, c0, headUpper, c1);
        {
          OpBuilder::InsertionGuard headGuard(rewriter);
          rewriter.setInsertionPoint(headLoop.getBody()->getTerminator());
          Value head = headLoop.getInductionVar();

          SmallVector<Value> tempOffsets = {batch, head, c0, c0};
          SmallVector<Value> outOffsets = {batch, head, c0, c0};

          Value tempSubview = createNpuSubview(
              rewriter, loc, tempBuffer, tempOffsets, {dimC, dimA});
          Value outSubview = createNpuSubview(
              rewriter, loc, outputMemRef, outOffsets, {dimA, dimC});

          emitTransposeRun(rewriter, loc, tempSubview, outSubview,
              dimA - 1, dimC - 1);
        }
      }

      rewriter.create<npux::SramFreeOp>(loc, tempBuffer);
      return eraseOriginalOp();
    };

    if (succeeded(emitRank2Transpose()))
      return success();
    if (succeeded(emitRank3BatchLast2Transpose()))
      return success();
    if (succeeded(emitRank4Perm0231()))
      return success();
    if (succeeded(emitRank4Perm0213()))
      return success();

    op.emitError()
        << "unsupported npu_transpose permutation for NPU lowering; rank="
        << rank;
    return failure();
  }
};

// ============================================================================
// Pattern 3: Resample Ops (MaxPool, AvgPool, Upsample) -> ResampleOp
// ============================================================================
class LinalgResampleToNpuxPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr) return failure();
    StringRef opName = libCallAttr.getValue();

    // 确定 Resample Type 和 Op
    npux::ResampleType typeEnum;
    npux::ResampleMode modeEnum;

    if (opName == "npu_maxpool") {
        typeEnum = npux::ResampleType::downsample;
        modeEnum = npux::ResampleMode::max_nearest;
    } else if (opName == "npu_avgpool") {
      op.emitWarning() << "NPU backend does not support AveragePool currently.";
      return failure();
    } else if (opName == "npu_upsample") {
        typeEnum = npux::ResampleType::upsample;
        modeEnum = npux::ResampleMode::max_nearest; // Nearest Neighbor
    } else {
        return failure();
    }

    Location loc = op.getLoc();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) return failure();

    Value inputMemRef = op.getInputs()[0];
    Value outputMemRef = op.getOutputs()[0];

    // 检查 Memory Space (SRAM=2)
    auto inType = mlir::dyn_cast<MemRefType>(inputMemRef.getType());
    if (!inType || inType.getMemorySpaceAsInt() != 2) return failure();

    // 计算 Shape: API 只支持 2D 输入，且 resample 仅允许 NCHW(4D) 输入。
    // 规则: [N, C, H, W] -> [N*C*H, W]
    // 传给 CAPI 的是 (col_num, row_num) = (mergedW-1, mergedH-1)。
    ArrayRef<int64_t> shape = inType.getShape();
    int rank = shape.size();
    if (rank < 2) return failure();

    if (rank == 5) {
      op.emitWarning()
          << "Resample forbids NCHWc32 (5D) input. Please keep resample in NCHW (4D) format.";
      return failure();
    }
    if (rank != 4) {
      op.emitWarning() << "Resample expects NCHW (4D) input, but got rank=" << rank;
      return failure();
    }

    for (int64_t dim : shape) {
      if (dim == ShapedType::kDynamic || dim <= 0) return failure();
    }

    int64_t mergedRows = shape[0] * shape[1] * shape[2];
    int64_t mergedCols = shape[3];

    int64_t rows = mergedRows - 1;
    int64_t cols = mergedCols - 1;

    Value vInputCol = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vInputRow = rewriter.create<arith::ConstantIntOp>(loc, rows, 16);

    // 构建 Enum 属性
    auto typeAttr = ResampleTypeAttr::get(rewriter.getContext(), typeEnum);
    auto modeAttr = ResampleModeAttr::get(rewriter.getContext(), modeEnum);

    // 创建 ResampleOp
    rewriter.replaceOpWithNewOp<npux::ResampleOp>(op,
        typeAttr,
        modeAttr,
        inputMemRef,
        outputMemRef,
        vInputCol,
        vInputRow
    );

    return success();
  }
};

}// namespace

class LinalgLayoutToNpuxPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr) return failure();
    StringRef opName = libCallAttr.getValue();

    bool isPack = (opName == "npu_layout_nchw_to_nchwc32");
    bool isUnpack = (opName == "npu_layout_nchwc32_to_nchw");

    if (!isPack && !isUnpack) return failure();

    Location loc = op.getLoc();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) return failure();

    // 此时已经是 Bufferization 之后，操作数应为 MemRef
    Value inputMemRef = op.getInputs()[0];
    Value outputMemRef = op.getOutputs()[0];

    // 检查 Memory Space (SRAM=2)
    // 确保数据已经在 SRAM 中，符合 npu_layout_* 的要求
    auto inType = mlir::dyn_cast<MemRefType>(inputMemRef.getType());
    auto outType = mlir::dyn_cast<MemRefType>(outputMemRef.getType());

    auto inshape = inType.getShape();
    auto outshape = outType.getShape();

    // 从属性中提取 N, C, H, W 参数
    // 这些属性是在 Conv.cpp 中 setAttr 的
    auto getIntParam = [&](StringRef name) -> int64_t {
        if (auto attr = op->getAttrOfType<IntegerAttr>(name)) {
            return attr.getInt();
        }
        // 如果没有找到参数，这是一个错误，但在 Pattern 中通常返回 failure
        // 为了安全起见这里返回 1
        return 1; 
    };
    int64_t n, c, h, w;
    if (isPack){
      n = inshape[0];
      c = inshape[1];
      h = inshape[2];
      w = inshape[3];
    } else if (isUnpack){
      n = outshape[0];
      c = outshape[1];
      h = outshape[2];
      w = outshape[3];
    } else {
      return failure();
    }

    // 创建 i16 常量作为参数传递给 Op
    Value vN = rewriter.create<arith::ConstantIntOp>(loc, n, 16);
    Value vC = rewriter.create<arith::ConstantIntOp>(loc, c, 16);
    Value vH = rewriter.create<arith::ConstantIntOp>(loc, h, 16);
    Value vW = rewriter.create<arith::ConstantIntOp>(loc, w, 16);

    // 替换为具体的 Npux Op
    if (isPack) {
        rewriter.replaceOpWithNewOp<npux::LayoutNchwToNchwc32Op>(op,
            inputMemRef, outputMemRef, vN, vC, vH, vW);
    } else {
        rewriter.replaceOpWithNewOp<npux::LayoutNchwc32ToNchwOp>(op,
            inputMemRef, outputMemRef, vN, vC, vH, vW);
    }

    return success();
  }
};

// ============================================================================
// Registration
// ============================================================================
void npux::populateLinalgSfuToNpuxPattern(RewritePatternSet &patterns) {
  patterns.add<LinalgSfuToNpuxPattern>(patterns.getContext());
  patterns.add<LinalgTransposeToNpuxPattern>(patterns.getContext());
  patterns.add<LinalgResampleToNpuxPattern>(patterns.getContext());
  patterns.add<LinalgLayoutToNpuxPattern>(patterns.getContext());
}
