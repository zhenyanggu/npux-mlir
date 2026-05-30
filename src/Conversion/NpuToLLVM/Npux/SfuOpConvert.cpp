//=======================================
//src/Conversion/NpuToLLVM/SfuOpConvert.cpp
//this file implements lowering from npucore SFU/layout ops
// to custom npux runtime ops
//=======================================

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include "src/Dialect/Npux/NpuxOps.hpp"
#include "src/Conversion/NpuToLLVM/NpuxConversionHelper.hpp"

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

static LogicalResult lowerLayoutOpToNpux(Location loc, Value inputMemRef,
    Value outputMemRef, bool isPack, PatternRewriter &rewriter,
    Operation *sourceOp) {
  auto inType = dyn_cast<MemRefType>(inputMemRef.getType());
  auto outType = dyn_cast<MemRefType>(outputMemRef.getType());
  if (!inType || !outType)
    return failure();

  ArrayRef<int64_t> inshape = inType.getShape();
  ArrayRef<int64_t> outshape = outType.getShape();

  int64_t n = 0;
  int64_t c = 0;
  int64_t h = 0;
  int64_t w = 0;
  if (isPack) {
    if (inshape.size() != 4)
      return failure();
    n = inshape[0];
    c = inshape[1];
    h = inshape[2];
    w = inshape[3];
  } else {
    if (outshape.size() != 4)
      return failure();
    n = outshape[0];
    c = outshape[1];
    h = outshape[2];
    w = outshape[3];
  }

  if (n == ShapedType::kDynamic || c == ShapedType::kDynamic ||
      h == ShapedType::kDynamic || w == ShapedType::kDynamic) {
    sourceOp->emitOpError("requires static N/C/H/W for layout lowering");
    return failure();
  }

  Value vN = rewriter.create<arith::ConstantIntOp>(loc, n, 16);
  Value vC = rewriter.create<arith::ConstantIntOp>(loc, c, 16);
  Value vH = rewriter.create<arith::ConstantIntOp>(loc, h, 16);
  Value vW = rewriter.create<arith::ConstantIntOp>(loc, w, 16);

  if (isPack) {
    rewriter.replaceOpWithNewOp<npux::LayoutNchwToNchwc32Op>(
        sourceOp, inputMemRef, outputMemRef, vN, vC, vH, vW);
  } else {
    rewriter.replaceOpWithNewOp<npux::LayoutNchwc32ToNchwOp>(
        sourceOp, inputMemRef, outputMemRef, vN, vC, vH, vW);
  }
  return success();
}

// ============================================================================
// Helper: lower npucore.maxpool(memref) to npux.resample.
// ============================================================================
static LogicalResult lowerNpucoreMaxPoolToResample(
    npucore::MaxPoolOp op, PatternRewriter &rewriter) {
  if (op.hasTensorSemantics())
    return failure();
  if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
    return failure();

  auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
  auto outputType = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
  if (!inputType || !outputType)
    return failure();
  if (inputType.getMemorySpaceAsInt() != 2 || outputType.getMemorySpaceAsInt() != 2)
    return failure();
  if (inputType.getRank() != 4 || outputType.getRank() != 4) {
    return rewriter.notifyMatchFailure(
        op, "npucore.maxpool lowering expects rank-4 NCHW memrefs");
  }

  int64_t kernelH = cast<IntegerAttr>(op.getKernelShape()[0]).getInt();
  int64_t kernelW = cast<IntegerAttr>(op.getKernelShape()[1]).getInt();
  int64_t strideH = cast<IntegerAttr>(op.getStrides()[0]).getInt();
  int64_t strideW = cast<IntegerAttr>(op.getStrides()[1]).getInt();
  int64_t dilationH = cast<IntegerAttr>(op.getDilations()[0]).getInt();
  int64_t dilationW = cast<IntegerAttr>(op.getDilations()[1]).getInt();
  int64_t padTop = cast<IntegerAttr>(op.getPads()[0]).getInt();
  int64_t padLeft = cast<IntegerAttr>(op.getPads()[1]).getInt();
  int64_t padBottom = cast<IntegerAttr>(op.getPads()[2]).getInt();
  int64_t padRight = cast<IntegerAttr>(op.getPads()[3]).getInt();

  if (kernelH != 2 || kernelW != 2 || strideH != 2 || strideW != 2 ||
      dilationH != 1 || dilationW != 1 || padTop != 0 || padLeft != 0 ||
      padBottom != 0 || padRight != 0) {
    return rewriter.notifyMatchFailure(op,
        "current npux.resample path only supports kernel=2x2 stride=2x2 "
        "dilation=1x1 pads=0");
  }

  ArrayRef<int64_t> shape = inputType.getShape();
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim <= 0) {
      return rewriter.notifyMatchFailure(
          op, "npucore.maxpool lowering requires static positive NCHW shape");
    }
  }

  int64_t mergedRows = shape[0] * shape[1] * shape[2];
  int64_t mergedCols = shape[3];
  Value vInputCol =
      rewriter.create<arith::ConstantIntOp>(op.getLoc(), mergedCols - 1, 16);
  Value vInputRow =
      rewriter.create<arith::ConstantIntOp>(op.getLoc(), mergedRows - 1, 16);

  auto typeAttr =
      ResampleTypeAttr::get(rewriter.getContext(), npux::ResampleType::downsample);
  auto modeAttr = ResampleModeAttr::get(
      rewriter.getContext(), npux::ResampleMode::max_nearest);

  rewriter.replaceOpWithNewOp<npux::ResampleOp>(op, typeAttr, modeAttr,
      op.getInputs()[0], op.getOutputs()[0], vInputCol, vInputRow);
  return success();
}

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

class NpucoreLayoutNchwToNchwc32ToNpuxPattern
    : public OpRewritePattern<npucore::LayoutNchwToNchwc32Op> {
public:
  using OpRewritePattern<npucore::LayoutNchwToNchwc32Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(npucore::LayoutNchwToNchwc32Op op,
      PatternRewriter &rewriter) const override {
    if (op.hasTensorSemantics())
      return failure();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();
    return lowerLayoutOpToNpux(
        op.getLoc(), op.getInputs()[0], op.getOutputs()[0], true, rewriter, op);
  }
};

class NpucoreLayoutNchwc32ToNchwToNpuxPattern
    : public OpRewritePattern<npucore::LayoutNchwc32ToNchwOp> {
public:
  using OpRewritePattern<npucore::LayoutNchwc32ToNchwOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(npucore::LayoutNchwc32ToNchwOp op,
      PatternRewriter &rewriter) const override {
    if (op.hasTensorSemantics())
      return failure();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();
    return lowerLayoutOpToNpux(op.getLoc(), op.getInputs()[0],
        op.getOutputs()[0], false, rewriter, op);
  }
};

class NpucoreTransposeToNpuxPattern
    : public OpRewritePattern<npucore::TransposeOp> {
public:
  using OpRewritePattern<npucore::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::TransposeOp op, PatternRewriter &rewriter) const override {
    if (op.hasTensorSemantics())
      return failure();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    Location loc = op.getLoc();
    Value inputMemRef = op.getInputs()[0];
    Value outputMemRef = op.getOutputs()[0];

    auto inType = dyn_cast<MemRefType>(inputMemRef.getType());
    if (!inType || inType.getMemorySpaceAsInt() != 2)
      return failure();

    ArrayRef<int64_t> shape = inType.getShape();
    int rank = shape.size();
    if (rank < 2)
      return failure();

    int64_t rows = shape[rank - 2] - 1;
    int64_t cols = shape[rank - 1] - 1;

    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows, 16);

    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, inputMemRef, outputMemRef, vCol, vRow);
    return success();
  }
};

class NpucoreMaxPoolToNpuxPattern
    : public OpRewritePattern<npucore::MaxPoolOp> {
public:
  using OpRewritePattern<npucore::MaxPoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::MaxPoolOp op, PatternRewriter &rewriter) const override {
    return lowerNpucoreMaxPoolToResample(op, rewriter);
  }
};

// ============================================================================
// Pattern 4: npucore.gelu(memref) -> SfuRunOp
// ============================================================================
class NpucoreGeluToNpuxPattern : public OpRewritePattern<npucore::GeluOp> {
public:
  using OpRewritePattern<npucore::GeluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::GeluOp op, PatternRewriter &rewriter) const override {
    if (op.hasTensorSemantics())
      return failure();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    Value inputMemRef = op.getInputs()[0];
    Value outputMemRef = op.getOutputs()[0];
    auto inType = mlir::dyn_cast<MemRefType>(inputMemRef.getType());
    if (!inType || inType.getMemorySpaceAsInt() != 2)
      return failure();

    ArrayRef<int64_t> shape = inType.getShape();
    int rank = shape.size();
    if (rank < 2)
      return failure();

    Location loc = op.getLoc();
    int64_t rows = shape[rank - 2] - 1;
    int64_t cols = shape[rank - 1] - 1;
    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows, 16);

    auto inFP = getFixedPointParams(op.getInScale().convertToDouble());
    double outScale = op.getOutScale().convertToDouble();
    auto outFP =
        getFixedPointParams((std::abs(outScale) > 1e-9) ? (1.0 / outScale) : 1.0);

    Value vInScale =
        rewriter.create<arith::ConstantIntOp>(loc, inFP.multiplier, 16);
    Value vInShift =
        rewriter.create<arith::ConstantIntOp>(loc, inFP.shift, 16);
    Value vInZp =
        rewriter.create<arith::ConstantIntOp>(loc, op.getInZp(), 32);
    Value vOutScale =
        rewriter.create<arith::ConstantIntOp>(loc, outFP.multiplier, 16);
    Value vOutShift =
        rewriter.create<arith::ConstantIntOp>(loc, outFP.shift, 16);
    Value vOutZp =
        rewriter.create<arith::ConstantIntOp>(loc, op.getOutZp(), 16);

    int8_t intTypeVal = 0;
    if (inType.getElementType().isInteger(16))
      intTypeVal = 1;
    else if (inType.getElementType().isInteger(32))
      intTypeVal = 2;
    Value vIntType = rewriter.create<arith::ConstantIntOp>(loc, intTypeVal, 8);
    Value vIsQuant = rewriter.create<arith::ConstantIntOp>(loc, true, 1);
    auto opTypeAttr =
        SFUOpTypeAttr::get(rewriter.getContext(), npux::SFUOpType::gelu);

    rewriter.replaceOpWithNewOp<SfuRunOp>(op, opTypeAttr, vIntType, vIsQuant,
        inputMemRef, vCol, vRow, outputMemRef, vInZp, vOutZp, vInScale,
        vInShift, vOutScale, vOutShift);
    return success();
  }
};

// ============================================================================
// Pattern 5: npucore.softmax(memref) -> SfuRunOp
// ============================================================================
class NpucoreSoftmaxToNpuxPattern
    : public OpRewritePattern<npucore::SoftmaxOp> {
public:
  using OpRewritePattern<npucore::SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::SoftmaxOp op, PatternRewriter &rewriter) const override {
    if (op.hasTensorSemantics())
      return failure();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    Value inputMemRef = op.getInputs()[0];
    Value outputMemRef = op.getOutputs()[0];
    auto inType = mlir::dyn_cast<MemRefType>(inputMemRef.getType());
    if (!inType || inType.getMemorySpaceAsInt() != 2)
      return failure();

    ArrayRef<int64_t> shape = inType.getShape();
    int rank = shape.size();
    if (rank < 2)
      return failure();

    Location loc = op.getLoc();
    int64_t rows = shape[rank - 2] - 1;
    int64_t cols = shape[rank - 1] - 1;
    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows, 16);

    auto inFP = getFixedPointParams(op.getInScale().convertToDouble());
    double outScale = op.getOutScale().convertToDouble();
    auto outFP =
        getFixedPointParams((std::abs(outScale) > 1e-9) ? (1.0 / outScale) : 1.0);

    Value vInScale =
        rewriter.create<arith::ConstantIntOp>(loc, inFP.multiplier, 16);
    Value vInShift =
        rewriter.create<arith::ConstantIntOp>(loc, inFP.shift, 16);
    Value vInZp =
        rewriter.create<arith::ConstantIntOp>(loc, op.getInZp(), 32);
    Value vOutScale =
        rewriter.create<arith::ConstantIntOp>(loc, outFP.multiplier, 16);
    Value vOutShift =
        rewriter.create<arith::ConstantIntOp>(loc, outFP.shift, 16);
    Value vOutZp =
        rewriter.create<arith::ConstantIntOp>(loc, op.getOutZp(), 16);

    int8_t intTypeVal = 0;
    if (inType.getElementType().isInteger(16))
      intTypeVal = 1;
    else if (inType.getElementType().isInteger(32))
      intTypeVal = 2;
    Value vIntType = rewriter.create<arith::ConstantIntOp>(loc, intTypeVal, 8);
    Value vIsQuant = rewriter.create<arith::ConstantIntOp>(loc, true, 1);
    auto opTypeAttr =
        SFUOpTypeAttr::get(rewriter.getContext(), npux::SFUOpType::softmax);

    rewriter.replaceOpWithNewOp<SfuRunOp>(op, opTypeAttr, vIntType, vIsQuant,
        inputMemRef, vCol, vRow, outputMemRef, vInZp, vOutZp, vInScale,
        vInShift, vOutScale, vOutShift);
    return success();
  }
};

// ============================================================================
// Pattern 6: npucore.layernorm(memref) -> SfuRunOp
// ============================================================================
class NpucoreLayerNormToNpuxPattern
    : public OpRewritePattern<npucore::LayerNormOp> {
public:
  using OpRewritePattern<npucore::LayerNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::LayerNormOp op, PatternRewriter &rewriter) const override {
    if (op.hasTensorSemantics())
      return failure();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    Value inputMemRef = op.getInputs()[0];
    Value outputMemRef = op.getOutputs()[0];
    auto inType = mlir::dyn_cast<MemRefType>(inputMemRef.getType());
    if (!inType || inType.getMemorySpaceAsInt() != 2)
      return failure();

    ArrayRef<int64_t> shape = inType.getShape();
    int rank = shape.size();
    if (rank < 2)
      return failure();

    Location loc = op.getLoc();
    int64_t rows = shape[rank - 2] - 1;
    int64_t cols = shape[rank - 1] - 1;
    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows, 16);

    auto inFP = getFixedPointParams(op.getInScale().convertToDouble());
    double outScale = op.getOutScale().convertToDouble();
    auto outFP =
        getFixedPointParams((std::abs(outScale) > 1e-9) ? (1.0 / outScale) : 1.0);

    Value vInScale =
        rewriter.create<arith::ConstantIntOp>(loc, inFP.multiplier, 16);
    Value vInShift =
        rewriter.create<arith::ConstantIntOp>(loc, inFP.shift, 16);
    Value vInZp =
        rewriter.create<arith::ConstantIntOp>(loc, op.getInZp(), 32);
    Value vOutScale =
        rewriter.create<arith::ConstantIntOp>(loc, outFP.multiplier, 16);
    Value vOutShift =
        rewriter.create<arith::ConstantIntOp>(loc, outFP.shift, 16);
    Value vOutZp =
        rewriter.create<arith::ConstantIntOp>(loc, op.getOutZp(), 16);

    int8_t intTypeVal = 0;
    if (inType.getElementType().isInteger(16))
      intTypeVal = 1;
    else if (inType.getElementType().isInteger(32))
      intTypeVal = 2;
    Value vIntType = rewriter.create<arith::ConstantIntOp>(loc, intTypeVal, 8);
    Value vIsQuant = rewriter.create<arith::ConstantIntOp>(loc, true, 1);
    auto opTypeAttr =
        SFUOpTypeAttr::get(rewriter.getContext(), npux::SFUOpType::layernorm);

    rewriter.replaceOpWithNewOp<SfuRunOp>(op, opTypeAttr, vIntType, vIsQuant,
        inputMemRef, vCol, vRow, outputMemRef, vInZp, vOutZp, vInScale,
        vInShift, vOutScale, vOutShift);
    return success();
  }
};

} // namespace

// ============================================================================
// Registration
// ============================================================================
void npux::populateNpucoreSfuToNpuxPattern(RewritePatternSet &patterns) {
  patterns.add<NpucoreGeluToNpuxPattern>(patterns.getContext());
  patterns.add<NpucoreSoftmaxToNpuxPattern>(patterns.getContext());
  patterns.add<NpucoreLayerNormToNpuxPattern>(patterns.getContext());
  patterns.add<NpucoreLayoutNchwToNchwc32ToNpuxPattern>(
      patterns.getContext());
  patterns.add<NpucoreLayoutNchwc32ToNchwToNpuxPattern>(
      patterns.getContext());
  patterns.add<NpucoreTransposeToNpuxPattern>(patterns.getContext());
  patterns.add<NpucoreMaxPoolToNpuxPattern>(patterns.getContext());
}
