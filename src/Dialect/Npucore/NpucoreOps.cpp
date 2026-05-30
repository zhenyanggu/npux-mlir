//===============================================
// src/Dialect/Npucore/NpucoreOps.cpp
// this file defines npucore dialect ops.
//===============================================

#include "src/Dialect/Npucore/NpucoreOps.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace npucore;

#include "src/Dialect/Npucore/NpucoreDialect.cpp.inc"

namespace {

// ============================================================================
// Helper: normalize mixed index values so constant Values become IndexAttr.
// This keeps slice/result type inference aligned with MLIR's static shape path.
// ============================================================================
static SmallVector<OpFoldResult> normalizeMixedIndexValues(
    OpBuilder &b, ArrayRef<OpFoldResult> values) {
  SmallVector<OpFoldResult> normalized;
  normalized.reserve(values.size());
  for (OpFoldResult value : values) {
    if (std::optional<int64_t> cst = getConstantIntValue(value))
      normalized.push_back(b.getIndexAttr(*cst));
    else
      normalized.push_back(value);
  }
  return normalized;
}

// ============================================================================
// Helper: extract a static integer from a mixed index value when possible.
// ============================================================================
static int64_t getStaticIntValueOrDynamic(OpFoldResult value) {
  if (std::optional<int64_t> cst = getConstantIntValue(value))
    return *cst;
  return ShapedType::kDynamic;
}

// Helper: create a subset of a shaped value for tiling.
// Mirrors Linalg's standard slice materialization path.
// ============================================================================
static Value createSlice(OpBuilder &b, Location loc, Value value,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  auto shapedType = cast<ShapedType>(value.getType());
  SmallVector<OpFoldResult> normalizedOffsets =
      normalizeMixedIndexValues(b, offsets);
  SmallVector<OpFoldResult> normalizedSizes =
      normalizeMixedIndexValues(b, sizes);
  SmallVector<OpFoldResult> strides =
      normalizeMixedIndexValues(
          b, SmallVector<OpFoldResult>(shapedType.getRank(), b.getIndexAttr(1)));
  SmallVector<int64_t> staticOffsets;
  SmallVector<int64_t> staticSizes;
  SmallVector<int64_t> staticStrides(shapedType.getRank(), 1);

  staticOffsets.reserve(normalizedOffsets.size());
  staticSizes.reserve(normalizedSizes.size());
  for (OpFoldResult offset : normalizedOffsets)
    staticOffsets.push_back(getStaticIntValueOrDynamic(offset));
  for (OpFoldResult size : normalizedSizes)
    staticSizes.push_back(getStaticIntValueOrDynamic(size));

  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
    auto resultType = RankedTensorType::get(
        staticSizes, tensorType.getElementType(), tensorType.getEncoding());
    return b.create<tensor::ExtractSliceOp>(
               loc, resultType, value, normalizedOffsets, normalizedSizes, strides)
        .getResult();
  }

  return memref::SubViewOp::create(
             b, loc, value, normalizedOffsets, normalizedSizes, strides)
      .getResult();
}

// ============================================================================
// Helper: read one shaped dimension as OpFoldResult.
// ============================================================================
static OpFoldResult getDimAsOFR(
    OpBuilder &b, Location loc, Value value, int64_t dim) {
  auto shapedType = cast<ShapedType>(value.getType());
  if (!shapedType.isDynamicDim(dim))
    return b.getIndexAttr(shapedType.getDimSize(dim));
  if (isa<TensorType>(value.getType()))
    return OpFoldResult(b.create<tensor::DimOp>(loc, value, dim).getResult());
  return OpFoldResult(b.create<memref::DimOp>(loc, value, dim).getResult());
}

// ============================================================================
// Helper: multiply one index OFR by a constant.
// ============================================================================
static OpFoldResult mulOFRByConst(
    OpBuilder &b, Location loc, OpFoldResult value, int64_t constant) {
  AffineExpr d0;
  bindDims(b.getContext(), d0);
  return affine::makeComposedFoldedAffineApply(
      b, loc, d0 * constant, ArrayRef<OpFoldResult>{value});
}

// ============================================================================
// Helper: add two index OFRs.
// ============================================================================
static OpFoldResult addOFR(
    OpBuilder &b, Location loc, OpFoldResult lhs, OpFoldResult rhs) {
  AffineExpr d0, d1;
  bindDims(b.getContext(), d0, d1);
  return affine::makeComposedFoldedAffineApply(
      b, loc, d0 + d1, ArrayRef<OpFoldResult>{lhs, rhs});
}

// ============================================================================
// Helper: subtract a constant from one index OFR.
// ============================================================================
static OpFoldResult subConstFromOFR(
    OpBuilder &b, Location loc, OpFoldResult value, int64_t constant) {
  AffineExpr d0;
  bindDims(b.getContext(), d0);
  return affine::makeComposedFoldedAffineApply(
      b, loc, d0 - constant, ArrayRef<OpFoldResult>{value});
}

// ============================================================================
// Helper: floor-divide one index OFR by a positive constant.
// ============================================================================
static OpFoldResult floorDivOFRByConst(
    OpBuilder &b, Location loc, OpFoldResult value, int64_t constant) {
  AffineExpr d0;
  bindDims(b.getContext(), d0);
  return affine::makeComposedFoldedAffineApply(
      b, loc, d0.floorDiv(constant), ArrayRef<OpFoldResult>{value});
}

// ============================================================================
// Helper: ceil-divide one index OFR by a positive constant.
// ============================================================================
static OpFoldResult ceilDivOFRByConst(
    OpBuilder &b, Location loc, OpFoldResult value, int64_t constant) {
  AffineExpr d0;
  bindDims(b.getContext(), d0);
  return affine::makeComposedFoldedAffineApply(
      b, loc, d0.ceilDiv(constant), ArrayRef<OpFoldResult>{value});
}

// ============================================================================
// Helper: build the Gelu tile op for tensor or memref semantics.
// ============================================================================
static FailureOr<TilingResult> buildTiledGelu(
    GeluOp op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<Value> tiledInputs;
  for (Value input : op.getInputs())
    tiledInputs.push_back(createSlice(b, op.getLoc(), input, offsets, sizes));

  SmallVector<Value> tiledOutputs;
  for (Value output : op.getOutputs())
    tiledOutputs.push_back(createSlice(b, op.getLoc(), output, offsets, sizes));

  SmallVector<Type> resultTypes;
  if (op.hasTensorSemantics()) {
    for (Value output : tiledOutputs)
      resultTypes.push_back(output.getType());
  }

  auto tiledOp = b.create<GeluOp>(op.getLoc(), resultTypes, tiledInputs,
      tiledOutputs, op.getInScaleAttr(), op.getInZpAttr(), op.getOutScaleAttr(),
      op.getOutZpAttr());
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "in_scale" ||
        name == "in_zp" || name == "out_scale" || name == "out_zp")
      continue;
    tiledOp->setAttr(attr.getName(), attr.getValue());
  }

  TilingResult result;
  result.tiledOps.push_back(tiledOp.getOperation());
  if (op.hasTensorSemantics())
    result.tiledValues.append(
        tiledOp.getResultTensors().begin(), tiledOp.getResultTensors().end());
  return result;
}

// ============================================================================
// Helper: build the Softmax tile op for tensor or memref semantics.
// ============================================================================
static FailureOr<TilingResult> buildTiledSoftmax(
    SoftmaxOp op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<Value> tiledInputs;
  for (Value input : op.getInputs())
    tiledInputs.push_back(createSlice(b, op.getLoc(), input, offsets, sizes));

  SmallVector<Value> tiledOutputs;
  for (Value output : op.getOutputs())
    tiledOutputs.push_back(createSlice(b, op.getLoc(), output, offsets, sizes));

  SmallVector<Type> resultTypes;
  if (op.hasTensorSemantics()) {
    for (Value output : tiledOutputs)
      resultTypes.push_back(output.getType());
  }

  auto tiledOp = b.create<SoftmaxOp>(op.getLoc(), resultTypes, tiledInputs,
      tiledOutputs, op.getInScaleAttr(), op.getInZpAttr(),
      op.getOutScaleAttr(), op.getOutZpAttr(), op.getAxisAttr());
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "in_scale" ||
        name == "in_zp" || name == "out_scale" || name == "out_zp" ||
        name == "axis")
      continue;
    tiledOp->setAttr(attr.getName(), attr.getValue());
  }

  TilingResult result;
  result.tiledOps.push_back(tiledOp.getOperation());
  if (op.hasTensorSemantics())
    result.tiledValues.append(tiledOp.getResultTensors().begin(),
        tiledOp.getResultTensors().end());
  return result;
}

// ============================================================================
// Helper: build the LayerNorm tile op for tensor or memref semantics.
// ============================================================================
static FailureOr<TilingResult> buildTiledLayerNorm(
    LayerNormOp op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<Value> tiledInputs;
  for (Value input : op.getInputs())
    tiledInputs.push_back(createSlice(b, op.getLoc(), input, offsets, sizes));

  SmallVector<Value> tiledOutputs;
  for (Value output : op.getOutputs())
    tiledOutputs.push_back(createSlice(b, op.getLoc(), output, offsets, sizes));

  SmallVector<Type> resultTypes;
  if (op.hasTensorSemantics()) {
    for (Value output : tiledOutputs)
      resultTypes.push_back(output.getType());
  }

  auto tiledOp = b.create<LayerNormOp>(op.getLoc(), resultTypes, tiledInputs,
      tiledOutputs, op.getInScaleAttr(), op.getInZpAttr(),
      op.getOutScaleAttr(), op.getOutZpAttr(), op.getAxisAttr(),
      op.getEpsilonAttr());
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "in_scale" ||
        name == "in_zp" || name == "out_scale" || name == "out_zp" ||
        name == "axis" || name == "epsilon")
      continue;
    tiledOp->setAttr(attr.getName(), attr.getValue());
  }

  TilingResult result;
  result.tiledOps.push_back(tiledOp.getOperation());
  if (op.hasTensorSemantics())
    result.tiledValues.append(tiledOp.getResultTensors().begin(),
        tiledOp.getResultTensors().end());
  return result;
}

// ============================================================================
// Helper: build the MatAdd tile op for tensor or memref semantics.
// ============================================================================
static FailureOr<TilingResult> buildTiledMatAdd(
    MatAddOp op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<Value> tiledInputs;
  for (Value input : op.getInputs())
    tiledInputs.push_back(createSlice(b, op.getLoc(), input, offsets, sizes));

  SmallVector<Value> tiledOutputs;
  for (Value output : op.getOutputs())
    tiledOutputs.push_back(createSlice(b, op.getLoc(), output, offsets, sizes));

  SmallVector<Type> resultTypes;
  if (op.hasTensorSemantics()) {
    for (Value output : tiledOutputs)
      resultTypes.push_back(output.getType());
  }

  auto tiledOp = b.create<MatAddOp>(op.getLoc(), resultTypes, tiledInputs,
      tiledOutputs, op.getIn1ScaleAttr(), op.getIn1ZpAttr(),
      op.getIn2ScaleAttr(), op.getIn2ZpAttr(), op.getOutScaleAttr(),
      op.getOutZpAttr());
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "in1_scale" ||
        name == "in1_zp" || name == "in2_scale" || name == "in2_zp" ||
        name == "out_scale" || name == "out_zp")
      continue;
    tiledOp->setAttr(attr.getName(), attr.getValue());
  }

  TilingResult result;
  result.tiledOps.push_back(tiledOp.getOperation());
  if (op.hasTensorSemantics())
    result.tiledValues.append(tiledOp.getResultTensors().begin(),
        tiledOp.getResultTensors().end());
  return result;
}

// ============================================================================
// Helper: build the MatMul tile op for tensor or memref semantics.
// ============================================================================
static FailureOr<TilingResult> buildTiledMatMul(
    MatMulOp op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  unsigned outRank =
      cast<ShapedType>(op.getOutputs().front().getType()).getRank();

  SmallVector<Value> tiledInputs;
  if (outRank == 2) {
    tiledInputs.push_back(createSlice(
        b, op.getLoc(), op.getInputs()[0], {offsets[1], offsets[2]},
        {sizes[1], sizes[2]}));
    tiledInputs.push_back(createSlice(
        b, op.getLoc(), op.getInputs()[1], {offsets[2], offsets[0]},
        {sizes[2], sizes[0]}));
    if (op.getInputs().size() == 3) {
      Value bias = op.getInputs()[2];
      auto biasType = cast<ShapedType>(bias.getType());
      if (biasType.getRank() == 1) {
        tiledInputs.push_back(createSlice(
            b, op.getLoc(), bias, {offsets[0]}, {sizes[0]}));
      } else {
        tiledInputs.push_back(createSlice(
            b, op.getLoc(), bias, {offsets[1], offsets[0]},
            {sizes[1], sizes[0]}));
      }
    }
  } else {
    Value lhs = op.getInputs()[0];
    Value rhs = op.getInputs()[1];
    auto lhsType = cast<ShapedType>(lhs.getType());
    auto rhsType = cast<ShapedType>(rhs.getType());

    if (lhsType.getRank() == 3) {
      tiledInputs.push_back(createSlice(b, op.getLoc(), lhs,
          {offsets[0], offsets[2], offsets[3]},
          {sizes[0], sizes[2], sizes[3]}));
    } else {
      tiledInputs.push_back(createSlice(
          b, op.getLoc(), lhs, {offsets[2], offsets[3]},
          {sizes[2], sizes[3]}));
    }

    if (rhsType.getRank() == 3) {
      tiledInputs.push_back(createSlice(b, op.getLoc(), rhs,
          {offsets[0], offsets[3], offsets[1]},
          {sizes[0], sizes[3], sizes[1]}));
    } else {
      tiledInputs.push_back(createSlice(
          b, op.getLoc(), rhs, {offsets[3], offsets[1]},
          {sizes[3], sizes[1]}));
    }

    if (op.getInputs().size() == 3) {
      Value bias = op.getInputs()[2];
      auto biasType = cast<ShapedType>(bias.getType());
      if (biasType.getRank() == 1) {
        tiledInputs.push_back(createSlice(
            b, op.getLoc(), bias, {offsets[1]}, {sizes[1]}));
      } else if (biasType.getRank() == 2) {
        tiledInputs.push_back(createSlice(
            b, op.getLoc(), bias, {offsets[2], offsets[1]},
            {sizes[2], sizes[1]}));
      } else {
        tiledInputs.push_back(createSlice(b, op.getLoc(), bias,
            {offsets[0], offsets[2], offsets[1]},
            {sizes[0], sizes[2], sizes[1]}));
      }
    }
  }

  SmallVector<Value> tiledOutputs;
  if (outRank == 2) {
    tiledOutputs.push_back(createSlice(
        b, op.getLoc(), op.getOutputs()[0], {offsets[1], offsets[0]},
        {sizes[1], sizes[0]}));
  } else {
    tiledOutputs.push_back(createSlice(b, op.getLoc(), op.getOutputs()[0],
        {offsets[0], offsets[2], offsets[1]},
        {sizes[0], sizes[2], sizes[1]}));
  }

  SmallVector<Type> resultTypes;
  if (op.hasTensorSemantics()) {
    for (Value output : tiledOutputs)
      resultTypes.push_back(output.getType());
  }

  auto tiledOp = b.create<MatMulOp>(op.getLoc(), resultTypes, tiledInputs,
      tiledOutputs, op.getLhsScaleAttr(), op.getLhsZpAttr(),
      op.getRhsScaleAttr(), op.getRhsZpAttr(), op.getOutScaleAttr(),
      op.getOutZpAttr(), op.getWithBiasAttr(), op.getDoReluAttr(),
      op.getReluTypeAttr());
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "lhs_scale" ||
        name == "lhs_zp" || name == "rhs_scale" || name == "rhs_zp" ||
        name == "out_scale" || name == "out_zp" || name == "with_bias" ||
        name == "do_relu" || name == "relu_type")
      continue;
    tiledOp->setAttr(attr.getName(), attr.getValue());
  }

  TilingResult result;
  result.tiledOps.push_back(tiledOp.getOperation());
  if (op.hasTensorSemantics())
    result.tiledValues.append(tiledOp.getResultTensors().begin(),
        tiledOp.getResultTensors().end());
  return result;
}

// ============================================================================
// Helper: build the MaxPool tile op for tensor or memref semantics.
// ============================================================================
static FailureOr<TilingResult> buildTiledMaxPool(
    MaxPoolOp op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = op.getLoc();
  Value input = op.getInputs().front();
  Value output = op.getOutputs().front();

  int64_t strideH = cast<IntegerAttr>(op.getStrides()[0]).getInt();
  int64_t strideW = cast<IntegerAttr>(op.getStrides()[1]).getInt();
  int64_t dilationH = cast<IntegerAttr>(op.getDilations()[0]).getInt();
  int64_t dilationW = cast<IntegerAttr>(op.getDilations()[1]).getInt();
  int64_t kernelH = cast<IntegerAttr>(op.getKernelShape()[0]).getInt();
  int64_t kernelW = cast<IntegerAttr>(op.getKernelShape()[1]).getInt();

  SmallVector<OpFoldResult> outputOffsets(offsets.begin(), offsets.end());
  SmallVector<OpFoldResult> outputSizes(sizes.begin(), sizes.end());

  SmallVector<OpFoldResult> inputOffsets = {
      offsets[0], offsets[1], mulOFRByConst(b, loc, offsets[2], strideH),
      mulOFRByConst(b, loc, offsets[3], strideW)};
  SmallVector<OpFoldResult> inputSizes = {
      sizes[0], sizes[1],
      addOFR(b, loc,
          mulOFRByConst(b, loc, subConstFromOFR(b, loc, sizes[2], 1), strideH),
          b.getIndexAttr((kernelH - 1) * dilationH + 1)),
      addOFR(b, loc,
          mulOFRByConst(b, loc, subConstFromOFR(b, loc, sizes[3], 1), strideW),
          b.getIndexAttr((kernelW - 1) * dilationW + 1))};

  SmallVector<Value> tiledInputs = {
      createSlice(b, loc, input, inputOffsets, inputSizes)};
  SmallVector<Value> tiledOutputs = {
      createSlice(b, loc, output, outputOffsets, outputSizes)};

  SmallVector<Type> resultTypes;
  if (op.hasTensorSemantics())
    resultTypes.push_back(tiledOutputs.front().getType());

  auto tiledOp = b.create<MaxPoolOp>(loc, resultTypes, tiledInputs,
      tiledOutputs, op.getInScaleAttr(), op.getInZpAttr(),
      op.getOutScaleAttr(), op.getOutZpAttr(), op.getKernelShapeAttr(),
      op.getStridesAttr(), op.getDilationsAttr(), op.getPadsAttr());
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "in_scale" ||
        name == "in_zp" || name == "out_scale" || name == "out_zp" ||
        name == "kernel_shape" || name == "strides" ||
        name == "dilations" || name == "pads")
      continue;
    tiledOp->setAttr(attr.getName(), attr.getValue());
  }

  TilingResult result;
  result.tiledOps.push_back(tiledOp.getOperation());
  if (op.hasTensorSemantics())
    result.tiledValues.push_back(tiledOp.getResultTensors().front());
  return result;
}

// ============================================================================
// Helper: build the transpose tile op for tensor or memref semantics.
// ============================================================================
static FailureOr<TilingResult> buildTiledTranspose(
    TransposeOp op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  auto shapedType = cast<ShapedType>(op.getOutputs().front().getType());
  unsigned rank = shapedType.getRank();

  SmallVector<OpFoldResult> inputOffsets(offsets.begin(), offsets.end());
  SmallVector<OpFoldResult> inputSizes(sizes.begin(), sizes.end());
  if (rank >= 2) {
    std::swap(inputOffsets[rank - 1], inputOffsets[rank - 2]);
    std::swap(inputSizes[rank - 1], inputSizes[rank - 2]);
  }

  SmallVector<Value> tiledInputs;
  tiledInputs.push_back(createSlice(
      b, op.getLoc(), op.getInputs()[0], inputOffsets, inputSizes));

  SmallVector<Value> tiledOutputs;
  tiledOutputs.push_back(
      createSlice(b, op.getLoc(), op.getOutputs()[0], offsets, sizes));

  SmallVector<Type> resultTypes;
  if (op.hasTensorSemantics()) {
    for (Value output : tiledOutputs)
      resultTypes.push_back(output.getType());
  }

  auto tiledOp = b.create<TransposeOp>(
      op.getLoc(), resultTypes, tiledInputs, tiledOutputs);
  for (NamedAttribute attr : op->getAttrs()) {
    if (attr.getName().strref() == "operandSegmentSizes")
      continue;
    tiledOp->setAttr(attr.getName(), attr.getValue());
  }

  TilingResult result;
  result.tiledOps.push_back(tiledOp.getOperation());
  if (op.hasTensorSemantics())
    result.tiledValues.append(tiledOp.getResultTensors().begin(),
        tiledOp.getResultTensors().end());
  return result;
}

// ============================================================================
// Helper: build the NCHW -> NCHWc32 layout tile op.
// ============================================================================
static FailureOr<TilingResult> buildTiledLayoutNchwToNchwc32(
    LayoutNchwToNchwc32Op op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = op.getLoc();
  int64_t tileFactor = op.getTileFactor();

  SmallVector<OpFoldResult> inputOffsets = {offsets[0],
      addOFR(b, loc, mulOFRByConst(b, loc, offsets[1], tileFactor), offsets[4]),
      offsets[2], offsets[3]};
  SmallVector<OpFoldResult> inputSizes = {sizes[0],
      addOFR(b, loc,
          mulOFRByConst(
              b, loc, subConstFromOFR(b, loc, sizes[1], 1), tileFactor),
          sizes[4]),
      sizes[2], sizes[3]};

  SmallVector<Value> tiledInputs;
  tiledInputs.push_back(
      createSlice(b, loc, op.getInputs()[0], inputOffsets, inputSizes));

  SmallVector<Value> tiledOutputs;
  tiledOutputs.push_back(
      createSlice(b, loc, op.getOutputs()[0], offsets, sizes));

  SmallVector<Type> resultTypes;
  if (op.hasTensorSemantics()) {
    for (Value output : tiledOutputs)
      resultTypes.push_back(output.getType());
  }

  auto tiledOp = b.create<LayoutNchwToNchwc32Op>(loc, resultTypes, tiledInputs,
      tiledOutputs, op.getTileFactorAttr());
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "tile_factor")
      continue;
    tiledOp->setAttr(attr.getName(), attr.getValue());
  }

  TilingResult result;
  result.tiledOps.push_back(tiledOp.getOperation());
  if (op.hasTensorSemantics())
    result.tiledValues.append(tiledOp.getResultTensors().begin(),
        tiledOp.getResultTensors().end());
  return result;
}

// ============================================================================
// Helper: build the NCHWc32 -> NCHW layout tile op.
// ============================================================================
static FailureOr<TilingResult> buildTiledLayoutNchwc32ToNchw(
    LayoutNchwc32ToNchwOp op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = op.getLoc();
  int64_t tileFactor = op.getTileFactor();

  SmallVector<OpFoldResult> inputOffsets = {offsets[0],
      floorDivOFRByConst(b, loc, offsets[1], tileFactor), offsets[2], offsets[3],
      b.getIndexAttr(0)};
  SmallVector<OpFoldResult> inputSizes = {sizes[0],
      ceilDivOFRByConst(b, loc, sizes[1], tileFactor), sizes[2], sizes[3],
      b.getIndexAttr(tileFactor)};

  SmallVector<Value> tiledInputs;
  tiledInputs.push_back(
      createSlice(b, loc, op.getInputs()[0], inputOffsets, inputSizes));

  SmallVector<Value> tiledOutputs;
  tiledOutputs.push_back(
      createSlice(b, loc, op.getOutputs()[0], offsets, sizes));

  SmallVector<Type> resultTypes;
  if (op.hasTensorSemantics()) {
    for (Value output : tiledOutputs)
      resultTypes.push_back(output.getType());
  }

  auto tiledOp = b.create<LayoutNchwc32ToNchwOp>(loc, resultTypes, tiledInputs,
      tiledOutputs, op.getTileFactorAttr());
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "tile_factor")
      continue;
    tiledOp->setAttr(attr.getName(), attr.getValue());
  }

  TilingResult result;
  result.tiledOps.push_back(tiledOp.getOperation());
  if (op.hasTensorSemantics())
    result.tiledValues.append(tiledOp.getResultTensors().begin(),
        tiledOp.getResultTensors().end());
  return result;
}

// ============================================================================
// Helper: build the ACC to SPM move tile op for tensor or memref semantics.
// ============================================================================
static FailureOr<TilingResult> buildTiledMvAccToSpm(
    MvAccToSpmOp op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<Value> tiledInputs;
  SmallVector<Operation *> generatedSlices;
  for (Value input : op.getInputs()) {
    Value sliced = createSlice(b, op.getLoc(), input, offsets, sizes);
    tiledInputs.push_back(sliced);
    if (Operation *sliceOp = sliced.getDefiningOp())
      generatedSlices.push_back(sliceOp);
  }

  SmallVector<Value> tiledOutputs;
  for (Value output : op.getOutputs()) {
    Value sliced = createSlice(b, op.getLoc(), output, offsets, sizes);
    tiledOutputs.push_back(sliced);
    if (Operation *sliceOp = sliced.getDefiningOp())
      generatedSlices.push_back(sliceOp);
  }

  SmallVector<Type> resultTypes;
  if (op.hasTensorSemantics()) {
    for (Value output : tiledOutputs)
      resultTypes.push_back(output.getType());
  }

  auto tiledOp = b.create<MvAccToSpmOp>(
      op.getLoc(), resultTypes, tiledInputs, tiledOutputs);
  for (NamedAttribute attr : op->getAttrs()) {
    if (attr.getName().strref() == "operandSegmentSizes")
      continue;
    tiledOp->setAttr(attr.getName(), attr.getValue());
  }

  TilingResult result;
  result.tiledOps.push_back(tiledOp.getOperation());
  if (op.hasTensorSemantics())
    result.tiledValues.append(
        tiledOp.getResultTensors().begin(), tiledOp.getResultTensors().end());
  result.generatedSlices = std::move(generatedSlices);
  return result;
}

// ============================================================================
// Helper: build the DMA mvin tile op for tensor or memref semantics.
// ============================================================================
static FailureOr<TilingResult> buildTiledDmaMvin(
    DmaMvinOp op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<Value> tiledInputs;
  SmallVector<Operation *> generatedSlices;
  for (Value input : op.getInputs()) {
    Value sliced = createSlice(b, op.getLoc(), input, offsets, sizes);
    tiledInputs.push_back(sliced);
    if (Operation *sliceOp = sliced.getDefiningOp())
      generatedSlices.push_back(sliceOp);
  }

  SmallVector<Value> tiledOutputs;
  for (Value output : op.getOutputs()) {
    Value sliced = createSlice(b, op.getLoc(), output, offsets, sizes);
    tiledOutputs.push_back(sliced);
    if (Operation *sliceOp = sliced.getDefiningOp())
      generatedSlices.push_back(sliceOp);
  }

  SmallVector<Type> resultTypes;
  if (op.hasTensorSemantics()) {
    for (Value output : tiledOutputs)
      resultTypes.push_back(output.getType());
  }

  auto tiledOp = b.create<DmaMvinOp>(op.getLoc(), resultTypes, tiledInputs,
      tiledOutputs, op.getDmaTypeAttr(), op.getDmaColDimAttr(),
      op.getIsQuantAttr(), op.getQuantZeroAttr(), op.getQuantScaleAttr(),
      op.getQuantShiftAttr());
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "dma_type" ||
        name == "dma_col_dim" || name == "is_quant" ||
        name == "quant_zero" || name == "quant_scale" ||
        name == "quant_shift")
      continue;
    tiledOp->setAttr(attr.getName(), attr.getValue());
  }

  TilingResult result;
  result.tiledOps.push_back(tiledOp.getOperation());
  if (op.hasTensorSemantics()) {
    result.tiledValues.append(
        tiledOp.getResultTensors().begin(), tiledOp.getResultTensors().end());
  }
  result.generatedSlices = std::move(generatedSlices);
  return result;
}

// ============================================================================
// Helper: build the DMA mvout tile op for tensor or memref semantics.
// ============================================================================
static FailureOr<TilingResult> buildTiledDmaMvout(
    DmaMvoutOp op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<Value> tiledInputs;
  SmallVector<Operation *> generatedSlices;
  for (Value input : op.getInputs()) {
    Value sliced = createSlice(b, op.getLoc(), input, offsets, sizes);
    tiledInputs.push_back(sliced);
    if (Operation *sliceOp = sliced.getDefiningOp())
      generatedSlices.push_back(sliceOp);
  }

  SmallVector<Value> tiledOutputs;
  for (Value output : op.getOutputs()) {
    Value sliced = createSlice(b, op.getLoc(), output, offsets, sizes);
    tiledOutputs.push_back(sliced);
    if (Operation *sliceOp = sliced.getDefiningOp())
      generatedSlices.push_back(sliceOp);
  }

  SmallVector<Type> resultTypes;
  if (op.hasTensorSemantics()) {
    for (Value output : tiledOutputs)
      resultTypes.push_back(output.getType());
  }

  auto tiledOp = b.create<DmaMvoutOp>(op.getLoc(), resultTypes, tiledInputs,
      tiledOutputs, op.getDmaTypeAttr(), op.getDmaColDimAttr(),
      op.getIsQuantAttr(), op.getQuantZeroAttr(), op.getQuantScaleAttr(),
      op.getQuantShiftAttr());
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "dma_type" ||
        name == "dma_col_dim" || name == "is_quant" ||
        name == "quant_zero" || name == "quant_scale" ||
        name == "quant_shift")
      continue;
    tiledOp->setAttr(attr.getName(), attr.getValue());
  }

  TilingResult result;
  result.tiledOps.push_back(tiledOp.getOperation());
  if (op.hasTensorSemantics()) {
    result.tiledValues.append(
        tiledOp.getResultTensors().begin(), tiledOp.getResultTensors().end());
  }
  result.generatedSlices = std::move(generatedSlices);
  return result;
}

// ============================================================================
// Helper: build the Conv tile op for tensor or memref semantics.
// ============================================================================
static FailureOr<TilingResult> buildTiledConv(
    ConvOp op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = op.getLoc();
  Value input = op.getInputs()[0];
  Value weight = op.getInputs()[1];
  Value output = op.getOutputs()[0];
  auto strides = op.getStrides();
  auto dilations = op.getDilations();

  int64_t strideH = cast<IntegerAttr>(strides[0]).getInt();
  int64_t strideW = cast<IntegerAttr>(strides[1]).getInt();
  int64_t dilationH = cast<IntegerAttr>(dilations[0]).getInt();
  int64_t dilationW = cast<IntegerAttr>(dilations[1]).getInt();

  SmallVector<OpFoldResult> outputOffsets = {
      offsets[0], offsets[1], offsets[2], offsets[3], offsets[8]};
  SmallVector<OpFoldResult> outputSizes = {
      sizes[0], sizes[1], sizes[2], sizes[3], sizes[8]};

  OpFoldResult inputHOffset = addOFR(b, loc, mulOFRByConst(b, loc, offsets[2], strideH),
      mulOFRByConst(b, loc, offsets[5], dilationH));
  OpFoldResult inputWOffset = addOFR(b, loc, mulOFRByConst(b, loc, offsets[3], strideW),
      mulOFRByConst(b, loc, offsets[6], dilationW));
  OpFoldResult inputHSize = addOFR(b, loc,
      mulOFRByConst(b, loc, addOFR(b, loc, sizes[2], b.getIndexAttr(-1)), strideH),
      addOFR(b, loc,
          mulOFRByConst(
              b, loc, addOFR(b, loc, sizes[5], b.getIndexAttr(-1)), dilationH),
          b.getIndexAttr(1)));
  OpFoldResult inputWSize = addOFR(b, loc,
      mulOFRByConst(b, loc, addOFR(b, loc, sizes[3], b.getIndexAttr(-1)), strideW),
      addOFR(b, loc,
          mulOFRByConst(
              b, loc, addOFR(b, loc, sizes[6], b.getIndexAttr(-1)), dilationW),
          b.getIndexAttr(1)));

  SmallVector<OpFoldResult> inputOffsets = {
      offsets[0], offsets[4], inputHOffset, inputWOffset, offsets[7]};
  SmallVector<OpFoldResult> inputSizes = {
      sizes[0], sizes[4], inputHSize, inputWSize, sizes[7]};

  SmallVector<OpFoldResult> weightOffsets = {
      offsets[1], offsets[4], offsets[5], offsets[6], offsets[7], offsets[8]};
  SmallVector<OpFoldResult> weightSizes = {
      sizes[1], sizes[4], sizes[5], sizes[6], sizes[7], sizes[8]};

  SmallVector<Value> tiledInputs;
  tiledInputs.push_back(createSlice(b, loc, input, inputOffsets, inputSizes));
  tiledInputs.push_back(
      createSlice(b, loc, weight, weightOffsets, weightSizes));
  if (op.getInputs().size() == 3) {
    Value bias = op.getInputs()[2];
    SmallVector<OpFoldResult> biasOffsets = {offsets[1], offsets[8]};
    SmallVector<OpFoldResult> biasSizes = {sizes[1], sizes[8]};
    tiledInputs.push_back(createSlice(b, loc, bias, biasOffsets, biasSizes));
  }

  SmallVector<Value> tiledOutputs;
  tiledOutputs.push_back(
      createSlice(b, loc, output, outputOffsets, outputSizes));

  SmallVector<Type> resultTypes;
  if (op.hasTensorSemantics()) {
    for (Value output : tiledOutputs)
      resultTypes.push_back(output.getType());
  }

  auto tiledOp = b.create<ConvOp>(op.getLoc(), resultTypes, tiledInputs,
      tiledOutputs, op.getInScaleAttr(), op.getInZpAttr(), op.getWScaleAttr(),
      op.getWZpAttr(), op.getOutScaleAttr(), op.getOutZpAttr(),
      op.getPadsAttr(), op.getStridesAttr(), op.getDilationsAttr(),
      op.getGroupAttr(), op.getDoReluAttr(), op.getReluTypeAttr());
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "in_scale" ||
        name == "in_zp" || name == "w_scale" || name == "w_zp" ||
        name == "out_scale" || name == "out_zp" || name == "pads" ||
        name == "strides" || name == "dilations" || name == "group" ||
        name == "do_relu" || name == "relu_type")
      continue;
    tiledOp->setAttr(attr.getName(), attr.getValue());
  }

  TilingResult result;
  result.tiledOps.push_back(tiledOp.getOperation());
  if (op.hasTensorSemantics())
    result.tiledValues.append(
        tiledOp.getResultTensors().begin(), tiledOp.getResultTensors().end());
  return result;
}

// ============================================================================
// TilingInterface external model for npucore.gelu.
// ============================================================================
struct GeluTilingInterface
    : public TilingInterface::ExternalModel<GeluTilingInterface, GeluOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto geluOp = cast<GeluOp>(op);
    auto shapedType = cast<ShapedType>(geluOp.getOutputs().front().getType());
    return SmallVector<utils::IteratorType>(
        shapedType.getRank(), utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto geluOp = cast<GeluOp>(op);
    Value output = geluOp.getOutputs().front();
    auto shapedType = cast<ShapedType>(output.getType());
    SmallVector<Range> loopRanges;
    loopRanges.reserve(shapedType.getRank());
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    for (int64_t dim = 0; dim < shapedType.getRank(); ++dim) {
      OpFoldResult size = shapedType.isDynamicDim(dim)
                              ? OpFoldResult(b.create<tensor::DimOp>(
                                    geluOp.getLoc(), output, dim)
                                    .getResult())
                              : OpFoldResult(b.getIndexAttr(
                                    shapedType.getDimSize(dim)));
      loopRanges.push_back({zero, size, one});
    }
    return loopRanges;
  }

  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledGelu(cast<GeluOp>(op), b, offsets, sizes);
  }

  LogicalResult getResultTilePosition(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledGelu(cast<GeluOp>(op), b, offsets, sizes);
  }
};

// ============================================================================
// BufferizableOpInterface external model for npucore.gelu.
// ============================================================================
struct GeluBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          GeluBufferizableOpInterface, GeluOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options,
      bufferization::BufferizationState &state) const {
    auto geluOp = cast<GeluOp>(op);
    if (geluOp.hasBufferSemantics())
      return success();
    if (!geluOp.hasPureTensorSemantics())
      return op->emitError() << "npucore.gelu expects pure tensor semantics";

    SmallVector<Value> newInputs;
    for (OpOperand *inputOperand : geluOp.getDpsInputOperands()) {
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, inputOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newInputs.push_back(*buffer);
    }

    SmallVector<Value> newOutputs;
    for (OpResult result : geluOp.getResultTensors()) {
      OpOperand *initOperand = geluOp.getDpsInitOperand(result.getResultNumber());
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, initOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newOutputs.push_back(*buffer);
    }

    auto newOp = rewriter.create<GeluOp>(geluOp.getLoc(), TypeRange{}, newInputs,
        newOutputs, geluOp.getInScaleAttr(), geluOp.getInZpAttr(),
        geluOp.getOutScaleAttr(), geluOp.getOutZpAttr());
    for (NamedAttribute attr : geluOp->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (name == "operandSegmentSizes" || name == "in_scale" ||
          name == "in_zp" || name == "out_scale" || name == "out_zp")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);
    return success();
  }
};

// ============================================================================
// TilingInterface external model for npucore.softmax.
// ============================================================================
struct SoftmaxTilingInterface
    : public TilingInterface::ExternalModel<SoftmaxTilingInterface, SoftmaxOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto softmaxOp = cast<SoftmaxOp>(op);
    auto shapedType = cast<ShapedType>(softmaxOp.getOutputs().front().getType());
    return SmallVector<utils::IteratorType>(
        shapedType.getRank(), utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto softmaxOp = cast<SoftmaxOp>(op);
    Value output = softmaxOp.getOutputs().front();
    auto shapedType = cast<ShapedType>(output.getType());
    SmallVector<Range> loopRanges;
    loopRanges.reserve(shapedType.getRank());
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    for (int64_t dim = 0; dim < shapedType.getRank(); ++dim) {
      OpFoldResult size = shapedType.isDynamicDim(dim)
                              ? OpFoldResult(b.create<tensor::DimOp>(
                                    softmaxOp.getLoc(), output, dim)
                                    .getResult())
                              : OpFoldResult(b.getIndexAttr(
                                    shapedType.getDimSize(dim)));
      loopRanges.push_back({zero, size, one});
    }
    return loopRanges;
  }

  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledSoftmax(cast<SoftmaxOp>(op), b, offsets, sizes);
  }

  LogicalResult getResultTilePosition(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledSoftmax(cast<SoftmaxOp>(op), b, offsets, sizes);
  }
};

// ============================================================================
// BufferizableOpInterface external model for npucore.softmax.
// ============================================================================
struct SoftmaxBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          SoftmaxBufferizableOpInterface, SoftmaxOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options,
      bufferization::BufferizationState &state) const {
    auto softmaxOp = cast<SoftmaxOp>(op);
    if (softmaxOp.hasBufferSemantics())
      return success();
    if (!softmaxOp.hasPureTensorSemantics())
      return op->emitError() << "npucore.softmax expects pure tensor semantics";

    SmallVector<Value> newInputs;
    for (OpOperand *inputOperand : softmaxOp.getDpsInputOperands()) {
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, inputOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newInputs.push_back(*buffer);
    }

    SmallVector<Value> newOutputs;
    for (OpResult result : softmaxOp.getResultTensors()) {
      OpOperand *initOperand = softmaxOp.getDpsInitOperand(result.getResultNumber());
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, initOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newOutputs.push_back(*buffer);
    }

    auto newOp = rewriter.create<SoftmaxOp>(softmaxOp.getLoc(), TypeRange{},
        newInputs, newOutputs, softmaxOp.getInScaleAttr(),
        softmaxOp.getInZpAttr(), softmaxOp.getOutScaleAttr(),
        softmaxOp.getOutZpAttr(), softmaxOp.getAxisAttr());
    for (NamedAttribute attr : softmaxOp->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (name == "operandSegmentSizes" || name == "in_scale" ||
          name == "in_zp" || name == "out_scale" || name == "out_zp" ||
          name == "axis")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);
    return success();
  }
};

// ============================================================================
// TilingInterface external model for npucore.layernorm.
// ============================================================================
struct LayerNormTilingInterface
    : public TilingInterface::ExternalModel<LayerNormTilingInterface,
          LayerNormOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto layerNormOp = cast<LayerNormOp>(op);
    auto shapedType =
        cast<ShapedType>(layerNormOp.getOutputs().front().getType());
    return SmallVector<utils::IteratorType>(
        shapedType.getRank(), utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto layerNormOp = cast<LayerNormOp>(op);
    Value output = layerNormOp.getOutputs().front();
    auto shapedType = cast<ShapedType>(output.getType());
    SmallVector<Range> loopRanges;
    loopRanges.reserve(shapedType.getRank());
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    for (int64_t dim = 0; dim < shapedType.getRank(); ++dim) {
      OpFoldResult size = shapedType.isDynamicDim(dim)
                              ? OpFoldResult(b.create<tensor::DimOp>(
                                    layerNormOp.getLoc(), output, dim)
                                    .getResult())
                              : OpFoldResult(b.getIndexAttr(
                                    shapedType.getDimSize(dim)));
      loopRanges.push_back({zero, size, one});
    }
    return loopRanges;
  }

  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledLayerNorm(cast<LayerNormOp>(op), b, offsets, sizes);
  }

  LogicalResult getResultTilePosition(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledLayerNorm(cast<LayerNormOp>(op), b, offsets, sizes);
  }
};

// ============================================================================
// BufferizableOpInterface external model for npucore.layernorm.
// ============================================================================
struct LayerNormBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          LayerNormBufferizableOpInterface, LayerNormOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options,
      bufferization::BufferizationState &state) const {
    auto layerNormOp = cast<LayerNormOp>(op);
    if (layerNormOp.hasBufferSemantics())
      return success();
    if (!layerNormOp.hasPureTensorSemantics())
      return op->emitError()
             << "npucore.layernorm expects pure tensor semantics";

    SmallVector<Value> newInputs;
    for (OpOperand *inputOperand : layerNormOp.getDpsInputOperands()) {
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, inputOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newInputs.push_back(*buffer);
    }

    SmallVector<Value> newOutputs;
    for (OpResult result : layerNormOp.getResultTensors()) {
      OpOperand *initOperand =
          layerNormOp.getDpsInitOperand(result.getResultNumber());
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, initOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newOutputs.push_back(*buffer);
    }

    auto newOp = rewriter.create<LayerNormOp>(layerNormOp.getLoc(), TypeRange{},
        newInputs, newOutputs, layerNormOp.getInScaleAttr(),
        layerNormOp.getInZpAttr(), layerNormOp.getOutScaleAttr(),
        layerNormOp.getOutZpAttr(), layerNormOp.getAxisAttr(),
        layerNormOp.getEpsilonAttr());
    for (NamedAttribute attr : layerNormOp->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (name == "operandSegmentSizes" || name == "in_scale" ||
          name == "in_zp" || name == "out_scale" || name == "out_zp" ||
          name == "axis" || name == "epsilon")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);
    return success();
  }
};

// ============================================================================
// TilingInterface external model for npucore.matadd.
// ============================================================================
struct MatAddTilingInterface
    : public TilingInterface::ExternalModel<MatAddTilingInterface, MatAddOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto matAddOp = cast<MatAddOp>(op);
    auto shapedType = cast<ShapedType>(matAddOp.getOutputs().front().getType());
    return SmallVector<utils::IteratorType>(
        shapedType.getRank(), utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto matAddOp = cast<MatAddOp>(op);
    Value output = matAddOp.getOutputs().front();
    auto shapedType = cast<ShapedType>(output.getType());
    SmallVector<Range> loopRanges;
    loopRanges.reserve(shapedType.getRank());
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    for (int64_t dim = 0; dim < shapedType.getRank(); ++dim) {
      OpFoldResult size = shapedType.isDynamicDim(dim)
                              ? OpFoldResult(b.create<tensor::DimOp>(
                                    matAddOp.getLoc(), output, dim)
                                    .getResult())
                              : OpFoldResult(b.getIndexAttr(
                                    shapedType.getDimSize(dim)));
      loopRanges.push_back({zero, size, one});
    }
    return loopRanges;
  }

  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledMatAdd(cast<MatAddOp>(op), b, offsets, sizes);
  }

  LogicalResult getResultTilePosition(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledMatAdd(cast<MatAddOp>(op), b, offsets, sizes);
  }
};

// ============================================================================
// TilingInterface external model for npucore.matmul.
// ============================================================================
struct MatMulTilingInterface
    : public TilingInterface::ExternalModel<MatMulTilingInterface, MatMulOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto matmulOp = cast<MatMulOp>(op);
    unsigned outRank =
        cast<ShapedType>(matmulOp.getOutputs().front().getType()).getRank();
    if (outRank == 2) {
      return {utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::reduction};
    }
    return {utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::parallel, utils::IteratorType::reduction};
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto matmulOp = cast<MatMulOp>(op);
    Value lhs = matmulOp.getInputs()[0];
    Value rhs = matmulOp.getInputs()[1];
    Value output = matmulOp.getOutputs()[0];
    auto lhsType = cast<ShapedType>(lhs.getType());
    auto rhsType = cast<ShapedType>(rhs.getType());
    auto outType = cast<ShapedType>(output.getType());

    SmallVector<OpFoldResult> dims;
    if (outType.getRank() == 2) {
      dims.push_back(getDimAsOFR(b, matmulOp.getLoc(), output, 1));
      dims.push_back(getDimAsOFR(b, matmulOp.getLoc(), output, 0));
      dims.push_back(getDimAsOFR(b, matmulOp.getLoc(), lhs, lhsType.getRank() - 1));
    } else {
      Value batchRef = lhsType.getRank() == 3 ? lhs : rhs;
      dims.push_back(getDimAsOFR(b, matmulOp.getLoc(), batchRef, 0));
      dims.push_back(getDimAsOFR(b, matmulOp.getLoc(), output, 2));
      dims.push_back(getDimAsOFR(b, matmulOp.getLoc(), output, 1));
      dims.push_back(getDimAsOFR(b, matmulOp.getLoc(), lhs, lhsType.getRank() - 1));
    }

    SmallVector<Range> loopRanges;
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    for (OpFoldResult dim : dims)
      loopRanges.push_back({zero, dim, one});
    return loopRanges;
  }

  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledMatMul(cast<MatMulOp>(op), b, offsets, sizes);
  }

  LogicalResult getResultTilePosition(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    auto matmulOp = cast<MatMulOp>(op);
    unsigned outRank =
        cast<ShapedType>(matmulOp.getOutputs().front().getType()).getRank();
    if (outRank == 2) {
      resultOffsets.assign({offsets[1], offsets[0]});
      resultSizes.assign({sizes[1], sizes[0]});
    } else {
      resultOffsets.assign({offsets[0], offsets[2], offsets[1]});
      resultSizes.assign({sizes[0], sizes[2], sizes[1]});
    }
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    auto matmulOp = cast<MatMulOp>(op);
    auto lhsType = cast<ShapedType>(matmulOp.getInputs()[0].getType());
    auto outType = cast<ShapedType>(matmulOp.getOutputs().front().getType());
    if (outType.getRank() == 2) {
      iterDomainOffsets.assign(
          {offsets[1], offsets[0], b.getIndexAttr(0)});
      iterDomainSizes.assign({sizes[1], sizes[0],
          getDimAsOFR(
              b, matmulOp.getLoc(), matmulOp.getInputs()[0], lhsType.getRank() - 1)});
    } else {
      iterDomainOffsets.assign(
          {offsets[0], offsets[2], offsets[1], b.getIndexAttr(0)});
      iterDomainSizes.assign({sizes[0], sizes[2], sizes[1],
          getDimAsOFR(
              b, matmulOp.getLoc(), matmulOp.getInputs()[0], lhsType.getRank() - 1)});
    }
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    SmallVector<OpFoldResult> iterDomainOffsets;
    SmallVector<OpFoldResult> iterDomainSizes;
    if (failed(getIterationDomainTileFromResultTile(op, b, resultNumber,
            offsets, sizes, iterDomainOffsets, iterDomainSizes))) {
      return failure();
    }
    return buildTiledMatMul(
        cast<MatMulOp>(op), b, iterDomainOffsets, iterDomainSizes);
  }
};

// ============================================================================
// TilingInterface external model for npucore.maxpool.
// ============================================================================
struct MaxPoolTilingInterface
    : public TilingInterface::ExternalModel<MaxPoolTilingInterface, MaxPoolOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    return SmallVector<utils::IteratorType>(
        4, utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto maxpoolOp = cast<MaxPoolOp>(op);
    Value output = maxpoolOp.getOutputs().front();
    SmallVector<Range> loopRanges;
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    for (int64_t dim = 0; dim < 4; ++dim)
      loopRanges.push_back(
          {zero, getDimAsOFR(b, maxpoolOp.getLoc(), output, dim), one});
    return loopRanges;
  }

  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledMaxPool(cast<MaxPoolOp>(op), b, offsets, sizes);
  }

  LogicalResult getResultTilePosition(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    auto maxpoolOp = cast<MaxPoolOp>(op);
    Value output = maxpoolOp.getOutputs().front();
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    for (int64_t dim = 0; dim < 4; ++dim) {
      if (std::optional<int64_t> cst = getConstantIntValue(iterDomainSizes[dim]);
          cst && *cst == 0) {
        iterDomainSizes[dim] = getDimAsOFR(b, maxpoolOp.getLoc(), output, dim);
      }
    }
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    SmallVector<OpFoldResult> iterDomainOffsets;
    SmallVector<OpFoldResult> iterDomainSizes;
    if (failed(getIterationDomainTileFromResultTile(op, b, resultNumber,
            offsets, sizes, iterDomainOffsets, iterDomainSizes)))
      return failure();
    return buildTiledMaxPool(
        cast<MaxPoolOp>(op), b, iterDomainOffsets, iterDomainSizes);
  }
};

// ============================================================================
// TilingInterface external model for npucore.transpose.
// ============================================================================
struct TransposeTilingInterface
    : public TilingInterface::ExternalModel<TransposeTilingInterface,
          TransposeOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto transposeOp = cast<TransposeOp>(op);
    auto shapedType =
        cast<ShapedType>(transposeOp.getOutputs().front().getType());
    return SmallVector<utils::IteratorType>(
        shapedType.getRank(), utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto transposeOp = cast<TransposeOp>(op);
    Value output = transposeOp.getOutputs().front();
    auto shapedType = cast<ShapedType>(output.getType());
    SmallVector<Range> loopRanges;
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    for (int64_t dim = 0; dim < shapedType.getRank(); ++dim)
      loopRanges.push_back({zero, getDimAsOFR(b, transposeOp.getLoc(), output, dim), one});
    return loopRanges;
  }

  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledTranspose(cast<TransposeOp>(op), b, offsets, sizes);
  }

  LogicalResult getResultTilePosition(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledTranspose(cast<TransposeOp>(op), b, offsets, sizes);
  }
};

// ============================================================================
// TilingInterface external model for npucore.layout_nchw_to_nchwc32.
// ============================================================================
struct LayoutNchwToNchwc32TilingInterface
    : public TilingInterface::ExternalModel<LayoutNchwToNchwc32TilingInterface,
          LayoutNchwToNchwc32Op> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto layoutOp = cast<LayoutNchwToNchwc32Op>(op);
    auto shapedType = cast<ShapedType>(layoutOp.getOutputs().front().getType());
    return SmallVector<utils::IteratorType>(
        shapedType.getRank(), utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto layoutOp = cast<LayoutNchwToNchwc32Op>(op);
    Value output = layoutOp.getOutputs().front();
    auto shapedType = cast<ShapedType>(output.getType());
    SmallVector<Range> loopRanges;
    loopRanges.reserve(shapedType.getRank());
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    for (int64_t dim = 0; dim < shapedType.getRank(); ++dim)
      loopRanges.push_back({zero, getDimAsOFR(b, layoutOp.getLoc(), output, dim), one});
    return loopRanges;
  }

  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledLayoutNchwToNchwc32(
        cast<LayoutNchwToNchwc32Op>(op), b, offsets, sizes);
  }

  LogicalResult getResultTilePosition(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledLayoutNchwToNchwc32(
        cast<LayoutNchwToNchwc32Op>(op), b, offsets, sizes);
  }
};

// ============================================================================
// TilingInterface external model for npucore.layout_nchwc32_to_nchw.
// ============================================================================
struct LayoutNchwc32ToNchwTilingInterface
    : public TilingInterface::ExternalModel<LayoutNchwc32ToNchwTilingInterface,
          LayoutNchwc32ToNchwOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto layoutOp = cast<LayoutNchwc32ToNchwOp>(op);
    auto shapedType = cast<ShapedType>(layoutOp.getOutputs().front().getType());
    return SmallVector<utils::IteratorType>(
        shapedType.getRank(), utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto layoutOp = cast<LayoutNchwc32ToNchwOp>(op);
    Value output = layoutOp.getOutputs().front();
    auto shapedType = cast<ShapedType>(output.getType());
    SmallVector<Range> loopRanges;
    loopRanges.reserve(shapedType.getRank());
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    for (int64_t dim = 0; dim < shapedType.getRank(); ++dim)
      loopRanges.push_back({zero, getDimAsOFR(b, layoutOp.getLoc(), output, dim), one});
    return loopRanges;
  }

  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledLayoutNchwc32ToNchw(
        cast<LayoutNchwc32ToNchwOp>(op), b, offsets, sizes);
  }

  LogicalResult getResultTilePosition(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledLayoutNchwc32ToNchw(
        cast<LayoutNchwc32ToNchwOp>(op), b, offsets, sizes);
  }
};

// ============================================================================
// TilingInterface external model for npucore.mv_acc_to_spm.
// ============================================================================
struct MvAccToSpmTilingInterface
    : public TilingInterface::ExternalModel<MvAccToSpmTilingInterface,
          MvAccToSpmOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto moveOp = cast<MvAccToSpmOp>(op);
    auto shapedType = cast<ShapedType>(moveOp.getOutputs().front().getType());
    return SmallVector<utils::IteratorType>(
        shapedType.getRank(), utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto moveOp = cast<MvAccToSpmOp>(op);
    Value output = moveOp.getOutputs().front();
    auto shapedType = cast<ShapedType>(output.getType());
    SmallVector<Range> loopRanges;
    loopRanges.reserve(shapedType.getRank());
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    for (int64_t dim = 0; dim < shapedType.getRank(); ++dim) {
      OpFoldResult size = shapedType.isDynamicDim(dim)
                              ? OpFoldResult(b.create<tensor::DimOp>(
                                    moveOp.getLoc(), output, dim)
                                    .getResult())
                              : OpFoldResult(b.getIndexAttr(
                                    shapedType.getDimSize(dim)));
      loopRanges.push_back({zero, size, one});
    }
    return loopRanges;
  }

  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledMvAccToSpm(cast<MvAccToSpmOp>(op), b, offsets, sizes);
  }

  LogicalResult getResultTilePosition(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledMvAccToSpm(cast<MvAccToSpmOp>(op), b, offsets, sizes);
  }
};

// ============================================================================
// TilingInterface external model for npucore.dma_mvin.
// ============================================================================
struct DmaMvinTilingInterface
    : public TilingInterface::ExternalModel<DmaMvinTilingInterface,
          DmaMvinOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto dmaOp = cast<DmaMvinOp>(op);
    auto shapedType = cast<ShapedType>(dmaOp.getOutputs().front().getType());
    return SmallVector<utils::IteratorType>(
        shapedType.getRank(), utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto dmaOp = cast<DmaMvinOp>(op);
    Value output = dmaOp.getOutputs().front();
    auto shapedType = cast<ShapedType>(output.getType());
    SmallVector<Range> loopRanges;
    loopRanges.reserve(shapedType.getRank());
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    for (int64_t dim = 0; dim < shapedType.getRank(); ++dim) {
      OpFoldResult size = shapedType.isDynamicDim(dim)
                              ? OpFoldResult(
                                    b.create<tensor::DimOp>(dmaOp.getLoc(), output, dim)
                                        .getResult())
                              : OpFoldResult(
                                    b.getIndexAttr(shapedType.getDimSize(dim)));
      loopRanges.push_back({zero, size, one});
    }
    return loopRanges;
  }

  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledDmaMvin(cast<DmaMvinOp>(op), b, offsets, sizes);
  }

  LogicalResult getResultTilePosition(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledDmaMvin(cast<DmaMvinOp>(op), b, offsets, sizes);
  }
};

// ============================================================================
// TilingInterface external model for npucore.dma_mvout.
// ============================================================================
struct DmaMvoutTilingInterface
    : public TilingInterface::ExternalModel<DmaMvoutTilingInterface,
          DmaMvoutOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto dmaOp = cast<DmaMvoutOp>(op);
    auto shapedType = cast<ShapedType>(dmaOp.getOutputs().front().getType());
    return SmallVector<utils::IteratorType>(
        shapedType.getRank(), utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto dmaOp = cast<DmaMvoutOp>(op);
    Value output = dmaOp.getOutputs().front();
    auto shapedType = cast<ShapedType>(output.getType());
    SmallVector<Range> loopRanges;
    loopRanges.reserve(shapedType.getRank());
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    for (int64_t dim = 0; dim < shapedType.getRank(); ++dim) {
      OpFoldResult size = shapedType.isDynamicDim(dim)
                              ? OpFoldResult(
                                    b.create<tensor::DimOp>(dmaOp.getLoc(), output, dim)
                                        .getResult())
                              : OpFoldResult(
                                    b.getIndexAttr(shapedType.getDimSize(dim)));
      loopRanges.push_back({zero, size, one});
    }
    return loopRanges;
  }

  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledDmaMvout(cast<DmaMvoutOp>(op), b, offsets, sizes);
  }

  LogicalResult getResultTilePosition(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledDmaMvout(cast<DmaMvoutOp>(op), b, offsets, sizes);
  }
};

// ============================================================================
// TilingInterface external model for npucore.conv.
// ============================================================================
struct ConvTilingInterface
    : public TilingInterface::ExternalModel<ConvTilingInterface, ConvOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    return {
        utils::IteratorType::parallel,  utils::IteratorType::parallel,
        utils::IteratorType::parallel,  utils::IteratorType::parallel,
        utils::IteratorType::reduction, utils::IteratorType::reduction,
        utils::IteratorType::reduction, utils::IteratorType::reduction,
        utils::IteratorType::parallel};
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto convOp = cast<ConvOp>(op);
    Value input = convOp.getInputs()[0];
    Value weight = convOp.getInputs()[1];
    Value output = convOp.getOutputs()[0];
    SmallVector<OpFoldResult> dims = {
        getDimAsOFR(b, convOp.getLoc(), output, 0),
        getDimAsOFR(b, convOp.getLoc(), output, 1),
        getDimAsOFR(b, convOp.getLoc(), output, 2),
        getDimAsOFR(b, convOp.getLoc(), output, 3),
        getDimAsOFR(b, convOp.getLoc(), input, 1),
        getDimAsOFR(b, convOp.getLoc(), weight, 2),
        getDimAsOFR(b, convOp.getLoc(), weight, 3),
        getDimAsOFR(b, convOp.getLoc(), input, 4),
        getDimAsOFR(b, convOp.getLoc(), output, 4)};
    SmallVector<Range> loopRanges;
    loopRanges.reserve(dims.size());
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    for (OpFoldResult size : dims)
      loopRanges.push_back({zero, size, one});
    return loopRanges;
  }

  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    return buildTiledConv(cast<ConvOp>(op), b, offsets, sizes);
  }

  LogicalResult getResultTilePosition(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(
        {offsets[0], offsets[1], offsets[2], offsets[3], offsets[8]});
    resultSizes.assign({sizes[0], sizes[1], sizes[2], sizes[3], sizes[8]});
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    auto convOp = cast<ConvOp>(op);
    Value input = convOp.getInputs()[0];
    Value weight = convOp.getInputs()[1];
    Value output = convOp.getOutputs()[0];
    iterDomainOffsets.assign(
        {offsets[0], offsets[1], offsets[2], offsets[3], b.getIndexAttr(0),
            b.getIndexAttr(0), b.getIndexAttr(0), b.getIndexAttr(0),
            offsets[4]});
    iterDomainSizes.assign({sizes[0], sizes[1], sizes[2], sizes[3],
        getDimAsOFR(b, convOp.getLoc(), input, 1),
        getDimAsOFR(b, convOp.getLoc(), weight, 2),
        getDimAsOFR(b, convOp.getLoc(), weight, 3),
        getDimAsOFR(b, convOp.getLoc(), input, 4),
        getDimAsOFR(b, convOp.getLoc(), output, 4)});
    return success();
  }

  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b,
      unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes) const {
    SmallVector<OpFoldResult> iterDomainOffsets;
    SmallVector<OpFoldResult> iterDomainSizes;
    if (failed(getIterationDomainTileFromResultTile(op, b, resultNumber,
            offsets, sizes, iterDomainOffsets, iterDomainSizes))) {
      return failure();
    }
    return buildTiledConv(
        cast<ConvOp>(op), b, iterDomainOffsets, iterDomainSizes);
  }
};

// ============================================================================
// BufferizableOpInterface external model for npucore.matadd.
// ============================================================================
struct MatAddBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          MatAddBufferizableOpInterface, MatAddOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options,
      bufferization::BufferizationState &state) const {
    auto matAddOp = cast<MatAddOp>(op);
    if (matAddOp.hasBufferSemantics())
      return success();
    if (!matAddOp.hasPureTensorSemantics())
      return op->emitError() << "npucore.matadd expects pure tensor semantics";

    SmallVector<Value> newInputs;
    for (OpOperand *inputOperand : matAddOp.getDpsInputOperands()) {
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, inputOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newInputs.push_back(*buffer);
    }

    SmallVector<Value> newOutputs;
    for (OpResult result : matAddOp.getResultTensors()) {
      OpOperand *initOperand = matAddOp.getDpsInitOperand(result.getResultNumber());
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, initOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newOutputs.push_back(*buffer);
    }

    auto newOp = rewriter.create<MatAddOp>(matAddOp.getLoc(), TypeRange{},
        newInputs, newOutputs, matAddOp.getIn1ScaleAttr(),
        matAddOp.getIn1ZpAttr(), matAddOp.getIn2ScaleAttr(),
        matAddOp.getIn2ZpAttr(), matAddOp.getOutScaleAttr(),
        matAddOp.getOutZpAttr());
    for (NamedAttribute attr : matAddOp->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (name == "operandSegmentSizes" || name == "in1_scale" ||
          name == "in1_zp" || name == "in2_scale" || name == "in2_zp" ||
          name == "out_scale" || name == "out_zp")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);
    return success();
  }
};

// ============================================================================
// BufferizableOpInterface external model for npucore.matmul.
// ============================================================================
struct MatMulBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          MatMulBufferizableOpInterface, MatMulOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options,
      bufferization::BufferizationState &state) const {
    auto matmulOp = cast<MatMulOp>(op);
    if (matmulOp.hasBufferSemantics())
      return success();
    if (!matmulOp.hasPureTensorSemantics())
      return op->emitError() << "npucore.matmul expects pure tensor semantics";

    SmallVector<Value> newInputs;
    for (OpOperand *inputOperand : matmulOp.getDpsInputOperands()) {
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, inputOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newInputs.push_back(*buffer);
    }

    SmallVector<Value> newOutputs;
    for (OpResult result : matmulOp.getResultTensors()) {
      OpOperand *initOperand =
          matmulOp.getDpsInitOperand(result.getResultNumber());
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, initOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newOutputs.push_back(*buffer);
    }

    auto newOp = rewriter.create<MatMulOp>(matmulOp.getLoc(), TypeRange{},
        newInputs, newOutputs, matmulOp.getLhsScaleAttr(),
        matmulOp.getLhsZpAttr(), matmulOp.getRhsScaleAttr(),
        matmulOp.getRhsZpAttr(), matmulOp.getOutScaleAttr(),
        matmulOp.getOutZpAttr(), matmulOp.getWithBiasAttr(),
        matmulOp.getDoReluAttr(), matmulOp.getReluTypeAttr());
    for (NamedAttribute attr : matmulOp->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (name == "operandSegmentSizes" || name == "lhs_scale" ||
          name == "lhs_zp" || name == "rhs_scale" || name == "rhs_zp" ||
          name == "out_scale" || name == "out_zp" || name == "with_bias" ||
          name == "do_relu" || name == "relu_type")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);
    return success();
  }
};

// ============================================================================
// BufferizableOpInterface external model for npucore.maxpool.
// ============================================================================
struct MaxPoolBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          MaxPoolBufferizableOpInterface, MaxPoolOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options,
      bufferization::BufferizationState &state) const {
    auto maxpoolOp = cast<MaxPoolOp>(op);
    if (maxpoolOp.hasBufferSemantics())
      return success();
    if (!maxpoolOp.hasPureTensorSemantics())
      return op->emitError() << "npucore.maxpool expects pure tensor semantics";

    SmallVector<Value> newInputs;
    for (OpOperand *inputOperand : maxpoolOp.getDpsInputOperands()) {
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, inputOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newInputs.push_back(*buffer);
    }

    SmallVector<Value> newOutputs;
    for (OpResult result : maxpoolOp.getResultTensors()) {
      OpOperand *initOperand =
          maxpoolOp.getDpsInitOperand(result.getResultNumber());
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, initOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newOutputs.push_back(*buffer);
    }

    auto newOp = rewriter.create<MaxPoolOp>(maxpoolOp.getLoc(), TypeRange{},
        newInputs, newOutputs, maxpoolOp.getInScaleAttr(),
        maxpoolOp.getInZpAttr(), maxpoolOp.getOutScaleAttr(),
        maxpoolOp.getOutZpAttr(), maxpoolOp.getKernelShapeAttr(),
        maxpoolOp.getStridesAttr(), maxpoolOp.getDilationsAttr(),
        maxpoolOp.getPadsAttr());
    for (NamedAttribute attr : maxpoolOp->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (name == "operandSegmentSizes" || name == "in_scale" ||
          name == "in_zp" || name == "out_scale" || name == "out_zp" ||
          name == "kernel_shape" || name == "strides" ||
          name == "dilations" || name == "pads")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);
    return success();
  }
};

// ============================================================================
// BufferizableOpInterface external model for npucore.transpose.
// ============================================================================
struct TransposeBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          TransposeBufferizableOpInterface, TransposeOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options,
      bufferization::BufferizationState &state) const {
    auto transposeOp = cast<TransposeOp>(op);
    if (transposeOp.hasBufferSemantics())
      return success();
    if (!transposeOp.hasPureTensorSemantics())
      return op->emitError()
             << "npucore.transpose expects pure tensor semantics";

    SmallVector<Value> newInputs;
    for (OpOperand *inputOperand : transposeOp.getDpsInputOperands()) {
      FailureOr<Value> buffer = bufferization::getBuffer(
          rewriter, inputOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newInputs.push_back(*buffer);
    }

    SmallVector<Value> newOutputs;
    for (OpResult result : transposeOp.getResultTensors()) {
      OpOperand *initOperand =
          transposeOp.getDpsInitOperand(result.getResultNumber());
      FailureOr<Value> buffer = bufferization::getBuffer(
          rewriter, initOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newOutputs.push_back(*buffer);
    }

    auto newOp = rewriter.create<TransposeOp>(
        transposeOp.getLoc(), TypeRange{}, newInputs, newOutputs);
    for (NamedAttribute attr : transposeOp->getAttrs()) {
      if (attr.getName().strref() == "operandSegmentSizes")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);
    return success();
  }
};

// ============================================================================
// BufferizableOpInterface external model for npucore.layout_nchw_to_nchwc32.
// ============================================================================
struct LayoutNchwToNchwc32BufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          LayoutNchwToNchwc32BufferizableOpInterface,
          LayoutNchwToNchwc32Op> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options,
      bufferization::BufferizationState &state) const {
    auto layoutOp = cast<LayoutNchwToNchwc32Op>(op);
    if (layoutOp.hasBufferSemantics())
      return success();
    if (!layoutOp.hasPureTensorSemantics()) {
      return op->emitError()
             << "npucore.layout_nchw_to_nchwc32 expects pure tensor semantics";
    }

    SmallVector<Value> newInputs;
    for (OpOperand *inputOperand : layoutOp.getDpsInputOperands()) {
      FailureOr<Value> buffer = bufferization::getBuffer(
          rewriter, inputOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newInputs.push_back(*buffer);
    }

    SmallVector<Value> newOutputs;
    for (OpResult result : layoutOp.getResultTensors()) {
      OpOperand *initOperand = layoutOp.getDpsInitOperand(result.getResultNumber());
      FailureOr<Value> buffer = bufferization::getBuffer(
          rewriter, initOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newOutputs.push_back(*buffer);
    }

    auto newOp = rewriter.create<LayoutNchwToNchwc32Op>(layoutOp.getLoc(),
        TypeRange{}, newInputs, newOutputs, layoutOp.getTileFactorAttr());
    for (NamedAttribute attr : layoutOp->getAttrs()) {
      if (attr.getName().strref() == "operandSegmentSizes" ||
          attr.getName().strref() == "tile_factor")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);
    return success();
  }
};

// ============================================================================
// BufferizableOpInterface external model for npucore.layout_nchwc32_to_nchw.
// ============================================================================
struct LayoutNchwc32ToNchwBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          LayoutNchwc32ToNchwBufferizableOpInterface,
          LayoutNchwc32ToNchwOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options,
      bufferization::BufferizationState &state) const {
    auto layoutOp = cast<LayoutNchwc32ToNchwOp>(op);
    if (layoutOp.hasBufferSemantics())
      return success();
    if (!layoutOp.hasPureTensorSemantics()) {
      return op->emitError()
             << "npucore.layout_nchwc32_to_nchw expects pure tensor semantics";
    }

    SmallVector<Value> newInputs;
    for (OpOperand *inputOperand : layoutOp.getDpsInputOperands()) {
      FailureOr<Value> buffer = bufferization::getBuffer(
          rewriter, inputOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newInputs.push_back(*buffer);
    }

    SmallVector<Value> newOutputs;
    for (OpResult result : layoutOp.getResultTensors()) {
      OpOperand *initOperand = layoutOp.getDpsInitOperand(result.getResultNumber());
      FailureOr<Value> buffer = bufferization::getBuffer(
          rewriter, initOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newOutputs.push_back(*buffer);
    }

    auto newOp = rewriter.create<LayoutNchwc32ToNchwOp>(layoutOp.getLoc(),
        TypeRange{}, newInputs, newOutputs, layoutOp.getTileFactorAttr());
    for (NamedAttribute attr : layoutOp->getAttrs()) {
      if (attr.getName().strref() == "operandSegmentSizes" ||
          attr.getName().strref() == "tile_factor")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);
    return success();
  }
};

// ============================================================================
// BufferizableOpInterface external model for npucore.mv_acc_to_spm.
// ============================================================================
struct MvAccToSpmBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          MvAccToSpmBufferizableOpInterface, MvAccToSpmOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options,
      bufferization::BufferizationState &state) const {
    auto moveOp = cast<MvAccToSpmOp>(op);
    if (moveOp.hasBufferSemantics())
      return success();
    if (!moveOp.hasPureTensorSemantics())
      return op->emitError()
             << "npucore.mv_acc_to_spm expects pure tensor semantics";

    SmallVector<Value> newInputs;
    for (OpOperand *inputOperand : moveOp.getDpsInputOperands()) {
      FailureOr<Value> buffer = bufferization::getBuffer(
          rewriter, inputOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newInputs.push_back(*buffer);
    }

    SmallVector<Value> newOutputs;
    for (OpResult result : moveOp.getResultTensors()) {
      OpOperand *initOperand = moveOp.getDpsInitOperand(result.getResultNumber());
      FailureOr<Value> buffer = bufferization::getBuffer(
          rewriter, initOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newOutputs.push_back(*buffer);
    }

    auto newOp = rewriter.create<MvAccToSpmOp>(
        moveOp.getLoc(), TypeRange{}, newInputs, newOutputs);
    for (NamedAttribute attr : moveOp->getAttrs()) {
      if (attr.getName().strref() == "operandSegmentSizes")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);
    return success();
  }
};

// ============================================================================
// BufferizableOpInterface external model for npucore.dma_mvin.
// ============================================================================
struct DmaMvinBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          DmaMvinBufferizableOpInterface, DmaMvinOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options,
      bufferization::BufferizationState &state) const {
    auto dmaOp = cast<DmaMvinOp>(op);
    if (dmaOp.hasBufferSemantics())
      return success();
    if (!dmaOp.hasPureTensorSemantics())
      return op->emitError()
             << "npucore.dma_mvin expects pure tensor semantics";

    SmallVector<Value> newInputs;
    for (OpOperand *inputOperand : dmaOp.getDpsInputOperands()) {
      FailureOr<Value> buffer = bufferization::getBuffer(
          rewriter, inputOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newInputs.push_back(*buffer);
    }

    SmallVector<Value> newOutputs;
    for (OpResult result : dmaOp.getResultTensors()) {
      OpOperand *initOperand = dmaOp.getDpsInitOperand(result.getResultNumber());
      FailureOr<Value> buffer = bufferization::getBuffer(
          rewriter, initOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newOutputs.push_back(*buffer);
    }

    auto newOp = rewriter.create<DmaMvinOp>(dmaOp.getLoc(), TypeRange{},
        newInputs, newOutputs, dmaOp.getDmaTypeAttr(), dmaOp.getDmaColDimAttr(),
        dmaOp.getIsQuantAttr(), dmaOp.getQuantZeroAttr(),
        dmaOp.getQuantScaleAttr(), dmaOp.getQuantShiftAttr());
    for (NamedAttribute attr : dmaOp->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (name == "operandSegmentSizes" || name == "dma_type" ||
          name == "dma_col_dim" || name == "is_quant" ||
          name == "quant_zero" || name == "quant_scale" ||
          name == "quant_shift")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);
    return success();
  }
};

// ============================================================================
// BufferizableOpInterface external model for npucore.dma_mvout.
// ============================================================================
struct DmaMvoutBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          DmaMvoutBufferizableOpInterface, DmaMvoutOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options,
      bufferization::BufferizationState &state) const {
    auto dmaOp = cast<DmaMvoutOp>(op);
    if (dmaOp.hasBufferSemantics())
      return success();
    if (!dmaOp.hasPureTensorSemantics())
      return op->emitError()
             << "npucore.dma_mvout expects pure tensor semantics";

    SmallVector<Value> newInputs;
    for (OpOperand *inputOperand : dmaOp.getDpsInputOperands()) {
      FailureOr<Value> buffer = bufferization::getBuffer(
          rewriter, inputOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newInputs.push_back(*buffer);
    }

    SmallVector<Value> newOutputs;
    for (OpResult result : dmaOp.getResultTensors()) {
      OpOperand *initOperand = dmaOp.getDpsInitOperand(result.getResultNumber());
      FailureOr<Value> buffer = bufferization::getBuffer(
          rewriter, initOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newOutputs.push_back(*buffer);
    }

    auto newOp = rewriter.create<DmaMvoutOp>(dmaOp.getLoc(), TypeRange{},
        newInputs, newOutputs, dmaOp.getDmaTypeAttr(), dmaOp.getDmaColDimAttr(),
        dmaOp.getIsQuantAttr(), dmaOp.getQuantZeroAttr(),
        dmaOp.getQuantScaleAttr(), dmaOp.getQuantShiftAttr());
    for (NamedAttribute attr : dmaOp->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (name == "operandSegmentSizes" || name == "dma_type" ||
          name == "dma_col_dim" || name == "is_quant" ||
          name == "quant_zero" || name == "quant_scale" ||
          name == "quant_shift")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);
    return success();
  }
};

// ============================================================================
// BufferizableOpInterface external model for npucore.conv.
// ============================================================================
struct ConvBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          ConvBufferizableOpInterface, ConvOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options,
      bufferization::BufferizationState &state) const {
    auto convOp = cast<ConvOp>(op);
    if (convOp.hasBufferSemantics())
      return success();
    if (!convOp.hasPureTensorSemantics())
      return op->emitError() << "npucore.conv expects pure tensor semantics";

    SmallVector<Value> newInputs;
    for (OpOperand *inputOperand : convOp.getDpsInputOperands()) {
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, inputOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newInputs.push_back(*buffer);
    }

    SmallVector<Value> newOutputs;
    for (OpResult result : convOp.getResultTensors()) {
      OpOperand *initOperand = convOp.getDpsInitOperand(result.getResultNumber());
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, initOperand->get(), options, state);
      if (failed(buffer))
        return failure();
      newOutputs.push_back(*buffer);
    }

    auto newOp = rewriter.create<ConvOp>(convOp.getLoc(), TypeRange{}, newInputs,
        newOutputs, convOp.getInScaleAttr(), convOp.getInZpAttr(),
        convOp.getWScaleAttr(), convOp.getWZpAttr(), convOp.getOutScaleAttr(),
        convOp.getOutZpAttr(), convOp.getPadsAttr(), convOp.getStridesAttr(),
        convOp.getDilationsAttr(), convOp.getGroupAttr(),
        convOp.getDoReluAttr(), convOp.getReluTypeAttr());
    for (NamedAttribute attr : convOp->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (name == "operandSegmentSizes" || name == "in_scale" ||
          name == "in_zp" || name == "w_scale" || name == "w_zp" ||
          name == "out_scale" || name == "out_zp" || name == "pads" ||
          name == "strides" || name == "dilations" || name == "group" ||
          name == "do_relu" || name == "relu_type")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputs);
    return success();
  }
};

} // namespace

void NpucoreDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/Dialect/Npucore/NpucoreOps.cpp.inc"
      >();

  GeluOp::attachInterface<GeluTilingInterface>(*getContext());
  GeluOp::attachInterface<GeluBufferizableOpInterface>(*getContext());
  SoftmaxOp::attachInterface<SoftmaxTilingInterface>(*getContext());
  SoftmaxOp::attachInterface<SoftmaxBufferizableOpInterface>(*getContext());
  LayerNormOp::attachInterface<LayerNormTilingInterface>(*getContext());
  LayerNormOp::attachInterface<LayerNormBufferizableOpInterface>(*getContext());
  MatAddOp::attachInterface<MatAddTilingInterface>(*getContext());
  MatAddOp::attachInterface<MatAddBufferizableOpInterface>(*getContext());
  MatMulOp::attachInterface<MatMulTilingInterface>(*getContext());
  MatMulOp::attachInterface<MatMulBufferizableOpInterface>(*getContext());
  MaxPoolOp::attachInterface<MaxPoolTilingInterface>(*getContext());
  MaxPoolOp::attachInterface<MaxPoolBufferizableOpInterface>(*getContext());
  TransposeOp::attachInterface<TransposeTilingInterface>(*getContext());
  TransposeOp::attachInterface<TransposeBufferizableOpInterface>(*getContext());
  LayoutNchwToNchwc32Op::attachInterface<
      LayoutNchwToNchwc32TilingInterface>(*getContext());
  LayoutNchwToNchwc32Op::attachInterface<
      LayoutNchwToNchwc32BufferizableOpInterface>(*getContext());
  LayoutNchwc32ToNchwOp::attachInterface<
      LayoutNchwc32ToNchwTilingInterface>(*getContext());
  LayoutNchwc32ToNchwOp::attachInterface<
      LayoutNchwc32ToNchwBufferizableOpInterface>(*getContext());
  MvAccToSpmOp::attachInterface<MvAccToSpmTilingInterface>(*getContext());
  MvAccToSpmOp::attachInterface<MvAccToSpmBufferizableOpInterface>(
      *getContext());
  DmaMvinOp::attachInterface<DmaMvinTilingInterface>(*getContext());
  DmaMvinOp::attachInterface<DmaMvinBufferizableOpInterface>(*getContext());
  DmaMvoutOp::attachInterface<DmaMvoutTilingInterface>(*getContext());
  DmaMvoutOp::attachInterface<DmaMvoutBufferizableOpInterface>(*getContext());
  ConvOp::attachInterface<ConvTilingInterface>(*getContext());
  ConvOp::attachInterface<ConvBufferizableOpInterface>(*getContext());
}

// ============================================================================
// Npucore structured interface helpers.
// ============================================================================
template <typename OpTy>
static unsigned getFirstOutputRank(OpTy op) {
  return cast<ShapedType>(op.getOutputs().front().getType()).getRank();
}

// ============================================================================
// Gelu structured interface.
// ============================================================================
llvm::StringRef GeluOp::getNpucoreOpKind() { return "compute"; }

unsigned GeluOp::getLogicalLoopRank() { return getFirstOutputRank(*this); }

LogicalResult GeluOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Value output = getOutputs().front();
  auto resultType = cast<RankedTensorType>(getResultTensors().front().getType());
  SmallVector<OpFoldResult> dims;
  dims.reserve(resultType.getRank());
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dims.push_back(b.create<tensor::DimOp>(getLoc(), output, dim).getResult());
    else
      dims.push_back(b.getIndexAttr(resultType.getDimSize(dim)));
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

bool GeluOp::isReductionLoop(unsigned dim) { return false; }

bool GeluOp::requiresAccToSpmMove() { return false; }

// ============================================================================
// Softmax structured interface.
// ============================================================================
llvm::StringRef SoftmaxOp::getNpucoreOpKind() { return "compute"; }

unsigned SoftmaxOp::getLogicalLoopRank() { return getFirstOutputRank(*this); }

LogicalResult SoftmaxOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Value output = getOutputs().front();
  auto resultType = cast<RankedTensorType>(getResultTensors().front().getType());
  SmallVector<OpFoldResult> dims;
  dims.reserve(resultType.getRank());
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dims.push_back(b.create<tensor::DimOp>(getLoc(), output, dim).getResult());
    else
      dims.push_back(b.getIndexAttr(resultType.getDimSize(dim)));
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

bool SoftmaxOp::isReductionLoop(unsigned dim) {
  return dim == static_cast<unsigned>(getAxis());
}

bool SoftmaxOp::requiresAccToSpmMove() { return false; }

// ============================================================================
// LayerNorm structured interface.
// ============================================================================
llvm::StringRef LayerNormOp::getNpucoreOpKind() { return "compute"; }

unsigned LayerNormOp::getLogicalLoopRank() { return getFirstOutputRank(*this); }

LogicalResult LayerNormOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Value output = getOutputs().front();
  auto resultType = cast<RankedTensorType>(getResultTensors().front().getType());
  SmallVector<OpFoldResult> dims;
  dims.reserve(resultType.getRank());
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dims.push_back(b.create<tensor::DimOp>(getLoc(), output, dim).getResult());
    else
      dims.push_back(b.getIndexAttr(resultType.getDimSize(dim)));
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

bool LayerNormOp::isReductionLoop(unsigned dim) {
  return dim == static_cast<unsigned>(getAxis());
}

bool LayerNormOp::requiresAccToSpmMove() { return false; }

// ============================================================================
// MatAdd structured interface.
// ============================================================================
llvm::StringRef MatAddOp::getNpucoreOpKind() { return "compute"; }

unsigned MatAddOp::getLogicalLoopRank() { return getFirstOutputRank(*this); }

LogicalResult MatAddOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Value output = getOutputs().front();
  auto resultType = cast<RankedTensorType>(getResultTensors().front().getType());
  SmallVector<OpFoldResult> dims;
  dims.reserve(resultType.getRank());
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dims.push_back(b.create<tensor::DimOp>(getLoc(), output, dim).getResult());
    else
      dims.push_back(b.getIndexAttr(resultType.getDimSize(dim)));
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

bool MatAddOp::isReductionLoop(unsigned dim) { return false; }

bool MatAddOp::requiresAccToSpmMove() { return false; }

// ============================================================================
// MatMul structured interface.
// ============================================================================
llvm::StringRef MatMulOp::getNpucoreOpKind() { return "compute"; }

unsigned MatMulOp::getLogicalLoopRank() {
  unsigned outRank = getFirstOutputRank(*this);
  return outRank == 2 ? 3 : 4;
}

LogicalResult MatMulOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Value output = getOutputs().front();
  auto resultType = cast<RankedTensorType>(getResultTensors().front().getType());
  SmallVector<OpFoldResult> dims;
  dims.reserve(resultType.getRank());
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dims.push_back(b.create<tensor::DimOp>(getLoc(), output, dim).getResult());
    else
      dims.push_back(b.getIndexAttr(resultType.getDimSize(dim)));
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

bool MatMulOp::isReductionLoop(unsigned dim) {
  unsigned outRank = getFirstOutputRank(*this);
  return dim == (outRank == 2 ? 2 : 3);
}

bool MatMulOp::requiresAccToSpmMove() { return true; }

// ============================================================================
// MaxPool structured interface.
// ============================================================================
llvm::StringRef MaxPoolOp::getNpucoreOpKind() { return "compute"; }

unsigned MaxPoolOp::getLogicalLoopRank() { return 4; }

LogicalResult MaxPoolOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Value output = getOutputs().front();
  auto resultType = cast<RankedTensorType>(getResultTensors().front().getType());
  SmallVector<OpFoldResult> dims;
  dims.reserve(resultType.getRank());
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dims.push_back(b.create<tensor::DimOp>(getLoc(), output, dim).getResult());
    else
      dims.push_back(b.getIndexAttr(resultType.getDimSize(dim)));
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

bool MaxPoolOp::isReductionLoop(unsigned dim) { return false; }

bool MaxPoolOp::requiresAccToSpmMove() { return false; }

// ============================================================================
// Transpose structured interface.
// ============================================================================
llvm::StringRef TransposeOp::getNpucoreOpKind() { return "layout"; }

unsigned TransposeOp::getLogicalLoopRank() { return getFirstOutputRank(*this); }

LogicalResult TransposeOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Value output = getOutputs().front();
  auto resultType = cast<RankedTensorType>(getResultTensors().front().getType());
  SmallVector<OpFoldResult> dims;
  dims.reserve(resultType.getRank());
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dims.push_back(b.create<tensor::DimOp>(getLoc(), output, dim).getResult());
    else
      dims.push_back(b.getIndexAttr(resultType.getDimSize(dim)));
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

bool TransposeOp::isReductionLoop(unsigned dim) { return false; }

bool TransposeOp::requiresAccToSpmMove() { return false; }

// ============================================================================
// Layout NCHW to NCHWc32 structured interface.
// ============================================================================
llvm::StringRef LayoutNchwToNchwc32Op::getNpucoreOpKind() { return "layout"; }

unsigned LayoutNchwToNchwc32Op::getLogicalLoopRank() {
  return getFirstOutputRank(*this);
}

LogicalResult LayoutNchwToNchwc32Op::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Value output = getOutputs().front();
  auto resultType = cast<RankedTensorType>(getResultTensors().front().getType());
  SmallVector<OpFoldResult> dims;
  dims.reserve(resultType.getRank());
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dims.push_back(b.create<tensor::DimOp>(getLoc(), output, dim).getResult());
    else
      dims.push_back(b.getIndexAttr(resultType.getDimSize(dim)));
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

bool LayoutNchwToNchwc32Op::isReductionLoop(unsigned dim) { return false; }

bool LayoutNchwToNchwc32Op::requiresAccToSpmMove() { return false; }

// ============================================================================
// Layout NCHWc32 to NCHW structured interface.
// ============================================================================
llvm::StringRef LayoutNchwc32ToNchwOp::getNpucoreOpKind() { return "layout"; }

unsigned LayoutNchwc32ToNchwOp::getLogicalLoopRank() {
  return getFirstOutputRank(*this);
}

LogicalResult LayoutNchwc32ToNchwOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Value output = getOutputs().front();
  auto resultType = cast<RankedTensorType>(getResultTensors().front().getType());
  SmallVector<OpFoldResult> dims;
  dims.reserve(resultType.getRank());
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dims.push_back(b.create<tensor::DimOp>(getLoc(), output, dim).getResult());
    else
      dims.push_back(b.getIndexAttr(resultType.getDimSize(dim)));
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

bool LayoutNchwc32ToNchwOp::isReductionLoop(unsigned dim) { return false; }

bool LayoutNchwc32ToNchwOp::requiresAccToSpmMove() { return false; }

// ============================================================================
// ACC to SPM move structured interface.
// ============================================================================
llvm::StringRef MvAccToSpmOp::getNpucoreOpKind() { return "move"; }

unsigned MvAccToSpmOp::getLogicalLoopRank() { return getFirstOutputRank(*this); }

LogicalResult MvAccToSpmOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Value output = getOutputs().front();
  auto resultType = cast<RankedTensorType>(getResultTensors().front().getType());
  SmallVector<OpFoldResult> dims;
  dims.reserve(resultType.getRank());
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dims.push_back(b.create<tensor::DimOp>(getLoc(), output, dim).getResult());
    else
      dims.push_back(b.getIndexAttr(resultType.getDimSize(dim)));
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

bool MvAccToSpmOp::isReductionLoop(unsigned dim) { return false; }

bool MvAccToSpmOp::requiresAccToSpmMove() { return false; }

// ============================================================================
// DMA mvin structured interface.
// ============================================================================
llvm::StringRef DmaMvinOp::getNpucoreOpKind() { return "move"; }

unsigned DmaMvinOp::getLogicalLoopRank() { return getFirstOutputRank(*this); }

LogicalResult DmaMvinOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Value output = getOutputs().front();
  auto resultType = cast<RankedTensorType>(getResultTensors().front().getType());
  SmallVector<OpFoldResult> dims;
  dims.reserve(resultType.getRank());
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dims.push_back(b.create<tensor::DimOp>(getLoc(), output, dim).getResult());
    else
      dims.push_back(b.getIndexAttr(resultType.getDimSize(dim)));
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

bool DmaMvinOp::isReductionLoop(unsigned dim) { return false; }

bool DmaMvinOp::requiresAccToSpmMove() { return false; }

// ============================================================================
// DMA mvout structured interface.
// ============================================================================
llvm::StringRef DmaMvoutOp::getNpucoreOpKind() { return "move"; }

unsigned DmaMvoutOp::getLogicalLoopRank() { return getFirstOutputRank(*this); }

LogicalResult DmaMvoutOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Value output = getOutputs().front();
  auto resultType = cast<RankedTensorType>(getResultTensors().front().getType());
  SmallVector<OpFoldResult> dims;
  dims.reserve(resultType.getRank());
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dims.push_back(b.create<tensor::DimOp>(getLoc(), output, dim).getResult());
    else
      dims.push_back(b.getIndexAttr(resultType.getDimSize(dim)));
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

bool DmaMvoutOp::isReductionLoop(unsigned dim) { return false; }

bool DmaMvoutOp::requiresAccToSpmMove() { return false; }

// ============================================================================
// Conv structured interface.
// ============================================================================
llvm::StringRef ConvOp::getNpucoreOpKind() { return "compute"; }

unsigned ConvOp::getLogicalLoopRank() { return 9; }

LogicalResult ConvOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Value output = getOutputs().front();
  auto resultType = cast<RankedTensorType>(getResultTensors().front().getType());
  SmallVector<OpFoldResult> dims;
  dims.reserve(resultType.getRank());
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dims.push_back(b.create<tensor::DimOp>(getLoc(), output, dim).getResult());
    else
      dims.push_back(b.getIndexAttr(resultType.getDimSize(dim)));
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

bool ConvOp::isReductionLoop(unsigned dim) { return dim >= 4 && dim <= 7; }

bool ConvOp::requiresAccToSpmMove() { return true; }

// ============================================================================
// Gelu verifier.
// ============================================================================
LogicalResult GeluOp::verifyInvariantsImpl(Operation *op) {
  auto geluOp = cast<GeluOp>(op);

  if (geluOp.getInputs().size() != 1)
    return geluOp.emitOpError("expects exactly one input");
  if (geluOp.getOutputs().size() != 1)
    return geluOp.emitOpError("expects exactly one output");

  Type inputType = geluOp.getInputs().front().getType();
  Type outputType = geluOp.getOutputs().front().getType();
  if (!isa<ShapedType>(inputType) || !isa<ShapedType>(outputType))
    return geluOp.emitOpError("expects shaped input/output");
  if (cast<ShapedType>(inputType).getShape() !=
      cast<ShapedType>(outputType).getShape())
    return geluOp.emitOpError("expects input and output shapes to match");

  bool inputIsTensor = isa<TensorType>(inputType);
  bool outputIsTensor = isa<TensorType>(outputType);
  bool inputIsMemref = isa<BaseMemRefType>(inputType);
  bool outputIsMemref = isa<BaseMemRefType>(outputType);

  if (geluOp.hasTensorSemantics()) {
    if (!(inputIsTensor && outputIsTensor))
      return geluOp.emitOpError(
          "tensor semantics require tensor input and tensor output");
    if (geluOp.getResultTensors().size() != 1)
      return geluOp.emitOpError("tensor semantics require one tensor result");
    if (geluOp.getResultTensors().front().getType() != outputType)
      return geluOp.emitOpError("tensor result type must match outs type");
    return success();
  }

  if (!(inputIsMemref && outputIsMemref))
    return geluOp.emitOpError(
        "buffer semantics require memref input and memref output");
  return success();
}

LogicalResult GeluOp::fold(
    FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

// ============================================================================
// Gelu memory effects.
// ============================================================================
void GeluOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (hasTensorSemantics())
    return;

  effects.emplace_back(MemoryEffects::Read::get(), &*getInputsMutable().begin(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &*getOutputsMutable().begin(),
      SideEffects::DefaultResource::get());
}

// ============================================================================
// Softmax verifier.
// ============================================================================
LogicalResult SoftmaxOp::verifyInvariantsImpl(Operation *op) {
  auto softmaxOp = cast<SoftmaxOp>(op);

  if (softmaxOp.getInputs().size() != 1)
    return softmaxOp.emitOpError("expects exactly one input");
  if (softmaxOp.getOutputs().size() != 1)
    return softmaxOp.emitOpError("expects exactly one output");

  Type inputType = softmaxOp.getInputs().front().getType();
  Type outputType = softmaxOp.getOutputs().front().getType();
  if (!isa<ShapedType>(inputType) || !isa<ShapedType>(outputType))
    return softmaxOp.emitOpError("expects shaped input/output");
  if (cast<ShapedType>(inputType).getShape() !=
      cast<ShapedType>(outputType).getShape())
    return softmaxOp.emitOpError("expects input and output shapes to match");

  auto outputShapedType = cast<ShapedType>(outputType);
  int64_t rank = outputShapedType.getRank();
  int64_t axis = softmaxOp.getAxis();
  if (rank == 0)
    return softmaxOp.emitOpError("expects ranked tensor or memref");
  if (axis < -rank || axis >= rank)
    return softmaxOp.emitOpError("axis is out of range for result rank");

  bool inputIsTensor = isa<TensorType>(inputType);
  bool outputIsTensor = isa<TensorType>(outputType);
  bool inputIsMemref = isa<BaseMemRefType>(inputType);
  bool outputIsMemref = isa<BaseMemRefType>(outputType);

  if (softmaxOp.hasTensorSemantics()) {
    if (!(inputIsTensor && outputIsTensor))
      return softmaxOp.emitOpError(
          "tensor semantics require tensor input and tensor output");
    if (softmaxOp.getResultTensors().size() != 1)
      return softmaxOp.emitOpError("tensor semantics require one tensor result");
    if (softmaxOp.getResultTensors().front().getType() != outputType)
      return softmaxOp.emitOpError("tensor result type must match outs type");
    return success();
  }

  if (!(inputIsMemref && outputIsMemref))
    return softmaxOp.emitOpError(
        "buffer semantics require memref input and memref output");
  return success();
}

LogicalResult SoftmaxOp::fold(
    FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

// ============================================================================
// Softmax memory effects.
// ============================================================================
void SoftmaxOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (hasTensorSemantics())
    return;

  effects.emplace_back(MemoryEffects::Read::get(), &*getInputsMutable().begin(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &*getOutputsMutable().begin(),
      SideEffects::DefaultResource::get());
}

// ============================================================================
// LayerNorm verifier.
// ============================================================================
LogicalResult LayerNormOp::verifyInvariantsImpl(Operation *op) {
  auto layerNormOp = cast<LayerNormOp>(op);

  if (layerNormOp.getInputs().size() != 1)
    return layerNormOp.emitOpError("expects exactly one input");
  if (layerNormOp.getOutputs().size() != 1)
    return layerNormOp.emitOpError("expects exactly one output");

  Type inputType = layerNormOp.getInputs().front().getType();
  Type outputType = layerNormOp.getOutputs().front().getType();
  if (!isa<ShapedType>(inputType) || !isa<ShapedType>(outputType))
    return layerNormOp.emitOpError("expects shaped input/output");
  if (cast<ShapedType>(inputType).getShape() !=
      cast<ShapedType>(outputType).getShape())
    return layerNormOp.emitOpError("expects input and output shapes to match");

  auto outputShapedType = cast<ShapedType>(outputType);
  int64_t rank = outputShapedType.getRank();
  int64_t axis = layerNormOp.getAxis();
  if (rank == 0)
    return layerNormOp.emitOpError("expects ranked tensor or memref");
  if (axis < -rank || axis >= rank)
    return layerNormOp.emitOpError("axis is out of range for result rank");

  bool inputIsTensor = isa<TensorType>(inputType);
  bool outputIsTensor = isa<TensorType>(outputType);
  bool inputIsMemref = isa<BaseMemRefType>(inputType);
  bool outputIsMemref = isa<BaseMemRefType>(outputType);

  if (layerNormOp.hasTensorSemantics()) {
    if (!(inputIsTensor && outputIsTensor))
      return layerNormOp.emitOpError(
          "tensor semantics require tensor input and tensor output");
    if (layerNormOp.getResultTensors().size() != 1)
      return layerNormOp.emitOpError(
          "tensor semantics require one tensor result");
    if (layerNormOp.getResultTensors().front().getType() != outputType)
      return layerNormOp.emitOpError("tensor result type must match outs type");
    return success();
  }

  if (!(inputIsMemref && outputIsMemref))
    return layerNormOp.emitOpError(
        "buffer semantics require memref input and memref output");
  return success();
}

LogicalResult LayerNormOp::fold(
    FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

// ============================================================================
// LayerNorm memory effects.
// ============================================================================
void LayerNormOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (hasTensorSemantics())
    return;

  effects.emplace_back(MemoryEffects::Read::get(), &*getInputsMutable().begin(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &*getOutputsMutable().begin(),
      SideEffects::DefaultResource::get());
}

// ============================================================================
// MatAdd verifier.
// ============================================================================
LogicalResult MatAddOp::verifyInvariantsImpl(Operation *op) {
  auto matAddOp = cast<MatAddOp>(op);

  if (matAddOp.getInputs().size() != 2)
    return matAddOp.emitOpError("expects exactly two inputs");
  if (matAddOp.getOutputs().size() != 1)
    return matAddOp.emitOpError("expects exactly one output");

  Type inputTypeA = matAddOp.getInputs()[0].getType();
  Type inputTypeB = matAddOp.getInputs()[1].getType();
  Type outputType = matAddOp.getOutputs().front().getType();
  if (!isa<ShapedType>(inputTypeA) || !isa<ShapedType>(inputTypeB) ||
      !isa<ShapedType>(outputType))
    return matAddOp.emitOpError("expects shaped inputs/output");

  if (cast<ShapedType>(inputTypeA).getShape() !=
      cast<ShapedType>(inputTypeB).getShape())
    return matAddOp.emitOpError("expects identical input shapes");
  if (cast<ShapedType>(inputTypeA).getShape() !=
      cast<ShapedType>(outputType).getShape())
    return matAddOp.emitOpError("expects output shape to match input shapes");

  bool inputATensor = isa<TensorType>(inputTypeA);
  bool inputBTensor = isa<TensorType>(inputTypeB);
  bool outputTensor = isa<TensorType>(outputType);
  bool inputAMemref = isa<BaseMemRefType>(inputTypeA);
  bool inputBMemref = isa<BaseMemRefType>(inputTypeB);
  bool outputMemref = isa<BaseMemRefType>(outputType);

  if (matAddOp.hasTensorSemantics()) {
    if (!(inputATensor && inputBTensor && outputTensor))
      return matAddOp.emitOpError(
          "tensor semantics require tensor inputs and tensor output");
    if (matAddOp.getResultTensors().size() != 1)
      return matAddOp.emitOpError(
          "tensor semantics require one tensor result");
    if (matAddOp.getResultTensors().front().getType() != outputType)
      return matAddOp.emitOpError("tensor result type must match outs type");
    return success();
  }

  if (!(inputAMemref && inputBMemref && outputMemref))
    return matAddOp.emitOpError(
        "buffer semantics require memref inputs and memref output");
  return success();
}

LogicalResult MatAddOp::fold(
    FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

// ============================================================================
// MatAdd memory effects.
// ============================================================================
void MatAddOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (hasTensorSemantics())
    return;

  effects.emplace_back(MemoryEffects::Read::get(), &*getInputsMutable().begin(),
      SideEffects::DefaultResource::get());
  auto secondInput = getInputsMutable().begin();
  ++secondInput;
  effects.emplace_back(MemoryEffects::Read::get(), &*secondInput,
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &*getOutputsMutable().begin(),
      SideEffects::DefaultResource::get());
}

// ============================================================================
// MatMul verifier.
// ============================================================================
LogicalResult MatMulOp::verifyInvariantsImpl(Operation *op) {
  auto matmulOp = cast<MatMulOp>(op);
  auto sameOrDynamic = [](int64_t lhs, int64_t rhs) {
    return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs == rhs;
  };

  if (matmulOp.getInputs().size() != 2 && matmulOp.getInputs().size() != 3)
    return matmulOp.emitOpError("expects two or three inputs");
  if (matmulOp.getOutputs().size() != 1)
    return matmulOp.emitOpError("expects exactly one output");

  Type outputType = matmulOp.getOutputs().front().getType();
  if (!isa<ShapedType>(outputType))
    return matmulOp.emitOpError("expects shaped output");
  for (Type inputType : matmulOp.getInputs().getTypes()) {
    if (!isa<ShapedType>(inputType))
      return matmulOp.emitOpError("expects shaped inputs");
  }

  auto outShapedType = cast<ShapedType>(outputType);
  if (outShapedType.getRank() != 2 && outShapedType.getRank() != 3)
    return matmulOp.emitOpError("expects rank-2 or rank-3 output");

  auto lhsType = cast<ShapedType>(matmulOp.getInputs()[0].getType());
  auto rhsType = cast<ShapedType>(matmulOp.getInputs()[1].getType());
  if (lhsType.getRank() != 2 && lhsType.getRank() != 3)
    return matmulOp.emitOpError("expects lhs rank to be 2 or 3");
  if (rhsType.getRank() != 2 && rhsType.getRank() != 3)
    return matmulOp.emitOpError("expects rhs rank to be 2 or 3");
  if (lhsType.getRank() != rhsType.getRank() ||
      lhsType.getRank() != outShapedType.getRank())
    return matmulOp.emitOpError(
        "expects lhs, rhs, and output to have the same rank");

  int64_t rank = outShapedType.getRank();
  int64_t lhsM = lhsType.getShape()[rank - 2];
  int64_t lhsK = lhsType.getShape()[rank - 1];
  int64_t rhsK = rhsType.getShape()[rank - 2];
  int64_t rhsN = rhsType.getShape()[rank - 1];
  int64_t outM = outShapedType.getShape()[rank - 2];
  int64_t outN = outShapedType.getShape()[rank - 1];

  if (!sameOrDynamic(lhsK, rhsK))
    return matmulOp.emitOpError(
        "expects lhs K dimension to match rhs K dimension");
  if (!sameOrDynamic(lhsM, outM))
    return matmulOp.emitOpError(
        "expects output M dimension to match lhs M dimension");
  if (!sameOrDynamic(rhsN, outN))
    return matmulOp.emitOpError(
        "expects output N dimension to match rhs N dimension");

  if (rank == 3) {
    int64_t lhsBatch = lhsType.getShape()[0];
    int64_t rhsBatch = rhsType.getShape()[0];
    int64_t outBatch = outShapedType.getShape()[0];
    if (!sameOrDynamic(lhsBatch, rhsBatch) ||
        !sameOrDynamic(lhsBatch, outBatch))
      return matmulOp.emitOpError(
          "expects batch dimensions of lhs, rhs, and output to match");
  }

  if (matmulOp.getInputs().size() == 3) {
    auto biasType = cast<ShapedType>(matmulOp.getInputs()[2].getType());
    if (biasType.getRank() == 1) {
      if (!sameOrDynamic(biasType.getShape()[0], outN))
        return matmulOp.emitOpError(
            "expects rank-1 bias length to match output N dimension");
    } else if (biasType.getRank() == 2) {
      if (!sameOrDynamic(biasType.getShape()[0], outM) ||
          !sameOrDynamic(biasType.getShape()[1], outN))
        return matmulOp.emitOpError(
            "expects rank-2 bias shape to match output MxN dimensions");
    } else if (biasType.getRank() == 3) {
      if (rank != 3)
        return matmulOp.emitOpError(
            "expects rank-3 bias only for batched matmul");
      if (!sameOrDynamic(biasType.getShape()[0], outShapedType.getShape()[0]) ||
          !sameOrDynamic(biasType.getShape()[1], outM) ||
          !sameOrDynamic(biasType.getShape()[2], outN))
        return matmulOp.emitOpError(
            "expects rank-3 bias shape to match output BxMxN dimensions");
    } else {
      return matmulOp.emitOpError("expects bias rank to be 1, 2, or 3");
    }
  }

  bool allTensor = llvm::all_of(matmulOp.getInputs().getTypes(),
                       [](Type type) { return isa<TensorType>(type); }) &&
      isa<TensorType>(outputType);
  bool allMemref = llvm::all_of(matmulOp.getInputs().getTypes(),
                       [](Type type) { return isa<BaseMemRefType>(type); }) &&
      isa<BaseMemRefType>(outputType);

  if (matmulOp.hasTensorSemantics()) {
    if (!allTensor)
      return matmulOp.emitOpError(
          "tensor semantics require tensor inputs and tensor output");
    if (matmulOp.getResultTensors().size() != 1)
      return matmulOp.emitOpError(
          "tensor semantics require one tensor result");
    if (matmulOp.getResultTensors().front().getType() != outputType)
      return matmulOp.emitOpError("tensor result type must match outs type");
    return success();
  }

  if (!allMemref)
    return matmulOp.emitOpError(
        "buffer semantics require memref inputs and memref output");
  return success();
}

LogicalResult MatMulOp::fold(
    FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

// ============================================================================
// MatMul memory effects.
// ============================================================================
void MatMulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (hasTensorSemantics())
    return;

  for (OpOperand &inputOperand : getInputsMutable())
    effects.emplace_back(MemoryEffects::Read::get(), &inputOperand,
        SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &*getOutputsMutable().begin(),
      SideEffects::DefaultResource::get());
}

// ============================================================================
// MaxPool verifier.
// ============================================================================
LogicalResult MaxPoolOp::verifyInvariantsImpl(Operation *op) {
  auto maxpoolOp = cast<MaxPoolOp>(op);
  auto sameOrDynamic = [](int64_t lhs, int64_t rhs) {
    return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs == rhs;
  };
  auto computeOutDim = [](int64_t input, int64_t kernel, int64_t padBefore,
                           int64_t padAfter, int64_t stride, int64_t dilation) {
    int64_t effectiveKernel = (kernel - 1) * dilation + 1;
    return ((input + padBefore + padAfter - effectiveKernel) / stride) + 1;
  };

  if (maxpoolOp.getInputs().size() != 1)
    return maxpoolOp.emitOpError("expects exactly one input");
  if (maxpoolOp.getOutputs().size() != 1)
    return maxpoolOp.emitOpError("expects exactly one output");
  if (maxpoolOp.getKernelShape().size() != 2)
    return maxpoolOp.emitOpError("expects kernel_shape length to be 2");
  if (maxpoolOp.getStrides().size() != 2)
    return maxpoolOp.emitOpError("expects strides length to be 2");
  if (maxpoolOp.getDilations().size() != 2)
    return maxpoolOp.emitOpError("expects dilations length to be 2");
  if (maxpoolOp.getPads().size() != 4)
    return maxpoolOp.emitOpError("expects pads length to be 4");

  int64_t kernelH = cast<IntegerAttr>(maxpoolOp.getKernelShape()[0]).getInt();
  int64_t kernelW = cast<IntegerAttr>(maxpoolOp.getKernelShape()[1]).getInt();
  int64_t strideH = cast<IntegerAttr>(maxpoolOp.getStrides()[0]).getInt();
  int64_t strideW = cast<IntegerAttr>(maxpoolOp.getStrides()[1]).getInt();
  int64_t dilationH = cast<IntegerAttr>(maxpoolOp.getDilations()[0]).getInt();
  int64_t dilationW = cast<IntegerAttr>(maxpoolOp.getDilations()[1]).getInt();
  int64_t padTop = cast<IntegerAttr>(maxpoolOp.getPads()[0]).getInt();
  int64_t padLeft = cast<IntegerAttr>(maxpoolOp.getPads()[1]).getInt();
  int64_t padBottom = cast<IntegerAttr>(maxpoolOp.getPads()[2]).getInt();
  int64_t padRight = cast<IntegerAttr>(maxpoolOp.getPads()[3]).getInt();

  if (kernelH < 1 || kernelW < 1)
    return maxpoolOp.emitOpError("expects kernel_shape values >= 1");
  if (strideH < 1 || strideW < 1)
    return maxpoolOp.emitOpError("expects stride values >= 1");
  if (dilationH < 1 || dilationW < 1)
    return maxpoolOp.emitOpError("expects dilation values >= 1");
  if (padTop < 0 || padLeft < 0 || padBottom < 0 || padRight < 0)
    return maxpoolOp.emitOpError("expects pad values >= 0");

  Type inputType = maxpoolOp.getInputs().front().getType();
  Type outputType = maxpoolOp.getOutputs().front().getType();
  if (!isa<ShapedType>(inputType) || !isa<ShapedType>(outputType))
    return maxpoolOp.emitOpError("expects shaped input/output");

  auto inputShape = cast<ShapedType>(inputType);
  auto outputShape = cast<ShapedType>(outputType);
  if (inputShape.getRank() != 4)
    return maxpoolOp.emitOpError("expects rank-4 NCHW input");
  if (outputShape.getRank() != 4)
    return maxpoolOp.emitOpError("expects rank-4 NCHW output");

  bool allTensor = isa<TensorType>(inputType) && isa<TensorType>(outputType);
  bool allMemref =
      isa<BaseMemRefType>(inputType) && isa<BaseMemRefType>(outputType);

  if (maxpoolOp.hasTensorSemantics()) {
    if (!allTensor)
      return maxpoolOp.emitOpError(
          "tensor semantics require tensor input and tensor output");
    if (maxpoolOp.getResultTensors().size() != 1)
      return maxpoolOp.emitOpError(
          "tensor semantics require one tensor result");
    if (maxpoolOp.getResultTensors().front().getType() != outputType)
      return maxpoolOp.emitOpError("tensor result type must match outs type");
  } else if (!allMemref) {
    return maxpoolOp.emitOpError(
        "buffer semantics require memref input and memref output");
  }

  ArrayRef<int64_t> inShape = inputShape.getShape();
  ArrayRef<int64_t> outShape = outputShape.getShape();
  if (!sameOrDynamic(inShape[0], outShape[0]))
    return maxpoolOp.emitOpError("expects batch dimension to be preserved");
  if (!sameOrDynamic(inShape[1], outShape[1]))
    return maxpoolOp.emitOpError("expects channel dimension to be preserved");
  if (inShape[2] != ShapedType::kDynamic && outShape[2] != ShapedType::kDynamic) {
    int64_t expectedOh = computeOutDim(
        inShape[2], kernelH, padTop, padBottom, strideH, dilationH);
    if (expectedOh != outShape[2])
      return maxpoolOp.emitOpError(
          "expects output height to match maxpool formula");
  }
  if (inShape[3] != ShapedType::kDynamic && outShape[3] != ShapedType::kDynamic) {
    int64_t expectedOw = computeOutDim(
        inShape[3], kernelW, padLeft, padRight, strideW, dilationW);
    if (expectedOw != outShape[3])
      return maxpoolOp.emitOpError(
          "expects output width to match maxpool formula");
  }
  return success();
}

LogicalResult MaxPoolOp::fold(
    FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

// ============================================================================
// MaxPool memory effects.
// ============================================================================
void MaxPoolOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (hasTensorSemantics())
    return;

  effects.emplace_back(MemoryEffects::Read::get(), &*getInputsMutable().begin(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &*getOutputsMutable().begin(),
      SideEffects::DefaultResource::get());
}

// ============================================================================
// Transpose verifier.
// ============================================================================
LogicalResult TransposeOp::verifyInvariantsImpl(Operation *op) {
  auto transposeOp = cast<TransposeOp>(op);

  if (transposeOp.getInputs().size() != 1)
    return transposeOp.emitOpError("expects exactly one input");
  if (transposeOp.getOutputs().size() != 1)
    return transposeOp.emitOpError("expects exactly one output");

  Type inputType = transposeOp.getInputs().front().getType();
  Type outputType = transposeOp.getOutputs().front().getType();
  if (!isa<ShapedType>(inputType) || !isa<ShapedType>(outputType))
    return transposeOp.emitOpError("expects shaped input/output");

  auto inputShape = cast<ShapedType>(inputType);
  auto outputShape = cast<ShapedType>(outputType);
  if (inputShape.getRank() < 2 || outputShape.getRank() < 2)
    return transposeOp.emitOpError("expects rank >= 2");
  if (inputShape.getRank() != outputShape.getRank())
    return transposeOp.emitOpError("expects identical input/output rank");

  bool allTensor = isa<TensorType>(inputType) && isa<TensorType>(outputType);
  bool allMemref =
      isa<BaseMemRefType>(inputType) && isa<BaseMemRefType>(outputType);

  if (transposeOp.hasTensorSemantics()) {
    if (!allTensor)
      return transposeOp.emitOpError(
          "tensor semantics require tensor input and tensor output");
    if (transposeOp.getResultTensors().size() != 1)
      return transposeOp.emitOpError(
          "tensor semantics require one tensor result");
    if (transposeOp.getResultTensors().front().getType() != outputType)
      return transposeOp.emitOpError("tensor result type must match outs type");
  } else if (!allMemref) {
    return transposeOp.emitOpError(
        "buffer semantics require memref input and memref output");
  }

  auto inDims = inputShape.getShape();
  auto outDims = outputShape.getShape();
  for (int64_t i = 0; i < inputShape.getRank() - 2; ++i) {
    if (inDims[i] != ShapedType::kDynamic && outDims[i] != ShapedType::kDynamic &&
        inDims[i] != outDims[i]) {
      return transposeOp.emitOpError(
          "expects leading dimensions to be preserved");
    }
  }
  if (inDims[inputShape.getRank() - 1] != ShapedType::kDynamic &&
      outDims[outputShape.getRank() - 2] != ShapedType::kDynamic &&
      inDims[inputShape.getRank() - 1] != outDims[outputShape.getRank() - 2]) {
    return transposeOp.emitOpError("expects last input dimension to map to output rank-2");
  }
  if (inDims[inputShape.getRank() - 2] != ShapedType::kDynamic &&
      outDims[outputShape.getRank() - 1] != ShapedType::kDynamic &&
      inDims[inputShape.getRank() - 2] != outDims[outputShape.getRank() - 1]) {
    return transposeOp.emitOpError("expects input rank-2 dimension to map to output last dimension");
  }
  return success();
}

LogicalResult TransposeOp::fold(
    FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

// ============================================================================
// Transpose memory effects.
// ============================================================================
void TransposeOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (hasTensorSemantics())
    return;

  effects.emplace_back(MemoryEffects::Read::get(), &*getInputsMutable().begin(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &*getOutputsMutable().begin(),
      SideEffects::DefaultResource::get());
}

// ============================================================================
// Layout NCHW to NCHWc32 verifier.
// ============================================================================
LogicalResult LayoutNchwToNchwc32Op::verifyInvariantsImpl(Operation *op) {
  auto layoutOp = cast<LayoutNchwToNchwc32Op>(op);

  if (layoutOp.getInputs().size() != 1)
    return layoutOp.emitOpError("expects exactly one input");
  if (layoutOp.getOutputs().size() != 1)
    return layoutOp.emitOpError("expects exactly one output");
  if (layoutOp.getTileFactor() <= 0)
    return layoutOp.emitOpError("expects tile_factor > 0");

  Type inputType = layoutOp.getInputs().front().getType();
  Type outputType = layoutOp.getOutputs().front().getType();
  if (!isa<ShapedType>(inputType) || !isa<ShapedType>(outputType))
    return layoutOp.emitOpError("expects shaped input/output");

  auto inputShape = cast<ShapedType>(inputType);
  auto outputShape = cast<ShapedType>(outputType);
  if (inputShape.getRank() != 4)
    return layoutOp.emitOpError("expects 4D input");
  if (outputShape.getRank() != 5)
    return layoutOp.emitOpError("expects 5D output");

  bool allTensor = isa<TensorType>(inputType) && isa<TensorType>(outputType);
  bool allMemref =
      isa<BaseMemRefType>(inputType) && isa<BaseMemRefType>(outputType);

  if (layoutOp.hasTensorSemantics()) {
    if (!allTensor)
      return layoutOp.emitOpError(
          "tensor semantics require tensor input and tensor output");
    if (layoutOp.getResultTensors().size() != 1)
      return layoutOp.emitOpError(
          "tensor semantics require one tensor result");
    if (layoutOp.getResultTensors().front().getType() != outputType)
      return layoutOp.emitOpError("tensor result type must match outs type");
  } else if (!allMemref) {
    return layoutOp.emitOpError(
        "buffer semantics require memref input and memref output");
  }

  auto inputDims = inputShape.getShape();
  auto outputDims = outputShape.getShape();
  int64_t tileFactor = layoutOp.getTileFactor();
  if (outputDims[4] != ShapedType::kDynamic && outputDims[4] != tileFactor) {
    return layoutOp.emitOpError("expects output last dimension to equal tile_factor");
  }
  if (inputDims[0] != ShapedType::kDynamic && outputDims[0] != ShapedType::kDynamic &&
      inputDims[0] != outputDims[0]) {
    return layoutOp.emitOpError("expects batch dimension to be preserved");
  }
  if (inputDims[2] != ShapedType::kDynamic && outputDims[2] != ShapedType::kDynamic &&
      inputDims[2] != outputDims[2]) {
    return layoutOp.emitOpError("expects height dimension to be preserved");
  }
  if (inputDims[3] != ShapedType::kDynamic && outputDims[3] != ShapedType::kDynamic &&
      inputDims[3] != outputDims[3]) {
    return layoutOp.emitOpError("expects width dimension to be preserved");
  }
  if (inputDims[1] != ShapedType::kDynamic && outputDims[1] != ShapedType::kDynamic) {
    int64_t expectedChunks = llvm::divideCeil(inputDims[1], tileFactor);
    if (outputDims[1] != expectedChunks) {
      return layoutOp.emitOpError("expects packed channel chunk dimension to match tile_factor");
    }
  }
  return success();
}

LogicalResult LayoutNchwToNchwc32Op::fold(
    FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

// ============================================================================
// Layout NCHW to NCHWc32 memory effects.
// ============================================================================
void LayoutNchwToNchwc32Op::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (hasTensorSemantics())
    return;

  effects.emplace_back(MemoryEffects::Read::get(), &*getInputsMutable().begin(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &*getOutputsMutable().begin(),
      SideEffects::DefaultResource::get());
}

// ============================================================================
// Layout NCHWc32 to NCHW verifier.
// ============================================================================
LogicalResult LayoutNchwc32ToNchwOp::verifyInvariantsImpl(Operation *op) {
  auto layoutOp = cast<LayoutNchwc32ToNchwOp>(op);

  if (layoutOp.getInputs().size() != 1)
    return layoutOp.emitOpError("expects exactly one input");
  if (layoutOp.getOutputs().size() != 1)
    return layoutOp.emitOpError("expects exactly one output");
  if (layoutOp.getTileFactor() <= 0)
    return layoutOp.emitOpError("expects tile_factor > 0");

  Type inputType = layoutOp.getInputs().front().getType();
  Type outputType = layoutOp.getOutputs().front().getType();
  if (!isa<ShapedType>(inputType) || !isa<ShapedType>(outputType))
    return layoutOp.emitOpError("expects shaped input/output");

  auto inputShape = cast<ShapedType>(inputType);
  auto outputShape = cast<ShapedType>(outputType);
  if (inputShape.getRank() != 5)
    return layoutOp.emitOpError("expects 5D input");
  if (outputShape.getRank() != 4)
    return layoutOp.emitOpError("expects 4D output");

  bool allTensor = isa<TensorType>(inputType) && isa<TensorType>(outputType);
  bool allMemref =
      isa<BaseMemRefType>(inputType) && isa<BaseMemRefType>(outputType);

  if (layoutOp.hasTensorSemantics()) {
    if (!allTensor)
      return layoutOp.emitOpError(
          "tensor semantics require tensor input and tensor output");
    if (layoutOp.getResultTensors().size() != 1)
      return layoutOp.emitOpError(
          "tensor semantics require one tensor result");
    if (layoutOp.getResultTensors().front().getType() != outputType)
      return layoutOp.emitOpError("tensor result type must match outs type");
  } else if (!allMemref) {
    return layoutOp.emitOpError(
        "buffer semantics require memref input and memref output");
  }

  auto inputDims = inputShape.getShape();
  auto outputDims = outputShape.getShape();
  int64_t tileFactor = layoutOp.getTileFactor();
  if (inputDims[4] != ShapedType::kDynamic && inputDims[4] != tileFactor) {
    return layoutOp.emitOpError("expects input last dimension to equal tile_factor");
  }
  if (inputDims[0] != ShapedType::kDynamic && outputDims[0] != ShapedType::kDynamic &&
      inputDims[0] != outputDims[0]) {
    return layoutOp.emitOpError("expects batch dimension to be preserved");
  }
  if (inputDims[2] != ShapedType::kDynamic && outputDims[2] != ShapedType::kDynamic &&
      inputDims[2] != outputDims[2]) {
    return layoutOp.emitOpError("expects height dimension to be preserved");
  }
  if (inputDims[3] != ShapedType::kDynamic && outputDims[3] != ShapedType::kDynamic &&
      inputDims[3] != outputDims[3]) {
    return layoutOp.emitOpError("expects width dimension to be preserved");
  }
  if (inputDims[1] != ShapedType::kDynamic && outputDims[1] != ShapedType::kDynamic &&
      inputDims[4] != ShapedType::kDynamic) {
    int64_t expectedChannels = inputDims[1] * inputDims[4];
    if (outputDims[1] != expectedChannels) {
      return layoutOp.emitOpError("expects unpacked channel dimension to match tile_factor");
    }
  }
  return success();
}

LogicalResult LayoutNchwc32ToNchwOp::fold(
    FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

// ============================================================================
// Layout NCHWc32 to NCHW memory effects.
// ============================================================================
void LayoutNchwc32ToNchwOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (hasTensorSemantics())
    return;

  effects.emplace_back(MemoryEffects::Read::get(), &*getInputsMutable().begin(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &*getOutputsMutable().begin(),
      SideEffects::DefaultResource::get());
}

// ============================================================================
// ACC to SPM move verifier.
// ============================================================================
LogicalResult MvAccToSpmOp::verifyInvariantsImpl(Operation *op) {
  auto moveOp = cast<MvAccToSpmOp>(op);

  if (moveOp.getInputs().size() != 1)
    return moveOp.emitOpError("expects exactly one input");
  if (moveOp.getOutputs().size() != 1)
    return moveOp.emitOpError("expects exactly one output");

  Type inputType = moveOp.getInputs().front().getType();
  Type outputType = moveOp.getOutputs().front().getType();
  if (!isa<ShapedType>(inputType) || !isa<ShapedType>(outputType))
    return moveOp.emitOpError("expects shaped input/output");

  auto inputShape = cast<ShapedType>(inputType);
  auto outputShape = cast<ShapedType>(outputType);
  if (inputShape.getShape() != outputShape.getShape())
    return moveOp.emitOpError("expects identical input/output shapes");

  bool allTensor = isa<TensorType>(inputType) && isa<TensorType>(outputType);
  bool allMemref =
      isa<BaseMemRefType>(inputType) && isa<BaseMemRefType>(outputType);

  if (moveOp.hasTensorSemantics()) {
    if (!allTensor)
      return moveOp.emitOpError(
          "tensor semantics require tensor input and tensor output");
    if (moveOp.getResultTensors().size() != 1)
      return moveOp.emitOpError("tensor semantics require one tensor result");
    if (moveOp.getResultTensors().front().getType() != outputType)
      return moveOp.emitOpError("tensor result type must match outs type");
    return success();
  }

  if (!allMemref)
    return moveOp.emitOpError(
        "buffer semantics require memref input and memref output");
  return success();
}

LogicalResult MvAccToSpmOp::fold(
    FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

// ============================================================================
// ACC to SPM move memory effects.
// ============================================================================
void MvAccToSpmOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (hasTensorSemantics())
    return;

  effects.emplace_back(MemoryEffects::Read::get(), &*getInputsMutable().begin(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &*getOutputsMutable().begin(),
      SideEffects::DefaultResource::get());
}

// ============================================================================
// DMA mvin verifier.
// ============================================================================
LogicalResult DmaMvinOp::verifyInvariantsImpl(Operation *op) {
  auto dmaOp = cast<DmaMvinOp>(op);

  if (dmaOp.getInputs().size() != 1)
    return dmaOp.emitOpError("expects exactly one input");
  if (dmaOp.getOutputs().size() != 1)
    return dmaOp.emitOpError("expects exactly one output");

  Type inputType = dmaOp.getInputs().front().getType();
  Type outputType = dmaOp.getOutputs().front().getType();
  if (!isa<ShapedType>(inputType) || !isa<ShapedType>(outputType))
    return dmaOp.emitOpError("expects shaped input/output");

  auto inputShape = cast<ShapedType>(inputType);
  auto outputShape = cast<ShapedType>(outputType);
  if (inputShape.getShape() != outputShape.getShape())
    return dmaOp.emitOpError("expects identical input/output shapes");

  bool allTensor = isa<TensorType>(inputType) && isa<TensorType>(outputType);
  bool allMemref =
      isa<BaseMemRefType>(inputType) && isa<BaseMemRefType>(outputType);

  if (dmaOp.hasTensorSemantics()) {
    if (!allTensor)
      return dmaOp.emitOpError(
          "tensor semantics require tensor input and tensor output");
    if (dmaOp.getResultTensors().size() != 1)
      return dmaOp.emitOpError("tensor semantics require one tensor result");
    if (dmaOp.getResultTensors().front().getType() != outputType)
      return dmaOp.emitOpError("tensor result type must match outs type");
    return success();
  }

  if (!allMemref)
    return dmaOp.emitOpError(
        "buffer semantics require memref input and memref output");
  return success();
}

LogicalResult DmaMvinOp::fold(
    FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

// ============================================================================
// DMA mvin memory effects.
// ============================================================================
void DmaMvinOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (hasTensorSemantics())
    return;

  effects.emplace_back(MemoryEffects::Read::get(), &*getInputsMutable().begin(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &*getOutputsMutable().begin(),
      SideEffects::DefaultResource::get());
}

// ============================================================================
// DMA mvout verifier.
// ============================================================================
LogicalResult DmaMvoutOp::verifyInvariantsImpl(Operation *op) {
  auto dmaOp = cast<DmaMvoutOp>(op);

  if (dmaOp.getInputs().size() != 1)
    return dmaOp.emitOpError("expects exactly one input");
  if (dmaOp.getOutputs().size() != 1)
    return dmaOp.emitOpError("expects exactly one output");

  Type inputType = dmaOp.getInputs().front().getType();
  Type outputType = dmaOp.getOutputs().front().getType();
  if (!isa<ShapedType>(inputType) || !isa<ShapedType>(outputType))
    return dmaOp.emitOpError("expects shaped input/output");

  auto inputShape = cast<ShapedType>(inputType);
  auto outputShape = cast<ShapedType>(outputType);
  if (inputShape.getShape() != outputShape.getShape())
    return dmaOp.emitOpError("expects identical input/output shapes");

  bool allTensor = isa<TensorType>(inputType) && isa<TensorType>(outputType);
  bool allMemref =
      isa<BaseMemRefType>(inputType) && isa<BaseMemRefType>(outputType);

  if (dmaOp.hasTensorSemantics()) {
    if (!allTensor)
      return dmaOp.emitOpError(
          "tensor semantics require tensor input and tensor output");
    if (dmaOp.getResultTensors().size() != 1)
      return dmaOp.emitOpError("tensor semantics require one tensor result");
    if (dmaOp.getResultTensors().front().getType() != outputType)
      return dmaOp.emitOpError("tensor result type must match outs type");
    return success();
  }

  if (!allMemref)
    return dmaOp.emitOpError(
        "buffer semantics require memref input and memref output");
  return success();
}

LogicalResult DmaMvoutOp::fold(
    FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

// ============================================================================
// DMA mvout memory effects.
// ============================================================================
void DmaMvoutOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (hasTensorSemantics())
    return;

  effects.emplace_back(MemoryEffects::Read::get(), &*getInputsMutable().begin(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &*getOutputsMutable().begin(),
      SideEffects::DefaultResource::get());
}

// ============================================================================
// Conv verifier.
// ============================================================================
LogicalResult ConvOp::verifyInvariantsImpl(Operation *op) {
  auto convOp = cast<ConvOp>(op);
  auto sameOrDynamic = [](int64_t lhs, int64_t rhs) {
    return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs == rhs;
  };
  auto computeOutDim = [](int64_t input, int64_t kernel, int64_t padBefore,
                           int64_t padAfter, int64_t stride, int64_t dilation) {
    int64_t effectiveKernel = dilation * (kernel - 1) + 1;
    return ((input + padBefore + padAfter - effectiveKernel) / stride) + 1;
  };

  if (convOp.getInputs().size() != 2 && convOp.getInputs().size() != 3)
    return convOp.emitOpError("expects two or three inputs");
  if (convOp.getOutputs().size() != 1)
    return convOp.emitOpError("expects exactly one output");

  Type outputType = convOp.getOutputs().front().getType();
  if (!isa<ShapedType>(outputType))
    return convOp.emitOpError("expects shaped output");
  for (Type inputType : convOp.getInputs().getTypes()) {
    if (!isa<ShapedType>(inputType))
      return convOp.emitOpError("expects shaped inputs");
  }

  if (convOp.getPads().size() != 4)
    return convOp.emitOpError("expects exactly four pad values");
  if (convOp.getStrides().size() != 2)
    return convOp.emitOpError("expects exactly two stride values");
  if (convOp.getDilations().size() != 2)
    return convOp.emitOpError("expects exactly two dilation values");
  if (convOp.getGroup() < 1)
    return convOp.emitOpError("expects group >= 1");
  if (cast<IntegerAttr>(convOp.getStrides()[0]).getInt() < 1 ||
      cast<IntegerAttr>(convOp.getStrides()[1]).getInt() < 1)
    return convOp.emitOpError("expects strides >= 1");
  if (cast<IntegerAttr>(convOp.getDilations()[0]).getInt() < 1 ||
      cast<IntegerAttr>(convOp.getDilations()[1]).getInt() < 1)
    return convOp.emitOpError("expects dilations >= 1");

  bool allTensor = llvm::all_of(convOp.getInputs().getTypes(),
                       [](Type type) { return isa<TensorType>(type); }) &&
      isa<TensorType>(outputType);
  bool allMemref = llvm::all_of(convOp.getInputs().getTypes(),
                       [](Type type) { return isa<BaseMemRefType>(type); }) &&
      isa<BaseMemRefType>(outputType);

  if (convOp.hasTensorSemantics()) {
    if (!allTensor)
      return convOp.emitOpError(
          "tensor semantics require tensor inputs and tensor output");
    if (convOp.getResultTensors().size() != 1)
      return convOp.emitOpError(
          "tensor semantics require one tensor result");
    if (convOp.getResultTensors().front().getType() != outputType)
      return convOp.emitOpError("tensor result type must match outs type");
    return success();
  }

  if (!allMemref)
    return convOp.emitOpError(
        "buffer semantics require memref inputs and memref output");

  auto inputType = cast<ShapedType>(convOp.getInputs()[0].getType());
  auto weightType = cast<ShapedType>(convOp.getInputs()[1].getType());
  auto outType = cast<ShapedType>(outputType);
  if (inputType.getRank() != 5)
    return convOp.emitOpError("expects packed input rank to be 5");
  if (weightType.getRank() != 6)
    return convOp.emitOpError("expects packed weight rank to be 6");
  if (outType.getRank() != 5)
    return convOp.emitOpError("expects packed output rank to be 5");

  ArrayRef<int64_t> inShape = inputType.getShape();
  ArrayRef<int64_t> wShape = weightType.getShape();
  ArrayRef<int64_t> outShape = outType.getShape();

  if (!sameOrDynamic(inShape[0], outShape[0]))
    return convOp.emitOpError("expects batch dimension to be preserved");
  if (!sameOrDynamic(inShape[1], wShape[1]))
    return convOp.emitOpError(
        "expects packed input channel chunk to match weight input channel chunk");
  if (!sameOrDynamic(inShape[4], wShape[4]))
    return convOp.emitOpError(
        "expects packed input channel block to match weight input channel block");
  if (!sameOrDynamic(wShape[0], outShape[1]))
    return convOp.emitOpError(
        "expects packed output channel chunk to match weight output channel chunk");
  if (!sameOrDynamic(wShape[5], outShape[4]))
    return convOp.emitOpError(
        "expects packed output channel block to match weight output channel block");

  if (convOp.getInputs().size() == 3) {
    auto biasType = cast<ShapedType>(convOp.getInputs()[2].getType());
    if (biasType.getRank() != 2)
      return convOp.emitOpError("expects packed bias rank to be 2");
    ArrayRef<int64_t> biasShape = biasType.getShape();
    if (!sameOrDynamic(biasShape[0], outShape[1]) ||
        !sameOrDynamic(biasShape[1], outShape[4]))
      return convOp.emitOpError(
          "expects packed bias shape to match output channel chunk/block");
  }

  int64_t strideH = cast<IntegerAttr>(convOp.getStrides()[0]).getInt();
  int64_t strideW = cast<IntegerAttr>(convOp.getStrides()[1]).getInt();
  int64_t dilationH = cast<IntegerAttr>(convOp.getDilations()[0]).getInt();
  int64_t dilationW = cast<IntegerAttr>(convOp.getDilations()[1]).getInt();
  int64_t padTop = cast<IntegerAttr>(convOp.getPads()[0]).getInt();
  int64_t padLeft = cast<IntegerAttr>(convOp.getPads()[1]).getInt();
  int64_t padBottom = cast<IntegerAttr>(convOp.getPads()[2]).getInt();
  int64_t padRight = cast<IntegerAttr>(convOp.getPads()[3]).getInt();

  if (inShape[2] != ShapedType::kDynamic && wShape[2] != ShapedType::kDynamic &&
      outShape[2] != ShapedType::kDynamic) {
    int64_t expectedOh = computeOutDim(
        inShape[2], wShape[2], padTop, padBottom, strideH, dilationH);
    if (expectedOh != outShape[2])
      return convOp.emitOpError(
          "expects output height to match convolution formula");
  }
  if (inShape[3] != ShapedType::kDynamic && wShape[3] != ShapedType::kDynamic &&
      outShape[3] != ShapedType::kDynamic) {
    int64_t expectedOw = computeOutDim(
        inShape[3], wShape[3], padLeft, padRight, strideW, dilationW);
    if (expectedOw != outShape[3])
      return convOp.emitOpError(
          "expects output width to match convolution formula");
  }
  return success();
}

LogicalResult ConvOp::fold(
    FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

// ============================================================================
// Conv memory effects.
// ============================================================================
void ConvOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (hasTensorSemantics())
    return;

  for (OpOperand &inputOperand : getInputsMutable())
    effects.emplace_back(MemoryEffects::Read::get(), &inputOperand,
        SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &*getOutputsMutable().begin(),
      SideEffects::DefaultResource::get());
}

#define GET_OP_CLASSES
#include "src/Dialect/Npucore/NpucoreOps.cpp.inc"
