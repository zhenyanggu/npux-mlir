//=======================================================
// src/Conversion/NpuTiling/NpuTilingHelper.cpp
// this file contains helper functions for npu tiling patterns
//=======================================================

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/DenseMap.h"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"


#include <cmath>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;

namespace npux {

namespace {

static constexpr StringLiteral kFusionLookupAttr = "__npux.fusion_lookup_id";

static StringRef getLibraryCallName(linalg::GenericOp op) {
  auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
  return libCallAttr ? libCallAttr.getValue() : StringRef{};
}

static RankedTensorType changeEncoding(
    RankedTensorType type, int64_t encoding, OpBuilder &builder) {
  Attribute encodingAttr =
      (encoding == 0) ? Attribute() : builder.getI64IntegerAttr(encoding);
  return RankedTensorType::get(
      type.getShape(), type.getElementType(), encodingAttr);
}

struct FixedPointParams {
  int16_t multiplier;
  int16_t shift;
};

static FixedPointParams getFixedPointParams(double scale) {
  if (std::abs(scale) < 1e-8)
    return {0, 0};
  int exponent;
  double mantissa = std::frexp(scale, &exponent);
  double mantissaScaled = std::round(mantissa * 32768.0);
  if (mantissaScaled >= 32768.0) {
    mantissaScaled /= 2.0;
    exponent += 1;
  }
  return {static_cast<int16_t>(mantissaScaled),
      static_cast<int16_t>(exponent - 15)};
}

static SmallVector<Value> buildDynamicSizesForTensor(
    PatternRewriter &rewriter, Location loc, Value value,
    RankedTensorType type) {
  SmallVector<Value> dynamicSizes;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (type.isDynamicDim(i))
      dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, value, i));
  }
  return dynamicSizes;
}

} // namespace

linalg::CopyOp createDmaGenericOp(PatternRewriter &rewriter,
    Location loc, Value input, StringRef dmaName, int64_t encoding,
    Value dest) {
  auto inputType = cast<RankedTensorType>(input.getType());
  Value finalDest = dest;
  if (!finalDest) {
    auto outType = changeEncoding(inputType, encoding, rewriter);
    SmallVector<Value> dynamicSizes =
        buildDynamicSizesForTensor(rewriter, loc, input, inputType);
    finalDest = rewriter.create<bufferization::AllocTensorOp>(
        loc, outType, dynamicSizes);
  }

  auto dmaOp = rewriter.create<linalg::CopyOp>(loc, input, finalDest);
  dmaOp->setAttr("library_call", rewriter.getStringAttr(dmaName));
  dmaOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
  dmaOp->setAttr("npu.tiled", rewriter.getUnitAttr());
  return dmaOp;
}

FailureOr<Value> maybeSplitDmaOp(linalg::CopyOp dmaOp,
    const DmaTileAnalysis &dmaAnalysis, PatternRewriter &rewriter) {
  dmaOp->setAttr(
      "npu.dma_col_dim", rewriter.getI32IntegerAttr(dmaAnalysis.colDimIdx));

  bool needSplit = llvm::any_of(
      dmaAnalysis.splitSizes, [](int64_t s) { return s > 0; });
  if (!needSplit)
    return dmaOp.getResult(0);

  SmallVector<OpFoldResult> tileSizes =
      getAsIndexOpFoldResult(rewriter.getContext(), dmaAnalysis.splitSizes);
  scf::SCFTilingOptions options;
  options.setTileSizes(tileSizes);

  FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(dmaOp.getOperation()), options);
  if (failed(tilingResult))
    return failure();

  SmallVector<Value> finalResults = tilingResult->replacements;
  for (int i = (int)tilingResult->loops.size() - 1; i >= 0; --i) {
    auto loopOp = dyn_cast<scf::ForOp>(tilingResult->loops[i].getOperation());
    if (!loopOp)
      continue;
    scf::ForOp partialIteration;
    if (succeeded(scf::peelForLoopAndSimplifyBounds(
            rewriter, loopOp, partialIteration))) {
      if (i == 0)
        finalResults = partialIteration->getResults();
    }
  }

  if (finalResults.empty())
    return failure();
  return finalResults.front();
}

FailureOr<linalg::GenericOp> cloneGenericOpToMemorySpace(
    linalg::GenericOp op, int64_t memorySpace, PatternRewriter &rewriter) {
  if (!op || op.getNumDpsInits() != 1 || op->getNumResults() != 1)
    return failure();

  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);

  OpOperand *outOperand = op.getDpsInitOperand(0);
  Value outVal = outOperand->get();
  auto outType = dyn_cast<RankedTensorType>(outVal.getType());
  if (!outType)
    return failure();

  SmallVector<Value> dynamicSizes;
  for (int64_t i = 0; i < outType.getRank(); ++i) {
    if (outType.isDynamicDim(i))
      dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, outVal, i));
  }

  auto memSpaceAttr = rewriter.getI64IntegerAttr(memorySpace);
  auto newTensorType = RankedTensorType::get(
      outType.getShape(), outType.getElementType(), memSpaceAttr);
  auto alloc = rewriter.create<bufferization::AllocTensorOp>(
      loc, newTensorType, dynamicSizes);

  auto newOp = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{newTensorType}, op.getInputs(), ValueRange{alloc},
      op.getIndexingMapsArray(), op.getIteratorTypesArray());
  rewriter.inlineRegionBefore(
      op.getRegion(), newOp.getRegion(), newOp.getRegion().begin());
  newOp->setAttrs(op->getAttrs());
  rewriter.replaceOp(op, newOp.getResults());
  return newOp;
}

std::optional<DmaTileAnalysis> buildFullTensorDmaAnalysis(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape())
    return std::nullopt;

  SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());
  return analyzeDmaTileFromKnownTile(shape, shape);
}

namespace {

static linalg::GenericOp findSingleTiledGenericOp(
    Block *block, StringRef libCallName) {
  if (!block)
    return nullptr;
  for (Operation &op : *block) {
    auto genericOp = dyn_cast<linalg::GenericOp>(&op);
    if (!genericOp || !genericOp->hasAttr("npu.tiled"))
      continue;
    if (getLibraryCallName(genericOp) == libCallName)
      return genericOp;
  }
  return nullptr;
}

static linalg::GenericOp findTiledGenericOpByMarker(
    Block *block, int64_t marker) {
  if (!block)
    return nullptr;
  for (Operation &op : *block) {
    auto genericOp = dyn_cast<linalg::GenericOp>(&op);
    if (!genericOp)
      continue;
    auto markerAttr =
        genericOp->getAttrOfType<IntegerAttr>(kFusionLookupAttr);
    if (markerAttr && markerAttr.getInt() == marker)
      return genericOp;
  }
  return nullptr;
}

static SmallVector<linalg::GenericOp> collectCurrentTiledGenericOps(Block *block) {
  SmallVector<linalg::GenericOp> ops;
  if (!block)
    return ops;

  for (Operation &op : *block) {
    auto genericOp = dyn_cast<linalg::GenericOp>(&op);
    if (!genericOp || !genericOp->hasAttr("npu.tiled"))
      continue;
    ops.push_back(genericOp);
  }
  return ops;
}

static int64_t getStandaloneComputeOutputMemorySpace(linalg::GenericOp op) {
  StringRef libCall = getLibraryCallName(op);
  if (libCall == "npu_conv" || libCall == "npu_gemm" ||
      libCall == "npu_matmul")
    return 3;
  return 2;
}

static LogicalResult materializeStandaloneTiledMatAddWithDma(
    linalg::GenericOp tiledOp, PatternRewriter &rewriter) {
  if (!tiledOp || tiledOp.getNumDpsInputs() != 2 || tiledOp.getNumDpsInits() != 1 ||
      tiledOp->getNumResults() != 1)
    return failure();

  Location loc = tiledOp.getLoc();
  rewriter.setInsertionPoint(tiledOp);
  Value originalOutput = tiledOp.getDpsInitOperand(0)->get();
  auto outputType = dyn_cast<RankedTensorType>(originalOutput.getType());
  if (!outputType)
    return failure();

  auto getFloatAttrOr = [&](StringRef name, double def) {
    if (auto attr = tiledOp->getAttrOfType<FloatAttr>(name))
      return attr.getValueAsDouble();
    return def;
  };

  double in1Scale = getFloatAttrOr("in1_scale", 1.0);
  double in2Scale = getFloatAttrOr("in2_scale", 1.0);
  double outScale = getFloatAttrOr("out_scale", 1.0);
  bool scalesSame = std::abs(in1Scale - in2Scale) < 1e-6;

  Type i32Type = rewriter.getI32Type();
  SmallVector<Value> dmaInputs;
  for (auto [operandIdx, operand] : llvm::enumerate(tiledOp.getInputs())) {
    auto inputType = cast<RankedTensorType>(operand.getType());
    auto mvinOutType = RankedTensorType::get(
        inputType.getShape(), i32Type, rewriter.getI64IntegerAttr(3));
    SmallVector<Value> dynamicSizes =
        buildDynamicSizesForTensor(rewriter, loc, operand, inputType);
    Value finalDest = rewriter.create<bufferization::AllocTensorOp>(
        loc, mvinOutType, dynamicSizes);

    SmallVector<utils::IteratorType> iteratorTypes(
        inputType.getRank(), utils::IteratorType::parallel);
    SmallVector<AffineMap> maps(
        2, rewriter.getMultiDimIdentityMap(inputType.getRank()));

    auto mvinOp = rewriter.create<linalg::GenericOp>(loc, mvinOutType,
        ValueRange{operand}, ValueRange{finalDest}, maps, iteratorTypes,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value val = args[0];
          if (val.getType() != i32Type && isa<IntegerType>(val.getType()))
            val = b.create<arith::ExtSIOp>(nestedLoc, i32Type, val);
          b.create<linalg::YieldOp>(nestedLoc, val);
        });

    mvinOp->setAttr("library_call", rewriter.getStringAttr("npu_dma_mvin"));
    mvinOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    mvinOp->setAttr("npu.tiled", rewriter.getUnitAttr());

    StringRef zpAttrName = (operandIdx == 0) ? "in1_zp" : "in2_zp";
    double currentInScale = (operandIdx == 0) ? in1Scale : in2Scale;
    int64_t zpVal = 0;
    if (auto zpAttr = tiledOp->getAttrOfType<IntegerAttr>(zpAttrName))
      zpVal = zpAttr.getInt();

    bool isQuant = false;
    double mvinScaleVal = 1.0;
    if (!scalesSame) {
      mvinScaleVal = currentInScale / outScale;
      isQuant = true;
    } else if (zpVal != 0) {
      isQuant = true;
    }

    if (isQuant) {
      mvinOp->setAttr("npu.quant_zero", rewriter.getI32IntegerAttr(zpVal));
      FixedPointParams qParams = getFixedPointParams(mvinScaleVal);
      mvinOp->setAttr(
          "npu.quant_scale", rewriter.getI64IntegerAttr(qParams.multiplier));
      mvinOp->setAttr(
          "npu.quant_shift", rewriter.getI64IntegerAttr(qParams.shift));
      mvinOp->setAttr("npu.is_quant", rewriter.getBoolAttr(true));
    }

    dmaInputs.push_back(mvinOp.getResult(0));
  }

  auto mediumType = changeEncoding(outputType, 2, rewriter);
  SmallVector<Value> outputDynamicSizes =
      buildDynamicSizesForTensor(rewriter, loc, originalOutput, outputType);
  Value mediumTensor = rewriter.create<bufferization::AllocTensorOp>(
      loc, mediumType, outputDynamicSizes);

  int64_t rank = outputType.getRank();
  SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(rank));
  SmallVector<utils::IteratorType> iteratorTypes(
      rank, utils::IteratorType::parallel);

  auto newOp = rewriter.create<linalg::GenericOp>(loc, TypeRange{mediumType},
      dmaInputs, ValueRange{mediumTensor}, maps, iteratorTypes,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value lhs = args[0];
        Value rhs = args[1];
        Value addRes = b.create<arith::AddIOp>(nestedLoc, lhs, rhs);
        Value finalRes = addRes;
        if (addRes.getType() != args[2].getType() &&
            isa<IntegerType>(args[2].getType())) {
          finalRes = b.create<arith::TruncIOp>(
              nestedLoc, args[2].getType(), addRes);
        }
        b.create<linalg::YieldOp>(nestedLoc, finalRes);
      });

  for (NamedAttribute attr : tiledOp->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (!scalesSame &&
        (name == "in1_scale" || name == "in2_scale" || name == "out_scale")) {
      newOp->setAttr(name, rewriter.getF32FloatAttr(1.0f));
    } else {
      newOp->setAttr(name, attr.getValue());
    }
  }
  newOp->setAttr("npu.tiled", rewriter.getUnitAttr());

  rewriter.setInsertionPointAfter(newOp);
  auto mvoutOp = createDmaGenericOp(
      rewriter, loc, newOp.getResult(0), "npu_dma_mvout", 1, originalOutput);
  if (auto outputAnalysis = buildFullTensorDmaAnalysis(originalOutput)) {
    if (failed(maybeSplitDmaOp(mvoutOp, *outputAnalysis, rewriter)))
      return failure();
  }

  rewriter.replaceOp(tiledOp, mvoutOp.getResults());
  return success();
}

static LogicalResult materializeStandaloneTiledOpWithDma(
    linalg::GenericOp tiledOp, PatternRewriter &rewriter) {
  if (!tiledOp || tiledOp.getNumDpsInits() != 1 || tiledOp->getNumResults() != 1)
    return failure();

  if (getLibraryCallName(tiledOp) == "npu_matadd")
    return materializeStandaloneTiledMatAddWithDma(tiledOp, rewriter);

  Value externalDest = tiledOp.getDpsInitOperand(0)->get();
  auto relocatedOp = cloneGenericOpToMemorySpace(
      tiledOp, getStandaloneComputeOutputMemorySpace(tiledOp), rewriter);
  if (failed(relocatedOp))
    return failure();

  (*relocatedOp)->setAttr("npu.tiled", rewriter.getUnitAttr());

  SmallVector<Value> dmaInputs;
  dmaInputs.reserve((*relocatedOp).getInputs().size());
  for (Value operand : (*relocatedOp).getInputs()) {
    rewriter.setInsertionPoint(*relocatedOp);
    auto mvinOp = createDmaGenericOp(
        rewriter, (*relocatedOp).getLoc(), operand, "npu_dma_mvin", 2);
    FailureOr<Value> dmaResult = mvinOp.getResult(0);
    if (auto inputAnalysis = buildFullTensorDmaAnalysis(operand)) {
      dmaResult = maybeSplitDmaOp(mvinOp, *inputAnalysis, rewriter);
    }
    if (failed(dmaResult))
      return failure();
    dmaInputs.push_back(*dmaResult);
  }

  rewriter.modifyOpInPlace(*relocatedOp, [&]() {
    (*relocatedOp).getInputsMutable().assign(dmaInputs);
  });

  rewriter.setInsertionPointAfter(*relocatedOp);
  auto mvoutOp = createDmaGenericOp(rewriter, (*relocatedOp).getLoc(),
      (*relocatedOp).getResult(0), "npu_dma_mvout", 1, externalDest);
  if (auto outputAnalysis = buildFullTensorDmaAnalysis(externalDest)) {
    if (failed(maybeSplitDmaOp(mvoutOp, *outputAnalysis, rewriter)))
      return failure();
  }
  rewriter.replaceAllUsesExcept(
      (*relocatedOp).getResult(0), mvoutOp.getResult(0), mvoutOp.getOperation());
  return success();
}

} // namespace

// Derives the DMA split shape, column dimension and call count from a known
// tensor shape and the tile shape chosen for one logical op invocation.
std::optional<DmaTileAnalysis> analyzeDmaTileFromKnownTile(
    ArrayRef<int64_t> fullShape, ArrayRef<int64_t> tileShape) {
  int rank = tileShape.size();
  if ((int64_t)fullShape.size() != rank)
    return std::nullopt;

  SmallVector<int64_t> concreteTileShape;
  concreteTileShape.reserve(rank);
  for (auto [dim, tile] : llvm::zip(fullShape, tileShape)) {
    if (dim == ShapedType::kDynamic)
      return std::nullopt;
    concreteTileShape.push_back(tile > 0 ? tile : dim);
  }

  SmallVector<bool> isSplit(rank, false);
  for (int i = 0; i < rank; ++i) {
    if (concreteTileShape[i] == ShapedType::kDynamic)
      return std::nullopt;
    if (concreteTileShape[i] < fullShape[i])
      isSplit[i] = true;
  }

  int splitCount = 0;
  int idx1 = -1;
  int idx2 = -1;
  for (int i = rank - 1; i >= 0; --i) {
    if (!isSplit[i])
      continue;
    ++splitCount;
    if (splitCount == 1)
      idx1 = i;
    if (splitCount == 2) {
      idx2 = i;
      break;
    }
  }

  DmaTileAnalysis analysis;
  analysis.splitSizes.assign(rank, 0);

  constexpr int64_t kLimit = 65535;
  if (idx1 != -1) {
    int64_t trailingElements = 1;
    for (int i = idx1 + 1; i < rank; ++i)
      trailingElements *= concreteTileShape[i];

    int64_t sramStride = concreteTileShape[idx1] * trailingElements;
    if (sramStride > kLimit) {
      idx2 = idx1;
      idx1++;
    }
  } else {
    int64_t totalSize = 1;
    for (int64_t dim : concreteTileShape)
      totalSize *= dim;

    if (totalSize > kLimit) {
      int64_t acc = 1;
      for (int i = rank - 1; i >= 0; --i) {
        if (acc * concreteTileShape[i] > kLimit) {
          idx1 = i;
          if (idx1 == rank - 1)
            analysis.splitSizes[i] = kLimit;
          else
            idx1++;
          break;
        }
        acc *= concreteTileShape[i];
      }
    }
  }

  for (int i = idx2 - 1; i >= 0; --i)
    analysis.splitSizes[i] = 1;

  analysis.colDimIdx = idx1;
  analysis.callCount = 1;
  for (auto [tile, split] : llvm::zip(concreteTileShape, analysis.splitSizes)) {
    if (split > 0)
      analysis.callCount *= (tile + split - 1) / split;
  }

  return analysis;
}

LogicalResult tileStandaloneOp(
    linalg::GenericOp op, ArrayRef<int64_t> tileSizes,
    PatternRewriter &rewriter) {
  auto tilingInterfaceOp = cast<TilingInterface>(op.getOperation());
  SmallVector<OpFoldResult> tileSizesOfr =
      getAsOpFoldResult(rewriter.getI64ArrayAttr(tileSizes));
  scf::SCFTilingOptions options;
  options.setTileSizes(tileSizesOfr);

  FailureOr<scf::SCFTilingResult> tilingResult =
      scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

  for (auto loop : tilingResult->loops)
    loop->setAttr("npu.target", rewriter.getStringAttr("npu"));
  for (Operation *tiledOp : tilingResult->tiledOps)
    tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());

  SmallVector<Value> finalResults = tilingResult->replacements;
  for (int i = tilingResult->loops.size() - 1; i >= 0; --i) {
    auto loopOp = dyn_cast<scf::ForOp>(tilingResult->loops[i].getOperation());
    scf::ForOp partialIteration;
    if (succeeded(scf::peelForLoopAndSimplifyBounds(
            rewriter, loopOp, partialIteration))) {
      partialIteration->setAttr("npu.peeled_tail", rewriter.getUnitAttr());
      if (i == 0)
        finalResults = partialIteration->getResults();
    }
  }

  rewriter.replaceOp(op, finalResults);
  return success();
}

LogicalResult tileFusedChainOp(
    linalg::GenericOp seed, linalg::GenericOp root, ArrayRef<int64_t> tileSizes,
    PatternRewriter &rewriter, ArrayRef<DmaTileAnalysis> seedInputDmaAnalyses,
    const std::optional<DmaTileAnalysis> &rootOutputDmaAnalysis,
    llvm::function_ref<void(ArrayRef<Operation *>, PatternRewriter &)>
        postprocessTiledOps) {
  // A single-op fusion group is still materialized by the second pass, but it
  // should not go through producer-fusion utilities. Those helpers may inspect
  // unrelated tensor producers and require extra interfaces we don't register.
  if (seed == root) {
    auto tilingInterfaceOp = cast<TilingInterface>(root.getOperation());
    StringRef rootLibCall = getLibraryCallName(root);
    SmallVector<OpFoldResult> tileSizesOfr =
        getAsOpFoldResult(rewriter.getI64ArrayAttr(tileSizes));
    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizesOfr);

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);
    if (failed(tilingResult))
      return failure();

    for (auto loop : tilingResult->loops)
      loop->setAttr("npu.target", rewriter.getStringAttr("npu"));
    for (Operation *tiledOp : tilingResult->tiledOps)
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());

    Block *tiledBlock = nullptr;
    for (Operation *tiledOp : tilingResult->tiledOps) {
      if (auto genericOp = dyn_cast<linalg::GenericOp>(tiledOp)) {
        tiledBlock = genericOp->getBlock();
        break;
      }
    }

    if (postprocessTiledOps) {
      SmallVector<Operation *> tiledOps(
          tilingResult->tiledOps.begin(), tilingResult->tiledOps.end());
      postprocessTiledOps(tiledOps, rewriter);
    }

    if (auto tiledRoot = findSingleTiledGenericOp(tiledBlock, rootLibCall)) {
      if (failed(materializeStandaloneTiledOpWithDma(tiledRoot, rewriter)))
        return failure();
    }

    SmallVector<Value> finalResults = tilingResult->replacements;
    if (!tilingResult->loops.empty()) {
      if (auto outerLoop =
              dyn_cast<scf::ForOp>(tilingResult->loops.front().getOperation())) {
        finalResults.assign(
            outerLoop->getResults().begin(), outerLoop->getResults().end());
      }
    }
    for (int i = tilingResult->loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(tilingResult->loops[i].getOperation());
      scf::ForOp partialIteration;
      if (succeeded(scf::peelForLoopAndSimplifyBounds(
              rewriter, loopOp, partialIteration))) {
        partialIteration->setAttr("npu.peeled_tail", rewriter.getUnitAttr());
        if (i == 0)
          finalResults = partialIteration->getResults();
      }
    }

    rewriter.replaceOp(root, finalResults);
    return success();
  }

  auto rootTilingInterface = cast<TilingInterface>(root.getOperation());

  scf::SCFTileAndFuseOptions fuseOptions;
  fuseOptions.tilingOptions.setTileSizes(
      getAsOpFoldResult(rewriter.getI64ArrayAttr(tileSizes)));

  Operation *targetOpPtr = seed.getOperation();
  auto stopAfterSeed = std::make_shared<bool>(false);
  fuseOptions.setFusionControlFn(
      [targetOpPtr, stopAfterSeed](tensor::ExtractSliceOp, OpResult originalProducer, bool)
          -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
        if (*stopAfterSeed)
          return std::nullopt;
        if (originalProducer.getOwner() == targetOpPtr) {
          *stopAfterSeed = true;
          return scf::SCFTileAndFuseOptions::ControlFnResult{};
        }
        if (originalProducer.getOwner()->getBlock() == targetOpPtr->getBlock())
          return scf::SCFTileAndFuseOptions::ControlFnResult{};
        return std::nullopt;
      });

  seed->setAttr(kFusionLookupAttr, rewriter.getI64IntegerAttr(0));
  root->setAttr(kFusionLookupAttr, rewriter.getI64IntegerAttr(1));
  FailureOr<scf::SCFTileAndFuseResult> fuseResult =
      scf::tileConsumerAndFuseProducersUsingSCF(
          rewriter, rootTilingInterface, fuseOptions);
  seed->removeAttr(kFusionLookupAttr);
  root->removeAttr(kFusionLookupAttr);
  if (failed(fuseResult))
    return failure();

  Block *tiledBlock = nullptr;
  for (Operation *op : fuseResult->tiledAndFusedOps) {
    if (op && op->getBlock()) {
      tiledBlock = op->getBlock();
      break;
    }
  }

  for (Operation *op : fuseResult->tiledAndFusedOps)
    op->setAttr("npu.tiled", rewriter.getUnitAttr());
  SmallVector<Operation *> tiledAndFusedOps(
      fuseResult->tiledAndFusedOps.begin(),
      fuseResult->tiledAndFusedOps.end());
  if (postprocessTiledOps) {
    postprocessTiledOps(tiledAndFusedOps, rewriter);
  }

  auto tiledSeed = findTiledGenericOpByMarker(tiledBlock, 0);
  auto tiledRoot = findTiledGenericOpByMarker(tiledBlock, 1);
  if (!tiledSeed || !tiledRoot)
    return failure();
  tiledSeed->removeAttr(kFusionLookupAttr);
  tiledRoot->removeAttr(kFusionLookupAttr);

  Value rootExternalDest = tiledRoot.getDpsInitOperand(0)->get();

  auto relocatedRoot = cloneGenericOpToMemorySpace(tiledRoot, 2, rewriter);
  if (failed(relocatedRoot))
    return failure();
  tiledRoot = *relocatedRoot;
  tiledRoot->setAttr("npu.tiled", rewriter.getUnitAttr());

  if (tiledSeed != tiledRoot) {
    int64_t seedMemorySpace = getStandaloneComputeOutputMemorySpace(tiledSeed);
    auto relocatedSeed = cloneGenericOpToMemorySpace(
        tiledSeed, seedMemorySpace, rewriter);
    if (failed(relocatedSeed))
      return failure();
    tiledSeed = *relocatedSeed;
    tiledSeed->setAttr("npu.tiled", rewriter.getUnitAttr());
  }

  SmallVector<linalg::GenericOp> currentTiledOps =
      collectCurrentTiledGenericOps(tiledBlock);

  llvm::DenseMap<Value, Value> dmaReplacementMap;
  for (linalg::GenericOp genericOp : currentTiledOps) {

    rewriter.setInsertionPoint(genericOp);
    SmallVector<Value> updatedInputs;
    updatedInputs.reserve(genericOp.getInputs().size());
    for (auto [inputIdx, operand] : llvm::enumerate(genericOp.getInputs())) {
      auto replacementIt = dmaReplacementMap.find(operand);
      if (replacementIt != dmaReplacementMap.end()) {
        updatedInputs.push_back(replacementIt->second);
        continue;
      }

      Operation *defOp = operand.getDefiningOp();
      if (defOp && defOp->getBlock() == genericOp->getBlock()) {
        updatedInputs.push_back(operand);
        continue;
      }

      if (getLibraryCallName(genericOp) == "npu_maxpool" &&
          genericOp == tiledRoot && inputIdx == 1) {
        updatedInputs.push_back(operand);
        continue;
      }

      auto mvinOp = createDmaGenericOp(
          rewriter, genericOp.getLoc(), operand, "npu_dma_mvin", 2);
      FailureOr<Value> dmaResult = mvinOp.getResult(0);
      if (genericOp == tiledSeed && inputIdx < seedInputDmaAnalyses.size()) {
        dmaResult = maybeSplitDmaOp(
            mvinOp, seedInputDmaAnalyses[inputIdx], rewriter);
      } else if (auto inputAnalysis = buildFullTensorDmaAnalysis(operand)) {
        dmaResult = maybeSplitDmaOp(mvinOp, *inputAnalysis, rewriter);
      }
      if (failed(dmaResult))
        return failure();
      dmaReplacementMap[operand] = *dmaResult;
      updatedInputs.push_back(*dmaResult);
    }

    rewriter.modifyOpInPlace(genericOp, [&]() {
      genericOp.getInputsMutable().assign(updatedInputs);
    });
  }

  rewriter.setInsertionPointAfter(tiledRoot);
  auto mvoutOp = createDmaGenericOp(rewriter, tiledRoot.getLoc(),
      tiledRoot.getResult(0), "npu_dma_mvout", 1, rootExternalDest);
  if (rootOutputDmaAnalysis) {
    if (failed(maybeSplitDmaOp(
            mvoutOp, *rootOutputDmaAnalysis, rewriter)))
      return failure();
  } else if (auto outputAnalysis = buildFullTensorDmaAnalysis(rootExternalDest)) {
    if (failed(maybeSplitDmaOp(mvoutOp, *outputAnalysis, rewriter)))
      return failure();
  }
  rewriter.replaceAllUsesExcept(
      tiledRoot.getResult(0), mvoutOp.getResult(0), mvoutOp.getOperation());

  for (auto loop : fuseResult->loops)
    loop->setAttr("npu.target", rewriter.getStringAttr("npu"));

  SmallVector<Value> finalResults;
  for (Value result : root->getResults())
    finalResults.push_back(fuseResult->replacements.lookup(result));
  for (int i = fuseResult->loops.size() - 1; i >= 0; --i) {
    auto loopOp = dyn_cast<scf::ForOp>(fuseResult->loops[i].getOperation());
    scf::ForOp partialIteration;
    if (succeeded(scf::peelForLoopAndSimplifyBounds(
            rewriter, loopOp, partialIteration))) {
      partialIteration->setAttr("npu.peeled_tail", rewriter.getUnitAttr());
      if (i == 0)
        finalResults = partialIteration->getResults();
    }
  }

  rewriter.replaceOp(root, finalResults);
  return success();
}

void populateNpuTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  populateElemWiseTilingPatterns(patterns, context);
  populateConvTilingPatterns(patterns, context);
  populateGemmTilingPatterns(patterns, context);
  populateLayoutTilingPatterns(patterns, context);
  populateMaxPoolTilingPatterns(patterns, context);
};
} // namespace npux

LogicalResult peelForLoopLastIteration(
    RewriterBase &b, scf::ForOp forOp, scf::ForOp &lastIteration) {
  RewriterBase::InsertionGuard guard(b);
  auto lbInt = getConstantIntValue(forOp.getLowerBound());
  auto ubInt = getConstantIntValue(forOp.getUpperBound());
  auto stepInt = getConstantIntValue(forOp.getStep());

  if (lbInt && ubInt && stepInt &&
      std::ceil((double)(*ubInt - *lbInt) / *stepInt) <= 1) {
    return failure();
  }

  AffineExpr ubSymbol, stepSymbol;
  bindSymbols(b.getContext(), ubSymbol, stepSymbol);
  auto splitMap = AffineMap::get(0, 2, {ubSymbol - stepSymbol});
  b.setInsertionPoint(forOp);
  auto loc = forOp.getLoc();
  Value splitBound = b.createOrFold<affine::AffineApplyOp>(
      loc, splitMap, ValueRange{forOp.getUpperBound(), forOp.getStep()});

  IRMapping map;
  map.map(forOp.getLowerBound(), splitBound);
  b.setInsertionPointAfter(forOp);
  lastIteration = cast<scf::ForOp>(b.clone(*forOp.getOperation(), map));

  b.modifyOpInPlace(
      forOp, [&]() { forOp.getUpperBoundMutable().assign(splitBound); });

  if (forOp.getNumResults() > 0) {
    b.modifyOpInPlace(lastIteration, [&]() {
      lastIteration.getInitArgsMutable().assign(forOp.getResults());
    });
  }
  b.replaceOpUsesWithIf(forOp, lastIteration->getResults(),
      [&](OpOperand &use) { return use.getOwner() != lastIteration; });

  return success();
}
