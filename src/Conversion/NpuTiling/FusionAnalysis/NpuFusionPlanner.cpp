//=============================================================================
// Shared planning utilities for the two-stage NPU fusion tiling pipeline.
//=============================================================================

#include "src/Conversion/NpuTiling/FusionAnalysis/NpuFusionPlanner.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ErrorHandling.h"
#include <limits>

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "src/Conversion/NpuTiling/Strategy/Conv.hpp"

using namespace mlir;

namespace {

constexpr StringLiteral kSeedTileSizesAttr = "npu.seed_tile_sizes";
constexpr StringLiteral kTailTileSizesAttr = "npu.tail_tile_sizes";
constexpr StringLiteral kSeedInputDmaAttr = "npu.seed_input_dma";
constexpr StringLiteral kRootOutputDmaAttr = "npu.root_output_dma";
constexpr StringLiteral kDmaSplitSizesAttr = "split_sizes";
constexpr StringLiteral kDmaColDimIdxAttr = "col_dim_idx";
constexpr StringLiteral kDmaCallCountAttr = "call_count";

int64_t ceilDiv(int64_t lhs, int64_t rhs) {
  if (rhs <= 0)
    return 0;
  return (lhs + rhs - 1) / rhs;
}

bool isSupportedFusionOp(Operation *op) {
  return isa<linalg::GenericOp, linalg::PackOp, linalg::UnPackOp>(op);
}

StringRef getLibraryCallName(Operation *op) {
  auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
  return libCallAttr ? libCallAttr.getValue() : StringRef{};
}

bool isNpuOp(Operation *op) {
  StringRef libCall = getLibraryCallName(op);
  return libCall.starts_with("npu_") || libCall.starts_with("mv_");
}

bool hasCostModelTile(Operation *op) {
  StringRef libCall = getLibraryCallName(op);
  return libCall == "npu_conv" || libCall == "npu_gemm" ||
         libCall == "npu_matmul" || libCall == "npu_gelu" ||
         libCall == "mv_acc_to_spm" || libCall == "npu_matadd" ||
         libCall == "npu_layout_nchw_to_nchwc32" ||
         libCall == "npu_layout_nchwc32_to_nchw" ||
         libCall == "npu_transpose" || libCall == "npu_maxpool" ||
         libCall == "npu_softmax" || libCall == "npu_layernorm";
}

bool isFullShapeOnlyOp(Operation *op) {
  StringRef libCall = getLibraryCallName(op);
  return libCall == "npu_softmax" || libCall == "npu_layernorm";
}

bool isFusionBarrier(Operation *op) {
  StringRef libCall = getLibraryCallName(op);
  return libCall == "npu_matadd" || libCall == "npu_conv" ||
         libCall == "npu_gemm" || libCall == "npu_matmul";
}

bool mustFuseConsumer(Operation *seed, Operation *consumer) {
  StringRef seedLibCall = getLibraryCallName(seed);
  return (seedLibCall == "npu_conv" || seedLibCall == "npu_gemm" ||
             seedLibCall == "npu_matmul") &&
         getLibraryCallName(consumer) == "mv_acc_to_spm";
}

bool isEligibleConsumer(Operation *seed, Operation *consumer) {
  if (!consumer || !isSupportedFusionOp(consumer) || !isNpuOp(consumer))
    return false;
  if (consumer->hasAttr("npu.tiled"))
    return false;
  if (isFusionBarrier(consumer))
    return false;
  return seed->getBlock() == consumer->getBlock();
}

SmallVector<Operation *> getDirectConsumers(Operation *op) {
  SmallVector<Operation *> consumers;
  llvm::SmallPtrSet<Operation *, 4> seen;
  for (OpResult result : op->getResults()) {
    if (!result.hasOneUse())
      continue;
    Operation *user = *result.getUsers().begin();
    if (isSupportedFusionOp(user) && seen.insert(user).second)
      consumers.push_back(user);
  }
  return consumers;
}

SmallVector<int64_t> getOutputShape(Operation *op) {
  RankedTensorType outputType;
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    if (genericOp.getNumDpsInits() == 0)
      return {};
    outputType = dyn_cast<RankedTensorType>(
        genericOp.getDpsInitOperand(0)->get().getType());
  } else if (auto packOp = dyn_cast<linalg::PackOp>(op)) {
    outputType = packOp.getDestType();
  } else if (auto unpackOp = dyn_cast<linalg::UnPackOp>(op)) {
    outputType = unpackOp.getDestType();
  } else {
    return {};
  }
  if (!outputType || !outputType.hasStaticShape())
    return {};
  return SmallVector<int64_t>(outputType.getShape().begin(),
      outputType.getShape().end());
}

SmallVector<int64_t> getDefaultOutputTile(Operation *op) {
  SmallVector<int64_t> shape = getOutputShape(op);
  for (int64_t &dim : shape) {
    if (dim < 0)
      dim = 1;
  }
  return shape;
}

SmallVector<int64_t> initializeSeedTileSizes(Operation *seed) {
  StringRef libCall = getLibraryCallName(seed);
  if (libCall == "npu_layout_nchw_to_nchwc32") {
    SmallVector<int64_t> inputShape;
    RankedTensorType inputType;
    if (auto genericOp = dyn_cast<linalg::GenericOp>(seed)) {
      if (genericOp.getNumDpsInputs() == 0)
        return {};
      inputType = dyn_cast<RankedTensorType>(
          genericOp.getDpsInputOperand(0)->get().getType());
    } else if (auto packOp = dyn_cast<linalg::PackOp>(seed)) {
      inputType = packOp.getSourceType();
    } else {
      return {};
    }
    if (!inputType || !inputType.hasStaticShape())
      return {};
    inputShape.assign(inputType.getShape().begin(), inputType.getShape().end());
    if (inputShape.size() == 5)
      inputShape.pop_back();
    return inputShape;
  }

  if (libCall == "npu_layout_nchwc32_to_nchw")
    return getOutputShape(seed);

  if (hasCostModelTile(seed)) {
    auto genericOp = dyn_cast<linalg::GenericOp>(seed);
    if (!genericOp)
      return getDefaultOutputTile(seed);
    npux::HardwareConfig hwConfig;
    npux::NPUCostModel costModel(hwConfig);
    SmallVector<int64_t> tileSizes = costModel.getOptimalTileSizes(genericOp);
    if (!tileSizes.empty())
      return tileSizes;
  }

  SmallVector<int64_t> outputShape = getOutputShape(seed);
  if (outputShape.empty())
    return {};
  return getDefaultOutputTile(seed);
}

SmallVector<int64_t> inferTileShapeFromIndexingMap(
    AffineMap map, ArrayRef<int64_t> loopTileSizes) {
  SmallVector<int64_t> tileShape;
  tileShape.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return {};
    unsigned pos = dimExpr.getPosition();
    if (pos >= loopTileSizes.size())
      return {};
    tileShape.push_back(loopTileSizes[pos]);
  }
  return tileShape;
}

SmallVector<int64_t> inferInputTileShapeForConsumer(
    Operation *consumer, unsigned inputIndex,
    ArrayRef<int64_t> outputTileShape) {
  StringRef libCall = getLibraryCallName(consumer);
  if (libCall == "npu_layout_nchwc32_to_nchw") {
    RankedTensorType inputType;
    if (auto genericOp = dyn_cast<linalg::GenericOp>(consumer)) {
      inputType = dyn_cast<RankedTensorType>(
          genericOp.getDpsInputOperand(inputIndex)->get().getType());
    } else if (auto unpackOp = dyn_cast<linalg::UnPackOp>(consumer)) {
      inputType = unpackOp.getSourceType();
    }
    if (!inputType || inputType.getRank() != 5 || outputTileShape.size() != 4)
      return {};
    return {outputTileShape[0],
        ceilDiv(outputTileShape[1], inputType.getShape()[4]),
        outputTileShape[2], outputTileShape[3], inputType.getShape()[4]};
  }

  if (libCall == "npu_layout_nchw_to_nchwc32") {
    RankedTensorType inputType;
    if (auto genericOp = dyn_cast<linalg::GenericOp>(consumer)) {
      inputType = dyn_cast<RankedTensorType>(
          genericOp.getDpsInputOperand(inputIndex)->get().getType());
    } else if (auto packOp = dyn_cast<linalg::PackOp>(consumer)) {
      inputType = packOp.getSourceType();
    }
    if (!inputType || inputType.getRank() != 4 || outputTileShape.size() != 5)
      return {};
    return {outputTileShape[0], outputTileShape[1] * outputTileShape[4],
        outputTileShape[2], outputTileShape[3]};
  }

  auto genericConsumer = dyn_cast<linalg::GenericOp>(consumer);
  if (!genericConsumer)
    return {};
  ArrayRef<AffineMap> indexingMaps = genericConsumer.getIndexingMapsArray();
  if (inputIndex >= indexingMaps.size())
    return {};
  return inferTileShapeFromIndexingMap(indexingMaps[inputIndex], outputTileShape);
}

bool hasReductionIterator(Operation *op) {
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
    return genericOp.getNumReductionLoops() > 0;
  return false;
}

AffineMap getPrimaryResultIndexingMap(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp)
    return AffineMap();
  ArrayRef<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
  unsigned resultMapIdx = genericOp.getNumDpsInputs();
  if (resultMapIdx >= indexingMaps.size())
    return AffineMap();
  return indexingMaps[resultMapIdx];
}

SmallVector<int64_t> initializeRootTileSizes(
    Operation *seed, ArrayRef<int64_t> seedTileSizes) {
  if (!hasReductionIterator(seed)) {
    StringRef libCall = getLibraryCallName(seed);
    if (libCall == "npu_layout_nchw_to_nchwc32") {
      SmallVector<int64_t> outputShape = getOutputShape(seed);
      if (outputShape.size() != 5 || seedTileSizes.size() != 4)
        return {};
      int64_t innerC = outputShape[4];
      return {seedTileSizes[0],
          std::min<int64_t>(outputShape[1],
              std::max<int64_t>(1, ceilDiv(seedTileSizes[1], innerC))),
          seedTileSizes[2], seedTileSizes[3],
          std::min<int64_t>(innerC, seedTileSizes[1])};
    }
    return SmallVector<int64_t>(seedTileSizes.begin(), seedTileSizes.end());
  }

  StringRef libCall = getLibraryCallName(seed);
  if (libCall == "npu_conv") {
    auto genericSeed = cast<linalg::GenericOp>(seed);
    auto outputType = dyn_cast<RankedTensorType>(
        genericSeed.getDpsInitOperand(0)->get().getType());
    if (!outputType || !outputType.hasStaticShape() || outputType.getRank() != 5 ||
        seedTileSizes.size() < 5)
      return {};

    // Conv cost-model tiles encode the reduction-side IC blocking in the last
    // entry, while the fused chain after conv consumes the real output tile.
    // Convert the strategy tile [N, OC_outer, OH, OW, IC_outer] into the
    // concrete output tile [N, OC_outer, OH, OW, OC_inner].
    return {seedTileSizes[0], seedTileSizes[1], seedTileSizes[2],
        seedTileSizes[3], outputType.getShape()[4]};
  }

  if (AffineMap resultMap = getPrimaryResultIndexingMap(seed)) {
    SmallVector<int64_t> projectedOutputTile =
        inferTileShapeFromIndexingMap(resultMap, seedTileSizes);
    if (!projectedOutputTile.empty())
      return projectedOutputTile;
  }

  return SmallVector<int64_t>(seedTileSizes.begin(), seedTileSizes.end());
}
LogicalResult populateRootOutputDmaAnalysis(
    npux::FusionCursor &cursor, ArrayRef<int64_t> rootOutputTileShape) {
  auto tail = cursor.tail;
  auto seed = cursor.seed;
  if (tail == seed)
    return success();

  SmallVector<int64_t> rootOutputShape = getOutputShape(tail);
  if (rootOutputShape.empty() || rootOutputTileShape.empty())
    return failure();

  cursor.rootOutputDmaAnalysis = npux::analyzeDmaTileFromKnownTile(
      rootOutputShape, rootOutputTileShape);
  return cursor.rootOutputDmaAnalysis ? success() : failure();
}

LogicalResult analyzeNonReductionSeedDma(
    Operation *seed, npux::FusionCursor &cursor) {
  StringRef libCall = getLibraryCallName(seed);
  auto genericSeed = dyn_cast<linalg::GenericOp>(seed);
  ArrayRef<AffineMap> indexingMaps =
      genericSeed ? genericSeed.getIndexingMapsArray() : ArrayRef<AffineMap>{};
  if (libCall != "npu_layout_nchw_to_nchwc32" &&
      libCall != "npu_layout_nchwc32_to_nchw" &&
      (!genericSeed || indexingMaps.size() < genericSeed.getNumDpsInputs()))
    return success();

  cursor.seedInputDmaAnalyses.clear();
  SmallVector<RankedTensorType> inputTypes;
  if (genericSeed) {
    for (OpOperand *operand : genericSeed.getDpsInputOperands())
      inputTypes.push_back(dyn_cast<RankedTensorType>(operand->get().getType()));
  } else if (auto packOp = dyn_cast<linalg::PackOp>(seed)) {
    inputTypes.push_back(packOp.getSourceType());
  } else if (auto unpackOp = dyn_cast<linalg::UnPackOp>(seed)) {
    inputTypes.push_back(unpackOp.getSourceType());
  }

  for (auto [idx, inputType] : llvm::enumerate(inputTypes)) {
    if (!inputType || !inputType.hasStaticShape())
      return success();

    SmallVector<int64_t> inputTileShape;
    if (libCall == "npu_layout_nchw_to_nchwc32") {
      inputTileShape = cursor.seedTileSizes;
    } else if (libCall == "npu_layout_nchwc32_to_nchw") {
      inputTileShape = inferInputTileShapeForConsumer(
          seed, idx, cursor.seedTileSizes);
    } else {
      inputTileShape = inferTileShapeFromIndexingMap(
          indexingMaps[idx], cursor.seedTileSizes);
    }
    if (inputTileShape.empty())
      return success();

    auto inputDmaAnalysis = npux::analyzeDmaTileFromKnownTile(
        inputType.getShape(), inputTileShape);
    if (!inputDmaAnalysis)
      return success();
    cursor.seedInputDmaAnalyses.push_back(*inputDmaAnalysis);
  }

  SmallVector<int64_t> rootOutputTileShape(
      cursor.tailTileSizes.begin(), cursor.tailTileSizes.end());
  (void)populateRootOutputDmaAnalysis(cursor, rootOutputTileShape);
  return success();
}

LogicalResult analyzeReductionSeedDma(
    Operation *seed, npux::FusionCursor &cursor) {
  StringRef seedLibCall = getLibraryCallName(seed);
  auto genericSeed = cast<linalg::GenericOp>(seed);

  if (seedLibCall == "npu_conv") {
    SmallVector<int64_t> activationTileShape =
        npux::inferConvActivationTileShape(genericSeed, cursor.seedTileSizes);
    SmallVector<int64_t> weightTileShape =
        npux::inferConvWeightTileShape(genericSeed, cursor.seedTileSizes);
    if (activationTileShape.empty() || weightTileShape.empty())
      return failure();

    auto inputType0 =
        dyn_cast<RankedTensorType>(genericSeed.getDpsInputOperand(0)->get().getType());
    auto inputType1 =
        dyn_cast<RankedTensorType>(genericSeed.getDpsInputOperand(1)->get().getType());
    if (!inputType0 || !inputType1 || !inputType0.hasStaticShape() ||
        !inputType1.hasStaticShape())
      return failure();

    auto activationDmaAnalysis = npux::analyzeDmaTileFromKnownTile(
        inputType0.getShape(), activationTileShape);
    auto weightDmaAnalysis = npux::analyzeDmaTileFromKnownTile(
        inputType1.getShape(), weightTileShape);
    if (!activationDmaAnalysis || !weightDmaAnalysis)
      return failure();

    cursor.seedInputDmaAnalyses = {
        *activationDmaAnalysis, *weightDmaAnalysis};

    SmallVector<int64_t> rootOutputTileShape(
        cursor.tailTileSizes.begin(), cursor.tailTileSizes.end());
    return populateRootOutputDmaAnalysis(cursor, rootOutputTileShape);
  }

  if (seedLibCall == "npu_gemm" || seedLibCall == "npu_matmul") {
    auto inputType0 =
        dyn_cast<RankedTensorType>(genericSeed.getDpsInputOperand(0)->get().getType());
    auto inputType1 =
        dyn_cast<RankedTensorType>(genericSeed.getDpsInputOperand(1)->get().getType());
    if (!inputType0 || !inputType1 || !inputType0.hasStaticShape() ||
        !inputType1.hasStaticShape())
      return failure();

    ArrayRef<AffineMap> indexingMaps = genericSeed.getIndexingMapsArray();
    if (indexingMaps.size() < 2)
      return failure();

    SmallVector<int64_t> lhsTileShape =
        inferTileShapeFromIndexingMap(indexingMaps[0], cursor.seedTileSizes);
    SmallVector<int64_t> rhsTileShape =
        inferTileShapeFromIndexingMap(indexingMaps[1], cursor.seedTileSizes);
    if (lhsTileShape.empty() || rhsTileShape.empty())
      return failure();

    auto lhsDmaAnalysis = npux::analyzeDmaTileFromKnownTile(
        inputType0.getShape(), lhsTileShape);
    auto rhsDmaAnalysis = npux::analyzeDmaTileFromKnownTile(
        inputType1.getShape(), rhsTileShape);
    if (!lhsDmaAnalysis || !rhsDmaAnalysis)
      return failure();

    cursor.seedInputDmaAnalyses = {*lhsDmaAnalysis, *rhsDmaAnalysis};

    SmallVector<int64_t> rootOutputTileShape(
        cursor.tailTileSizes.begin(), cursor.tailTileSizes.end());
    return populateRootOutputDmaAnalysis(cursor, rootOutputTileShape);
  }

  if (seedLibCall == "npu_maxpool") {
    auto inputType0 =
        dyn_cast<RankedTensorType>(genericSeed.getDpsInputOperand(0)->get().getType());
    if (!inputType0 || !inputType0.hasStaticShape())
      return failure();

    SmallVector<int64_t> outputTileShape(cursor.tailTileSizes.begin(),
        cursor.tailTileSizes.begin() +
            std::min<size_t>(cursor.tailTileSizes.size(), 4));
    if (outputTileShape.size() != 4)
      return success();

    // Keep DMA analysis aligned with the current pooling legalization contract:
    // the supported lowered form is 2x2 / stride=2 even if the attr is omitted.
    int64_t strideH = 2;
    int64_t strideW = 2;
    if (auto attr = genericSeed->getAttrOfType<ArrayAttr>("strides")) {
      if (attr.size() >= 2) {
        strideH = cast<IntegerAttr>(attr[0]).getInt();
        strideW = cast<IntegerAttr>(attr[1]).getInt();
      }
    }

    int64_t kernelH = 1;
    int64_t kernelW = 1;
    auto windowType =
        dyn_cast<RankedTensorType>(genericSeed.getDpsInputOperand(1)->get().getType());
    if (windowType && windowType.hasStaticShape() && windowType.getRank() == 2) {
      kernelH = windowType.getShape()[0];
      kernelW = windowType.getShape()[1];
    }

    SmallVector<int64_t> inputTileShape = {
        outputTileShape[0],
        outputTileShape[1],
        (outputTileShape[2] - 1) * strideH + kernelH,
        (outputTileShape[3] - 1) * strideW + kernelW};

    auto inputDmaAnalysis = npux::analyzeDmaTileFromKnownTile(
        inputType0.getShape(), inputTileShape);
    if (!inputDmaAnalysis)
      return success();
    cursor.seedInputDmaAnalyses = {*inputDmaAnalysis};

    SmallVector<int64_t> rootOutputTileShape(
        cursor.tailTileSizes.begin(), cursor.tailTileSizes.end());
    (void)populateRootOutputDmaAnalysis(cursor, rootOutputTileShape);
    return success();
  }

  return analyzeNonReductionSeedDma(seed, cursor);
}

LogicalResult updateMaxPoolTileSizes(
    Operation *consumer, SmallVector<int64_t> &tileSizes) {
  auto genericConsumer = cast<linalg::GenericOp>(consumer);
  SmallVector<int64_t> outputShape = getOutputShape(consumer);
  if (tileSizes.size() != 4 || outputShape.size() != 4)
    return failure();

  if (genericConsumer->getNumOperands() < 2)
    return failure();
  auto windowType =
      dyn_cast<RankedTensorType>(genericConsumer->getOperand(1).getType());
  if (!windowType || !windowType.hasStaticShape() || windowType.getRank() != 2)
    return failure();

  // Current NPU partitioning only legalizes the special 2x2 / stride=2 pool.
  // Some lowered maxpool ops do not carry the explicit strides attribute, so
  // keep the planner aligned with that legalization contract by defaulting to 2.
  int64_t strideH = 2;
  int64_t strideW = 2;
  if (auto stridesAttr = genericConsumer->getAttrOfType<ArrayAttr>("strides")) {
    if (stridesAttr.size() >= 2) {
      strideH = cast<IntegerAttr>(stridesAttr[0]).getInt();
      strideW = cast<IntegerAttr>(stridesAttr[1]).getInt();
    }
  } else if (auto denseStridesAttr =
                 genericConsumer->getAttrOfType<DenseI64ArrayAttr>("strides")) {
    if (denseStridesAttr.size() >= 2) {
      strideH = denseStridesAttr[0];
      strideW = denseStridesAttr[1];
    }
  }

  int64_t kH = windowType.getShape()[0];
  int64_t kW = windowType.getShape()[1];
  int64_t requiredInH = tileSizes[2];
  int64_t requiredInW = tileSizes[3];

  if (requiredInH < kH || requiredInW < kW)
    return failure();
  if ((requiredInH - kH) % strideH != 0 ||
      (requiredInW - kW) % strideW != 0)
    return failure();

  int64_t outTileH = (requiredInH - kH) / strideH + 1;
  int64_t outTileW = (requiredInW - kW) / strideW + 1;
  if (outTileH <= 0 || outTileW <= 0)
    return failure();

  tileSizes[0] = std::min<int64_t>(tileSizes[0], outputShape[0]);
  tileSizes[1] = std::min<int64_t>(tileSizes[1], outputShape[1]);
  tileSizes[2] = std::min<int64_t>(outTileH, outputShape[2]);
  tileSizes[3] = std::min<int64_t>(outTileW, outputShape[3]);
  return success();
}

LogicalResult updateLayoutTileSizes(
    Operation *consumer, SmallVector<int64_t> &tileSizes) {
  StringRef libCall = getLibraryCallName(consumer);
  SmallVector<int64_t> outputShape = getOutputShape(consumer);
  if (outputShape.empty())
    return failure();

  if (libCall == "npu_layout_nchw_to_nchwc32") {
    if (tileSizes.size() != 4 || outputShape.size() != 5)
      return failure();
    int64_t innerC = outputShape[4];
    int64_t channelTile = tileSizes[1];
    tileSizes = {std::min<int64_t>(tileSizes[0], outputShape[0]),
        std::min<int64_t>(outputShape[1],
            std::max<int64_t>(1, ceilDiv(channelTile, innerC))),
        std::min<int64_t>(tileSizes[2], outputShape[2]),
        std::min<int64_t>(tileSizes[3], outputShape[3]),
        std::min<int64_t>(innerC, channelTile)};
    return success();
  }

  if (libCall == "npu_layout_nchwc32_to_nchw") {
    if (tileSizes.size() != 5 || outputShape.size() != 4)
      return failure();
    int64_t channelTile = tileSizes[1] * tileSizes[4];
    tileSizes = {std::min<int64_t>(tileSizes[0], outputShape[0]),
        std::min<int64_t>(outputShape[1], channelTile),
        std::min<int64_t>(tileSizes[2], outputShape[2]),
        std::min<int64_t>(tileSizes[3], outputShape[3])};
    return success();
  }

  return failure();
}

LogicalResult updateTileSizesForConsumer(
    Operation *consumer, SmallVector<int64_t> &tileSizes) {
  StringRef libCall = getLibraryCallName(consumer);
  SmallVector<int64_t> outputShape = getOutputShape(consumer);

  if (isFullShapeOnlyOp(consumer)) {
    if (outputShape.empty() || outputShape.size() != tileSizes.size())
      return failure();
    SmallVector<int64_t> currentTileShape(tileSizes.begin(), tileSizes.end());
    if (currentTileShape != outputShape)
      return failure();
    tileSizes.assign(outputShape.begin(), outputShape.end());
    return success();
  }

  if (libCall == "npu_maxpool")
    return updateMaxPoolTileSizes(consumer, tileSizes);

  if (libCall == "npu_layout_nchw_to_nchwc32" ||
      libCall == "npu_layout_nchwc32_to_nchw")
    return updateLayoutTileSizes(consumer, tileSizes);

  if (outputShape.empty() || outputShape.size() != tileSizes.size())
    return failure();

  for (auto [idx, dim] : llvm::enumerate(outputShape))
    tileSizes[idx] = std::min<int64_t>(tileSizes[idx], dim);
  return success();
}

Operation *chooseBestConsumer(Operation *seed, Operation *tail,
    ArrayRef<int64_t> currentTileSizes, FusionCostEvaluator &evaluator,
    double &bestBenefit) {
  Operation *bestConsumer = nullptr;
  bestBenefit = 0.0;

  for (Operation *consumer : getDirectConsumers(tail)) {
    if (!isEligibleConsumer(seed, consumer))
      continue;

    if (mustFuseConsumer(seed, consumer)) {
      bestBenefit = std::numeric_limits<double>::infinity();
      return consumer;
    }

    SmallVector<int64_t> candidateTailTile(
        currentTileSizes.begin(), currentTileSizes.end());
    if (failed(updateTileSizesForConsumer(consumer, candidateTailTile)))
      continue;

    if (isFullShapeOnlyOp(tail)) {
      SmallVector<int64_t> fullTailShape = getOutputShape(tail);
      SmallVector<int64_t> fusedTailTile =
          inferInputTileShapeForConsumer(consumer, 0, candidateTailTile);
      if (fullTailShape.empty() || fusedTailTile.empty() ||
          fullTailShape != fusedTailTile)
        continue;
    }

    double benefit = evaluator.evaluateFusionBenefit(
        tail, consumer, currentTileSizes, candidateTailTile);
    if (benefit > bestBenefit) {
      bestBenefit = benefit;
      bestConsumer = consumer;
    }
  }

  return bestConsumer;
}

DictionaryAttr encodeDmaTileAnalysis(
    const npux::DmaTileAnalysis &analysis, Builder &builder) {
  NamedAttrList attrs;
  attrs.set(kDmaSplitSizesAttr,
      builder.getDenseI64ArrayAttr(analysis.splitSizes));
  attrs.set(kDmaColDimIdxAttr, builder.getI64IntegerAttr(analysis.colDimIdx));
  attrs.set(kDmaCallCountAttr, builder.getI64IntegerAttr(analysis.callCount));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<npux::DmaTileAnalysis> decodeDmaTileAnalysis(Attribute attr) {
  auto dictAttr = dyn_cast_or_null<DictionaryAttr>(attr);
  if (!dictAttr)
    return failure();

  auto splitSizesAttr =
      dictAttr.getAs<DenseI64ArrayAttr>(kDmaSplitSizesAttr);
  auto colDimIdxAttr = dictAttr.getAs<IntegerAttr>(kDmaColDimIdxAttr);
  auto callCountAttr = dictAttr.getAs<IntegerAttr>(kDmaCallCountAttr);
  if (!splitSizesAttr || !colDimIdxAttr || !callCountAttr)
    return failure();

  npux::DmaTileAnalysis analysis;
  analysis.splitSizes.assign(splitSizesAttr.asArrayRef().begin(),
      splitSizesAttr.asArrayRef().end());
  analysis.colDimIdx = colDimIdxAttr.getInt();
  analysis.callCount = callCountAttr.getInt();
  return analysis;
}

} // namespace

bool npux::isCandidateSeedOp(Operation *op) {
  return isSupportedFusionOp(op) && isNpuOp(op) && !op->hasAttr("npu.tiled");
}

FailureOr<npux::FusionCursor> npux::buildFusionCursorFromSeed(
    Operation *seed, FusionCostEvaluator &evaluator) {
  SmallVector<int64_t> seedTileSizes = initializeSeedTileSizes(seed);
  if (seedTileSizes.empty())
    return failure();

  npux::FusionCursor cursor;
  cursor.seed = seed;
  cursor.tail = seed;
  cursor.chainOps.push_back(seed);
  cursor.seedTileSizes = seedTileSizes;
  cursor.tailTileSizes = initializeRootTileSizes(seed, seedTileSizes);

  Operation *tail = seed;
  while (true) {
    double stepBenefit = 0.0;
    Operation *consumer = chooseBestConsumer(
        seed, tail, cursor.tailTileSizes, evaluator, stepBenefit);
    if (!consumer || stepBenefit <= 0.0)
      break;

    if (failed(updateTileSizesForConsumer(consumer, cursor.tailTileSizes)))
      break;

    tail = consumer;
    cursor.tail = consumer;
    cursor.chainOps.push_back(consumer);
  }

  if (hasReductionIterator(seed)) {
    if (failed(analyzeReductionSeedDma(seed, cursor)))
      return failure();
  } else {
    if (failed(analyzeNonReductionSeedDma(seed, cursor)))
      return failure();
  }

  return cursor;
}

void npux::serializeFusionCursorToGroup(
    const FusionCursor &cursor, npux::FusionGroupOp group, Builder &builder) {
  group->setAttr(kSeedTileSizesAttr,
      builder.getDenseI64ArrayAttr(cursor.seedTileSizes));
  group->setAttr(kTailTileSizesAttr,
      builder.getDenseI64ArrayAttr(cursor.tailTileSizes));

  SmallVector<Attribute> dmaAttrs;
  dmaAttrs.reserve(cursor.seedInputDmaAnalyses.size());
  for (const DmaTileAnalysis &analysis : cursor.seedInputDmaAnalyses)
    dmaAttrs.push_back(encodeDmaTileAnalysis(analysis, builder));
  group->setAttr(kSeedInputDmaAttr, builder.getArrayAttr(dmaAttrs));

  if (cursor.rootOutputDmaAnalysis) {
    group->setAttr(kRootOutputDmaAttr,
        encodeDmaTileAnalysis(*cursor.rootOutputDmaAnalysis, builder));
  }
}

FailureOr<npux::FusionCursor> npux::deserializeFusionCursorFromGroup(
    npux::FusionGroupOp group) {
  auto seedTileAttr = group->getAttrOfType<DenseI64ArrayAttr>(kSeedTileSizesAttr);
  auto tailTileAttr = group->getAttrOfType<DenseI64ArrayAttr>(kTailTileSizesAttr);
  auto seedInputDmaAttr = group->getAttrOfType<ArrayAttr>(kSeedInputDmaAttr);
  if (!seedTileAttr || !tailTileAttr || !seedInputDmaAttr)
    return failure();

  npux::FusionCursor cursor;
  for (Operation &op : group.getBodyRegion().front()) {
    if (isa<npux::GroupYieldOp>(op))
      continue;
    if (!isSupportedFusionOp(&op))
      continue;
    if (!cursor.seed)
      cursor.seed = &op;
    cursor.tail = &op;
    cursor.chainOps.push_back(&op);
  }

  if (!cursor.seed || !cursor.tail)
    return failure();

  cursor.seedTileSizes.assign(seedTileAttr.asArrayRef().begin(),
      seedTileAttr.asArrayRef().end());
  cursor.tailTileSizes.assign(tailTileAttr.asArrayRef().begin(),
      tailTileAttr.asArrayRef().end());

  cursor.seedInputDmaAnalyses.reserve(seedInputDmaAttr.size());
  for (Attribute attr : seedInputDmaAttr) {
    FailureOr<DmaTileAnalysis> analysis = decodeDmaTileAnalysis(attr);
    if (failed(analysis))
      return failure();
    cursor.seedInputDmaAnalyses.push_back(*analysis);
  }

  if (Attribute attr = group->getAttr(kRootOutputDmaAttr)) {
    FailureOr<DmaTileAnalysis> analysis = decodeDmaTileAnalysis(attr);
    if (failed(analysis))
      return failure();
    cursor.rootOutputDmaAnalysis = *analysis;
  }

  return cursor;
}
