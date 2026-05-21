//===============================================================
// src/Conversion/NpuTiling/FusionAnalysis/FusionEvaluator.cpp
// this file is for evaluator class of the profitability of 
// fusion candidates
//===============================================================

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "src/Conversion/NpuTiling/FusionAnalysis/FusionEvaluator.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

namespace mlir {

namespace {

static int64_t ceilDiv(int64_t lhs, int64_t rhs) {
  if (rhs <= 0)
    return 0;
  return (lhs + rhs - 1) / rhs;
}

static SmallVector<int64_t> getOutputShape(Operation *op) {
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
  return SmallVector<int64_t>(
      outputType.getShape().begin(), outputType.getShape().end());
}

static SmallVector<int64_t> materializeTileShape(
    ArrayRef<int64_t> fullShape, ArrayRef<int64_t> tileShape) {
  if (fullShape.size() != tileShape.size())
    return {};

  SmallVector<int64_t> result;
  result.reserve(tileShape.size());
  for (auto [dim, tile] : llvm::zip(fullShape, tileShape)) {
    if (dim <= 0)
      return {};
    result.push_back(tile > 0 ? tile : dim);
  }
  return result;
}

static std::optional<unsigned> findSharedInputIndex(
    Operation *tail, Operation *consumer) {
  Value tailResult = tail->getNumResults() > 0 ? tail->getResult(0) : Value{};
  if (!tailResult)
    return std::nullopt;

  if (auto genericConsumer = dyn_cast<linalg::GenericOp>(consumer)) {
    for (auto [idx, operand] :
         llvm::enumerate(genericConsumer.getDpsInputOperands())) {
      if (operand->get() == tailResult)
        return idx;
    }
    return std::nullopt;
  }

  if (auto packOp = dyn_cast<linalg::PackOp>(consumer))
    return packOp.getSource() == tailResult ? std::optional<unsigned>(0)
                                            : std::nullopt;
  if (auto unpackOp = dyn_cast<linalg::UnPackOp>(consumer))
    return unpackOp.getSource() == tailResult ? std::optional<unsigned>(0)
                                              : std::nullopt;
  return std::nullopt;
}

static int64_t countLogicalTiles(
    ArrayRef<int64_t> shape, ArrayRef<int64_t> tileShape) {
  if (shape.size() != tileShape.size())
    return 0;

  SmallVector<int64_t> concreteTileShape =
      materializeTileShape(shape, tileShape);
  if (concreteTileShape.empty())
    return 0;

  int64_t tiles = 1;
  for (auto [dim, tile] : llvm::zip(shape, concreteTileShape)) {
    if (dim <= 0 || tile <= 0)
      return 0;
    tiles *= ceilDiv(dim, tile);
  }
  return tiles;
}

static StringRef getLibraryCallName(Operation *op) {
  auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
  return libCallAttr ? libCallAttr.getValue() : StringRef{};
}

static void extractIntArrayAttr(
    Operation *op, StringRef attrName, int64_t &lhs, int64_t &rhs) {
  if (auto denseAttr = op->getAttrOfType<DenseI64ArrayAttr>(attrName)) {
    if (denseAttr.size() >= 2) {
      lhs = denseAttr[0];
      rhs = denseAttr[1];
    }
    return;
  }

  if (auto arrayAttr = op->getAttrOfType<ArrayAttr>(attrName)) {
    if (arrayAttr.size() >= 2) {
      lhs = cast<IntegerAttr>(arrayAttr[0]).getInt();
      rhs = cast<IntegerAttr>(arrayAttr[1]).getInt();
    }
    return;
  }
}

static SmallVector<int64_t> inferInputTileShape(
    Operation *op, unsigned inputIndex, ArrayRef<int64_t> outputTileShape) {
  StringRef libCall = getLibraryCallName(op);
  if (libCall.empty())
    return {};

  if (libCall == "npu_conv") {
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    if (!genericOp)
      return {};
    if (inputIndex >= genericOp.getNumDpsInputs() || outputTileShape.size() != 5)
      return {};

    if (inputIndex == 0) {
      auto inputType = dyn_cast<RankedTensorType>(
          genericOp.getDpsInputOperand(0)->get().getType());
      auto weightType = dyn_cast<RankedTensorType>(
          genericOp.getDpsInputOperand(1)->get().getType());
      if (!inputType || !weightType || inputType.getRank() != 5 ||
          weightType.getRank() != 6)
        return {};

      int64_t strideH = 1;
      int64_t strideW = 1;
      int64_t dilationH = 1;
      int64_t dilationW = 1;
      extractIntArrayAttr(op, "strides", strideH, strideW);
      extractIntArrayAttr(op, "dilations", dilationH, dilationW);

      int64_t kernelH = weightType.getShape()[2];
      int64_t kernelW = weightType.getShape()[3];
      int64_t effectiveKH = (kernelH - 1) * dilationH + 1;
      int64_t effectiveKW = (kernelW - 1) * dilationW + 1;
      return {outputTileShape[0], outputTileShape[4],
          (outputTileShape[2] - 1) * strideH + effectiveKH,
          (outputTileShape[3] - 1) * strideW + effectiveKW,
          inputType.getShape()[4]};
    }

    if (inputIndex == 1) {
      auto weightType = dyn_cast<RankedTensorType>(
          genericOp.getDpsInputOperand(1)->get().getType());
      if (!weightType || weightType.getRank() != 6)
        return {};
      return {outputTileShape[1], outputTileShape[4], weightType.getShape()[2],
          weightType.getShape()[3], weightType.getShape()[4],
          weightType.getShape()[5]};
    }
    return {};
  }

  if (libCall == "mv_acc_to_spm")
    return SmallVector<int64_t>(outputTileShape.begin(), outputTileShape.end());

  if (libCall == "npu_layout_nchwc32_to_nchw") {
    RankedTensorType inputType;
    if (auto unpackOp = dyn_cast<linalg::UnPackOp>(op))
      inputType = unpackOp.getSourceType();
    else if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
      inputType = dyn_cast<RankedTensorType>(
          genericOp.getDpsInputOperand(0)->get().getType());
    if (!inputType || inputType.getRank() != 5 || outputTileShape.size() != 4)
      return {};
    return {outputTileShape[0],
        ceilDiv(outputTileShape[1], inputType.getShape()[4]),
        outputTileShape[2], outputTileShape[3], inputType.getShape()[4]};
  }

  if (libCall == "npu_layout_nchw_to_nchwc32") {
    RankedTensorType inputType;
    if (auto packOp = dyn_cast<linalg::PackOp>(op))
      inputType = packOp.getSourceType();
    else if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
      inputType = dyn_cast<RankedTensorType>(
          genericOp.getDpsInputOperand(0)->get().getType());
    if (!inputType || inputType.getRank() != 4 || outputTileShape.size() != 5)
      return {};
    return {outputTileShape[0], outputTileShape[1] * outputTileShape[4],
        outputTileShape[2], outputTileShape[3]};
  }

  if (libCall == "npu_maxpool") {
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    if (!genericOp)
      return {};
    auto inputType = dyn_cast<RankedTensorType>(
        genericOp.getDpsInputOperand(0)->get().getType());
    auto windowType = dyn_cast<RankedTensorType>(
        genericOp.getDpsInputOperand(1)->get().getType());
    if (!inputType || !windowType || inputType.getRank() != 4 ||
        windowType.getRank() != 2 || outputTileShape.size() != 4)
      return {};

    int64_t strideH = 2;
    int64_t strideW = 2;
    extractIntArrayAttr(op, "strides", strideH, strideW);
    int64_t kernelH = windowType.getShape()[0];
    int64_t kernelW = windowType.getShape()[1];
    return {outputTileShape[0], outputTileShape[1],
        (outputTileShape[2] - 1) * strideH + kernelH,
        (outputTileShape[3] - 1) * strideW + kernelW};
  }

  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp || inputIndex >= genericOp.getNumDpsInputs())
    return {};
  auto inputType = dyn_cast<RankedTensorType>(
      genericOp.getDpsInputOperand(inputIndex)->get().getType());
  if (!inputType || inputType.getRank() != (int64_t)outputTileShape.size())
    return {};
  return SmallVector<int64_t>(outputTileShape.begin(), outputTileShape.end());
}

static SmallVector<int64_t> getStandaloneOutputTileShape(
    Operation *op, ArrayRef<int64_t> producerTileShape) {
  StringRef libCall = getLibraryCallName(op);
  if (libCall.empty())
    return {};

  if (libCall == "mv_acc_to_spm")
    return SmallVector<int64_t>(producerTileShape.begin(), producerTileShape.end());

  if (libCall == "npu_layout_nchw_to_nchwc32") {
    SmallVector<int64_t> outputShape = getOutputShape(op);
    if (producerTileShape.size() != 4 || outputShape.size() != 5)
      return {};
    int64_t innerC = outputShape[4];
    int64_t channelTile = producerTileShape[1];
    return {producerTileShape[0],
        std::min<int64_t>(outputShape[1],
            std::max<int64_t>(1, ceilDiv(channelTile, innerC))),
        producerTileShape[2], producerTileShape[3],
        std::min<int64_t>(innerC, channelTile)};
  }

  if (libCall == "npu_layout_nchwc32_to_nchw") {
    SmallVector<int64_t> outputShape = getOutputShape(op);
    if (producerTileShape.size() != 5 || outputShape.size() != 4)
      return {};
    return {producerTileShape[0],
        std::min<int64_t>(outputShape[1],
            producerTileShape[1] * producerTileShape[4]),
        producerTileShape[2], producerTileShape[3]};
  }

  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp)
    return {};
  SmallVector<int64_t> tileSizes = costModel.getOptimalTileSizes(genericOp);
  SmallVector<int64_t> outputShape = getOutputShape(op);
  if (tileSizes.empty() || outputShape.empty())
    return {};

  if (tileSizes.size() == outputShape.size())
    return tileSizes;

  if (libCall == "npu_maxpool" &&
      tileSizes.size() == 6 && outputShape.size() == 4) {
    return {tileSizes[0] > 0 ? tileSizes[0] : outputShape[0],
        tileSizes[1] > 0 ? tileSizes[1] : outputShape[1],
        tileSizes[2] > 0 ? tileSizes[2] : outputShape[2],
        tileSizes[3] > 0 ? tileSizes[3] : outputShape[3]};
  }
  return {};
}

static double calculateDmaCostForTile(
    Operation *op, ArrayRef<int64_t> outputTileSizes, double dmaTime) {
  SmallVector<int64_t> outputShape = getOutputShape(op);
  if (outputShape.empty() || outputShape.size() != outputTileSizes.size())
    return 0.0;

  int64_t opCalls = countLogicalTiles(outputShape, outputTileSizes);
  if (opCalls <= 0)
    return 0.0;

  SmallVector<RankedTensorType> inputTypes;
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    for (unsigned inputIndex = 0; inputIndex < genericOp.getNumDpsInputs();
         ++inputIndex) {
      inputTypes.push_back(dyn_cast<RankedTensorType>(
          genericOp.getDpsInputOperand(inputIndex)->get().getType()));
    }
  } else if (auto packOp = dyn_cast<linalg::PackOp>(op)) {
    inputTypes.push_back(packOp.getSourceType());
  } else if (auto unpackOp = dyn_cast<linalg::UnPackOp>(op)) {
    inputTypes.push_back(unpackOp.getSourceType());
  } else {
    return 0.0;
  }

  double totalDmaCalls = 0.0;
  for (auto [inputIndex, inputType] : llvm::enumerate(inputTypes)) {
    if (!inputType || !inputType.hasStaticShape())
      return 0.0;
    SmallVector<int64_t> inputTileShape =
        inferInputTileShape(op, inputIndex, outputTileSizes);
    if (inputTileShape.empty())
      return 0.0;
    auto dmaAnalysis =
        npux::analyzeDmaTileFromKnownTile(inputType.getShape(), inputTileShape);
    if (!dmaAnalysis)
      return 0.0;
    totalDmaCalls += dmaAnalysis->callCount;
  }

  auto outputDmaAnalysis =
      npux::analyzeDmaTileFromKnownTile(outputShape, outputTileSizes);
  if (!outputDmaAnalysis)
    return 0.0;
  totalDmaCalls += outputDmaAnalysis->callCount;
  return static_cast<double>(opCalls) * totalDmaCalls * dmaTime;
}

static double calculateSharedBoundaryCostForFusedTile(Operation *tail,
    Operation *consumer, ArrayRef<int64_t> fusedTailTileSizes,
    ArrayRef<int64_t> fusedConsumerTileSizes, double dmaTime) {
  std::optional<unsigned> sharedInputIdx = findSharedInputIndex(tail, consumer);
  if (!sharedInputIdx)
    return 0.0;

  SmallVector<int64_t> tailOutputShape = getOutputShape(tail);
  if (tailOutputShape.empty() ||
      tailOutputShape.size() != fusedTailTileSizes.size())
    return 0.0;

  auto tailOutputDmaAnalysis =
      npux::analyzeDmaTileFromKnownTile(tailOutputShape, fusedTailTileSizes);
  if (!tailOutputDmaAnalysis)
    return 0.0;

  int64_t tailOpCalls = countLogicalTiles(tailOutputShape, fusedTailTileSizes);
  if (tailOpCalls <= 0)
    return 0.0;

  RankedTensorType consumerInputType;
  if (auto genericConsumer = dyn_cast<linalg::GenericOp>(consumer)) {
    consumerInputType = dyn_cast<RankedTensorType>(
        genericConsumer.getDpsInputOperand(*sharedInputIdx)->get().getType());
  } else if (auto packOp = dyn_cast<linalg::PackOp>(consumer)) {
    consumerInputType = packOp.getSourceType();
  } else if (auto unpackOp = dyn_cast<linalg::UnPackOp>(consumer)) {
    consumerInputType = unpackOp.getSourceType();
  }
  SmallVector<int64_t> consumerInputTileShape = inferInputTileShape(
      consumer, *sharedInputIdx, fusedConsumerTileSizes);
  if (!consumerInputType || !consumerInputType.hasStaticShape() ||
      consumerInputTileShape.empty())
    return 0.0;

  auto consumerInputDmaAnalysis = npux::analyzeDmaTileFromKnownTile(
      consumerInputType.getShape(), consumerInputTileShape);
  if (!consumerInputDmaAnalysis)
    return 0.0;

  SmallVector<int64_t> consumerOutputShape = getOutputShape(consumer);
  int64_t consumerOpCalls =
      countLogicalTiles(consumerOutputShape, fusedConsumerTileSizes);
  if (consumerOpCalls <= 0)
    return 0.0;

  return static_cast<double>(tailOpCalls * tailOutputDmaAnalysis->callCount +
             consumerOpCalls * consumerInputDmaAnalysis->callCount) *
      dmaTime;
}

} // namespace

FusionCostEvaluator::FusionCostEvaluator(
    double cpuWaitIrqTime, double dmaCallTime)
    : waitIrqTime(cpuWaitIrqTime), dmaTime(dmaCallTime) {}

double FusionCostEvaluator::evaluateFusionBenefit(
    Operation *tail, Operation *consumer,
    llvm::ArrayRef<int64_t> currentTailTileSizes,
    llvm::ArrayRef<int64_t> fusedConsumerTileSizes) {
  SmallVector<int64_t> fusedTailTileSizes =
      inferInputTileShape(consumer, 0, fusedConsumerTileSizes);
  if (fusedTailTileSizes.empty())
    return 0.0;

  SmallVector<int64_t> standaloneConsumerTileSizes =
      getStandaloneOutputTileShape(consumer, currentTailTileSizes);
  if (standaloneConsumerTileSizes.empty())
    return 0.0;

  double unfusedCost =
      calculateOpCostWithTile(tail, currentTailTileSizes) +
      calculateOpCostWithTile(consumer, standaloneConsumerTileSizes);
  double fusedCost =
      calculateOpCostWithTile(tail, fusedTailTileSizes) +
      calculateOpCostWithTile(consumer, fusedConsumerTileSizes);
  double sharedBoundaryCost = calculateSharedBoundaryCostForFusedTile(
      tail, consumer, fusedTailTileSizes, fusedConsumerTileSizes, dmaTime);
  return unfusedCost - fusedCost + sharedBoundaryCost;
}

double FusionCostEvaluator::calculateOpCostWithTile(
    Operation *op, llvm::ArrayRef<int64_t> outputTileSizes) {
  SmallVector<int64_t> outputShape = getOutputShape(op);
  if (outputShape.empty() || outputShape.size() != outputTileSizes.size())
    return 0.0;

  int64_t irqCalls = countLogicalTiles(outputShape, outputTileSizes);
  if (irqCalls <= 0)
    return 0.0;

  return static_cast<double>(irqCalls) * waitIrqTime +
         calculateDmaCostForTile(op, outputTileSizes, dmaTime);
}

} // namespace mlir
