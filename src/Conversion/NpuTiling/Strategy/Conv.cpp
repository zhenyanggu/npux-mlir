#include "src/Conversion/NpuTiling/Strategy/Conv.hpp"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/DenseMap.h"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

#include <cmath>
#include <memory>

using namespace mlir;

namespace {

static const SmallVector<StringRef> kDimensionLabels = {
    "N", "OC", "OH", "OW", "IC"};
static constexpr StringLiteral kFusionLookupAttr = "__npux.fusion_lookup_id";

static bool isSupportedFusionOp(Operation *op) {
  return isa<linalg::GenericOp, linalg::PackOp, linalg::UnPackOp>(op);
}

static StringRef getLibraryCallName(Operation *op) {
  auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
  return libCallAttr ? libCallAttr.getValue() : StringRef{};
}

static void tagInnerComputeOp(
    Operation *containerOp, StringRef phase, RewriterBase &rewriter) {
  containerOp->walk([&](Operation *op) {
    if (op->hasAttr("npu.tiled")) {
      op->setAttr("npu.loop_stage", rewriter.getStringAttr(phase));
    }
  });
}

static int64_t getStaticTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());

  if (lb && ub && step) {
    return (int64_t)std::ceil((double)(*ub - *lb) / *step);
  }
  return -1;
}

static void inheritNpuAttributes(scf::ForOp source, scf::ForOp target) {
  if (!source || !target)
    return;
  if (auto attr = source->getAttr("npu.target"))
    target->setAttr("npu.target", attr);
  if (auto attr = source->getAttr("npu.loop_dim"))
    target->setAttr("npu.loop_dim", attr);
}

static SmallVector<Operation *> collectCurrentTiledOps(Block *block) {
  SmallVector<Operation *> ops;
  if (!block)
    return ops;
  for (Operation &op : *block) {
    if (!isSupportedFusionOp(&op) || !op.hasAttr("npu.tiled"))
      continue;
    ops.push_back(&op);
  }
  return ops;
}

static Operation *findTiledOpByMarker(
    scf::SCFTileAndFuseResult &fuseResult, int64_t marker) {
  for (Operation *op : fuseResult.tiledAndFusedOps) {
    if (!isSupportedFusionOp(op))
      continue;
    auto markerAttr = op->getAttrOfType<IntegerAttr>(kFusionLookupAttr);
    if (markerAttr && markerAttr.getInt() == marker)
      return op;
  }
  return nullptr;
}

static SmallVector<Value> getDpsInputs(Operation *op) {
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
    return SmallVector<Value>(
        genericOp.getDpsInputs().begin(), genericOp.getDpsInputs().end());
  if (auto packOp = dyn_cast<linalg::PackOp>(op))
    return {packOp.getSource()};
  if (auto unpackOp = dyn_cast<linalg::UnPackOp>(op))
    return {unpackOp.getSource()};
  return {};
}

static Value getDpsInitValue(Operation *op) {
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
    return genericOp.getDpsInitOperand(0)->get();
  if (auto packOp = dyn_cast<linalg::PackOp>(op))
    return packOp.getDest();
  if (auto unpackOp = dyn_cast<linalg::UnPackOp>(op))
    return unpackOp.getDest();
  return nullptr;
}

static void setDpsInputs(Operation *op, ArrayRef<Value> newInputs) {
  for (auto [idx, newValue] : llvm::enumerate(newInputs))
    op->setOperand(idx, newValue);
}

struct PreparedConvTiling {
  linalg::GenericOp originalConv;
  linalg::GenericOp activeConv;
  Operation *activeRoot = nullptr;
  SmallVector<LoopLikeOpInterface> outerLoops;
  SmallVector<Value> outerReplacements;
  SmallVector<Operation *> fusedOps;
  Operation *consumerOp = nullptr;
  Value consumerResult;
  Value consumerReplacement;
  Value rootExternalDest;
  SmallVector<int64_t> seedTileSizes;
  SmallVector<int64_t> tailTileSizes;
  SmallVector<npux::DmaTileAnalysis> seedInputDmaAnalyses;
  std::optional<npux::DmaTileAnalysis> rootOutputDmaAnalysis;
};

// Maps the conv output tile back to the activation tile that must be moved in.
static SmallVector<int64_t> inferConvActivationTileShapeImpl(
    linalg::GenericOp convOp, ArrayRef<int64_t> seedTileSizes) {
  auto inputType = dyn_cast<RankedTensorType>(
      convOp.getDpsInputOperand(0)->get().getType());
  if (!inputType || inputType.getRank() != 5 || seedTileSizes.size() < 5)
    return {};

  int64_t strideH = 1;
  int64_t strideW = 1;
  int64_t dilationH = 1;
  int64_t dilationW = 1;
  if (auto attr = convOp->getAttrOfType<DenseI64ArrayAttr>("strides")) {
    if (attr.size() >= 2) {
      strideH = attr[0];
      strideW = attr[1];
    }
  } else if (auto attr = convOp->getAttrOfType<ArrayAttr>("strides")) {
    if (attr.size() >= 2) {
      strideH = cast<IntegerAttr>(attr[0]).getInt();
      strideW = cast<IntegerAttr>(attr[1]).getInt();
    }
  }
  if (auto attr = convOp->getAttrOfType<DenseI64ArrayAttr>("dilations")) {
    if (attr.size() >= 2) {
      dilationH = attr[0];
      dilationW = attr[1];
    }
  } else if (auto attr = convOp->getAttrOfType<ArrayAttr>("dilations")) {
    if (attr.size() >= 2) {
      dilationH = cast<IntegerAttr>(attr[0]).getInt();
      dilationW = cast<IntegerAttr>(attr[1]).getInt();
    }
  }

  auto weightType = dyn_cast<RankedTensorType>(
      convOp.getDpsInputOperand(1)->get().getType());
  if (!weightType || weightType.getRank() != 6)
    return {};
  int64_t kernelH = weightType.getShape()[2];
  int64_t kernelW = weightType.getShape()[3];
  int64_t effectiveKH = (kernelH - 1) * dilationH + 1;
  int64_t effectiveKW = (kernelW - 1) * dilationW + 1;

  SmallVector<int64_t> tileShape = {
      seedTileSizes[0],
      seedTileSizes[4],
      (seedTileSizes[2] - 1) * strideH + effectiveKH,
      (seedTileSizes[3] - 1) * strideW + effectiveKW,
      inputType.getShape()[4]};
  return tileShape;
}

// Builds the weight tile shape consumed by one conv output tile.
static SmallVector<int64_t> inferConvWeightTileShapeImpl(
    linalg::GenericOp convOp, ArrayRef<int64_t> seedTileSizes) {
  auto weightType = dyn_cast<RankedTensorType>(
      convOp.getDpsInputOperand(1)->get().getType());
  if (!weightType || weightType.getRank() != 6 || seedTileSizes.size() < 5)
    return {};

  SmallVector<int64_t> tileShape = {
      seedTileSizes[1],
      seedTileSizes[4],
      weightType.getShape()[2],
      weightType.getShape()[3],
      weightType.getShape()[4],
      weightType.getShape()[5]};
  return tileShape;
}

static FailureOr<scf::SCFTileAndFuseResult> fuseConvToConsumer(
    Operation *consumerOp, linalg::GenericOp convOp,
    ArrayRef<int64_t> spatialTileSizes, PatternRewriter &rewriter) {
  auto consumerTilingInterface = cast<TilingInterface>(consumerOp);
  scf::SCFTileAndFuseOptions fuseOptions;
  fuseOptions.tilingOptions.setTileSizes(
      getAsOpFoldResult(rewriter.getI64ArrayAttr(spatialTileSizes)));

  Operation *targetOpPtr = convOp.getOperation();
  auto stopAfterSeed = std::make_shared<bool>(false);
  fuseOptions.setFusionControlFn(
      [targetOpPtr, stopAfterSeed](tensor::ExtractSliceOp candidateSliceOp,
          OpResult originalProducer, bool isDestinationOperand)
          -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
        if (*stopAfterSeed)
          return std::nullopt;

        if (originalProducer.getOwner() == targetOpPtr) {
          *stopAfterSeed = true;
          return scf::SCFTileAndFuseOptions::ControlFnResult{};
        }

        if (originalProducer.getOwner()->getBlock() ==
            targetOpPtr->getBlock())
          return scf::SCFTileAndFuseOptions::ControlFnResult{};

        return std::nullopt;
      });

  return scf::tileConsumerAndFuseProducersUsingSCF(
      rewriter, consumerTilingInterface, fuseOptions);
}

static void labelOuterLoops(ArrayRef<LoopLikeOpInterface> loops,
    PatternRewriter &rewriter) {
  for (size_t i = 0; i < std::min(loops.size(), (size_t)4); ++i) {
    loops[i]->setAttr(
        "npu.loop_dim", rewriter.getStringAttr(kDimensionLabels[i]));
    loops[i]->setAttr("npu.target", rewriter.getStringAttr("npu"));
  }
}

static FailureOr<PreparedConvTiling> prepareFusedConv(
    linalg::GenericOp seed, Operation *consumerOp,
    ArrayRef<int64_t> spatialTileSizes, ArrayRef<int64_t> seedTileSizes,
    ArrayRef<int64_t> tailTileSizes,
    ArrayRef<npux::DmaTileAnalysis> seedInputDmaAnalyses,
    const std::optional<npux::DmaTileAnalysis> &rootOutputDmaAnalysis,
    PatternRewriter &rewriter) {
  seed->setAttr(kFusionLookupAttr, rewriter.getI64IntegerAttr(0));
  consumerOp->setAttr(kFusionLookupAttr, rewriter.getI64IntegerAttr(1));
  FailureOr<scf::SCFTileAndFuseResult> fuseResult =
      fuseConvToConsumer(consumerOp, seed, spatialTileSizes, rewriter);
  seed->removeAttr(kFusionLookupAttr);
  consumerOp->removeAttr(kFusionLookupAttr);
  if (failed(fuseResult))
    return failure();

  for (Operation *op : fuseResult->tiledAndFusedOps) {
    op->setAttr("npu.tiled", rewriter.getUnitAttr());
  }

  auto fusedConvOp = dyn_cast_or_null<linalg::GenericOp>(
      findTiledOpByMarker(*fuseResult, 0));
  if (!fusedConvOp)
    return failure();
  Operation *fusedRootOp = findTiledOpByMarker(*fuseResult, 1);
  if (!fusedRootOp)
    return failure();
  fusedConvOp->removeAttr(kFusionLookupAttr);
  fusedRootOp->removeAttr(kFusionLookupAttr);

  PreparedConvTiling prepared;
  prepared.originalConv = seed;
  prepared.activeConv = fusedConvOp;
  prepared.activeRoot = fusedRootOp;
  prepared.outerLoops.assign(fuseResult->loops.begin(), fuseResult->loops.end());
  prepared.fusedOps.assign(
      fuseResult->tiledAndFusedOps.begin(), fuseResult->tiledAndFusedOps.end());
  prepared.consumerOp = consumerOp;
  prepared.consumerResult = consumerOp->getNumResults() > 0
      ? consumerOp->getResult(0)
      : Value{};
  prepared.rootExternalDest = getDpsInitValue(fusedRootOp);
  prepared.seedTileSizes =
      SmallVector<int64_t>(seedTileSizes.begin(), seedTileSizes.end());
  prepared.seedInputDmaAnalyses.assign(
      seedInputDmaAnalyses.begin(), seedInputDmaAnalyses.end());
  if (prepared.consumerResult &&
      fuseResult->replacements.count(prepared.consumerResult)) {
    prepared.consumerReplacement =
        fuseResult->replacements.lookup(prepared.consumerResult);
  }
  prepared.tailTileSizes =
      SmallVector<int64_t>(tailTileSizes.begin(), tailTileSizes.end());
  prepared.rootOutputDmaAnalysis = rootOutputDmaAnalysis;
  return prepared;
}

static FailureOr<PreparedConvTiling> prepareStandaloneConv(
    linalg::GenericOp seed, ArrayRef<int64_t> tileSizes,
    ArrayRef<int64_t> seedTileSizes, ArrayRef<int64_t> tailTileSizes,
    ArrayRef<npux::DmaTileAnalysis> seedInputDmaAnalyses,
    PatternRewriter &rewriter) {
  SmallVector<OpFoldResult> tileSizesOfr =
      getAsOpFoldResult(rewriter.getI64ArrayAttr(tileSizes));
  scf::SCFTilingOptions options;
  options.setTileSizes(tileSizesOfr);

  FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(seed.getOperation()), options);
  if (failed(tilingResult))
    return failure();

  for (Operation *tiledOp : tilingResult->tiledOps) {
    tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
  }

  linalg::GenericOp tiledConv = nullptr;
  for (Operation *tiledOp : tilingResult->tiledOps) {
    tiledConv = dyn_cast<linalg::GenericOp>(tiledOp);
    if (tiledConv)
      break;
  }
  if (!tiledConv)
    return failure();

  PreparedConvTiling prepared;
  prepared.originalConv = seed;
  prepared.activeConv = tiledConv;
  prepared.activeRoot = tiledConv.getOperation();
  prepared.rootExternalDest = tiledConv.getDpsInitOperand(0)->get();
  prepared.seedTileSizes =
      SmallVector<int64_t>(seedTileSizes.begin(), seedTileSizes.end());
  prepared.seedInputDmaAnalyses.assign(
      seedInputDmaAnalyses.begin(), seedInputDmaAnalyses.end());
  prepared.tailTileSizes =
      SmallVector<int64_t>(tailTileSizes.begin(), tailTileSizes.end());
  prepared.outerLoops.assign(tilingResult->loops.begin(), tilingResult->loops.end());
  prepared.outerReplacements.assign(
      tilingResult->replacements.begin(), tilingResult->replacements.end());
  return prepared;
}

static LogicalResult finalizePreparedConv(PreparedConvTiling &prepared,
    ArrayRef<int64_t> icTileSizes, PatternRewriter &rewriter) {
  labelOuterLoops(prepared.outerLoops, rewriter);

  if (!prepared.fusedOps.empty() && prepared.consumerOp) {
    for (Operation *op : prepared.fusedOps) {
      if (!isSupportedFusionOp(op) || op == prepared.activeConv.getOperation())
        continue;

      int64_t memorySpace = 2;
      auto relocatedOp =
          npux::cloneLinalgOpToMemorySpace(op, memorySpace, rewriter);
      if (failed(relocatedOp))
        return failure();
      (*relocatedOp)->setAttr("npu.tiled", rewriter.getUnitAttr());
      if (op == prepared.activeRoot)
        prepared.activeRoot = *relocatedOp;
      if (op == prepared.consumerOp)
        prepared.consumerOp = *relocatedOp;
    }
  }

  auto relocatedConv =
      npux::cloneGenericOpToMemorySpace(prepared.activeConv, 3, rewriter);
  if (failed(relocatedConv))
    return failure();

  auto fusedConvOp = *relocatedConv;
  prepared.activeConv = fusedConvOp;
  fusedConvOp->setAttr("npu.tiled", rewriter.getUnitAttr());

  Block *tiledBlock =
      prepared.activeRoot ? prepared.activeRoot->getBlock() : fusedConvOp->getBlock();
  SmallVector<Operation *> currentTiledOps = collectCurrentTiledOps(tiledBlock);

  llvm::DenseMap<Value, Value> dmaReplacementMap;
  auto rewriteOperandWithDma = [&](Operation *op, unsigned inputIdx, Value operand)
      -> FailureOr<Value> {
    auto replacementIt = dmaReplacementMap.find(operand);
    if (replacementIt != dmaReplacementMap.end())
      return replacementIt->second;

    Operation *defOp = operand.getDefiningOp();
    if (defOp && defOp->getBlock() == op->getBlock())
      return operand;

    if (op == fusedConvOp.getOperation() && inputIdx >= 2)
      return operand;

    if (getLibraryCallName(op) == "npu_maxpool" &&
        op == prepared.activeRoot && inputIdx == 1)
      return operand;

    auto mvinOp = npux::createDmaGenericOp(
        rewriter, op->getLoc(), operand, "npu_dma_mvin", 2);
    FailureOr<Value> dmaResult = mvinOp.getResult(0);
    if (op == fusedConvOp.getOperation() &&
        inputIdx < prepared.seedInputDmaAnalyses.size()) {
      dmaResult = npux::maybeSplitDmaOp(
          mvinOp, prepared.seedInputDmaAnalyses[inputIdx], rewriter);
    } else if (auto inputAnalysis = npux::buildFullTensorDmaAnalysis(operand)) {
      dmaResult = npux::maybeSplitDmaOp(mvinOp, *inputAnalysis, rewriter);
    }
    if (failed(dmaResult))
      return failure();

    dmaReplacementMap[operand] = *dmaResult;
    return *dmaResult;
  };

  for (Operation *op : currentTiledOps) {
    rewriter.setInsertionPoint(op);

    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      SmallVector<Value> updatedInputs;
      updatedInputs.reserve(genericOp.getInputs().size());

      for (auto [inputIdx, operand] : llvm::enumerate(genericOp.getInputs())) {
        FailureOr<Value> rewritten = rewriteOperandWithDma(op, inputIdx, operand);
        if (failed(rewritten))
          return failure();
        updatedInputs.push_back(*rewritten);
      }

      rewriter.modifyOpInPlace(genericOp, [&]() {
        genericOp.getInputsMutable().assign(updatedInputs);
      });
      continue;
    }

    if (auto packOp = dyn_cast<linalg::PackOp>(op)) {
      FailureOr<Value> rewritten = rewriteOperandWithDma(op, 0, packOp.getSource());
      if (failed(rewritten))
        return failure();
      rewriter.modifyOpInPlace(packOp, [&]() {
        packOp->setOperand(0, *rewritten);
      });
      continue;
    }

    if (auto unpackOp = dyn_cast<linalg::UnPackOp>(op)) {
      FailureOr<Value> rewritten =
          rewriteOperandWithDma(op, 0, unpackOp.getSource());
      if (failed(rewritten))
        return failure();
      rewriter.modifyOpInPlace(unpackOp, [&]() {
        unpackOp->setOperand(0, *rewritten);
      });
      continue;
    }

    SmallVector<Value> updatedInputs;
    SmallVector<Value> inputs = getDpsInputs(op);
    updatedInputs.reserve(inputs.size());
    for (auto [inputIdx, operand] : llvm::enumerate(inputs)) {
      auto replacementIt = dmaReplacementMap.find(operand);
      if (replacementIt != dmaReplacementMap.end()) {
        updatedInputs.push_back(replacementIt->second);
        continue;
      }

      Operation *defOp = operand.getDefiningOp();
      if (defOp && defOp->getBlock() == op->getBlock()) {
        updatedInputs.push_back(operand);
        continue;
      }

      // Keep parity with the legacy Conv path: only activation/weight use
      // mvin on the seed conv itself; bias-like operands stay external.
      if (op == fusedConvOp.getOperation() && inputIdx >= 2) {
        updatedInputs.push_back(operand);
        continue;
      }

      // MaxPool's window tensor is not materialized with mvin in the legacy
      // DMA pass; it is treated as a special in-SRAM auxiliary operand.
      if (getLibraryCallName(op) == "npu_maxpool" &&
          op == prepared.activeRoot && inputIdx == 1) {
        updatedInputs.push_back(operand);
        continue;
      }

      auto mvinOp = npux::createDmaGenericOp(
          rewriter, op->getLoc(), operand, "npu_dma_mvin", 2);
      FailureOr<Value> dmaResult = mvinOp.getResult(0);
      if (op == fusedConvOp.getOperation() &&
          inputIdx < prepared.seedInputDmaAnalyses.size()) {
        dmaResult = npux::maybeSplitDmaOp(
            mvinOp, prepared.seedInputDmaAnalyses[inputIdx], rewriter);
      } else if (auto inputAnalysis = npux::buildFullTensorDmaAnalysis(operand)) {
        dmaResult = npux::maybeSplitDmaOp(mvinOp, *inputAnalysis, rewriter);
      }
      if (failed(dmaResult))
        return failure();

      dmaReplacementMap[operand] = *dmaResult;
      updatedInputs.push_back(*dmaResult);
    }

    rewriter.modifyOpInPlace(op, [&]() {
      setDpsInputs(op, updatedInputs);
    });
  }

  if (prepared.consumerOp && prepared.activeRoot && prepared.rootExternalDest) {
    rewriter.setInsertionPointAfter(prepared.activeRoot);
    auto mvoutOp = npux::createDmaGenericOp(rewriter, prepared.activeRoot->getLoc(),
        prepared.activeRoot->getResult(0), "npu_dma_mvout", 1,
        prepared.rootExternalDest);
    rewriter.replaceAllUsesExcept(
        prepared.activeRoot->getResult(0), mvoutOp.getResult(0),
        mvoutOp.getOperation());
    if (!prepared.rootOutputDmaAnalysis)
      return failure();
    if (failed(npux::maybeSplitDmaOp(
            mvoutOp, *prepared.rootOutputDmaAnalysis, rewriter)))
      return failure();
    prepared.consumerReplacement = mvoutOp.getResult(0);
  }

  scf::SCFTilingOptions icOptions;
  icOptions.setTileSizes(
      getAsOpFoldResult(rewriter.getI64ArrayAttr(icTileSizes)));

  FailureOr<scf::SCFTilingResult> icTilingResult = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(fusedConvOp.getOperation()), icOptions);
  if (failed(icTilingResult))
    return failure();

  if (!icTilingResult->loops.empty()) {
    icTilingResult->loops.front()->setAttr(
        "npu.loop_dim", rewriter.getStringAttr("IC"));
  }

  auto icLoops = icTilingResult->loops;
  SmallVector<Value> finalIcResults = icTilingResult->replacements;

  if (!icLoops.empty()) {
    auto loopOp = cast<scf::ForOp>(icLoops.back().getOperation());
    loopOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

    int64_t tripCount = getStaticTripCount(loopOp);
    if (tripCount == 1) {
      tagInnerComputeOp(loopOp, "single", rewriter);
    } else {
      scf::ForOp restLoop = loopOp;
      scf::ForOp headLoop;
      if (succeeded(peelForLoopFirstIteration(rewriter, loopOp, headLoop))) {
        inheritNpuAttributes(restLoop, headLoop);
        tagInnerComputeOp(headLoop, "head", rewriter);
      }

      int64_t restTripCount = getStaticTripCount(restLoop);
      if (restTripCount == 1) {
        tagInnerComputeOp(restLoop, "tail", rewriter);
        finalIcResults = restLoop->getResults();
      } else {
        scf::ForOp tailLoop;
        bool hasTail = false;
        if (succeeded(scf::peelForLoopAndSimplifyBounds(
                rewriter, restLoop, tailLoop))) {
          hasTail = true;
        } else if (succeeded(peelForLoopLastIteration(
                       rewriter, restLoop, tailLoop))) {
          hasTail = true;
        }

        if (hasTail) {
          inheritNpuAttributes(restLoop, tailLoop);
          tagInnerComputeOp(tailLoop, "tail", rewriter);
          tagInnerComputeOp(restLoop, "body", rewriter);
          finalIcResults = tailLoop->getResults();
        } else {
          tagInnerComputeOp(restLoop, "body", rewriter);
          finalIcResults = restLoop->getResults();
        }
      }
    }
  }

  rewriter.replaceOp(fusedConvOp, finalIcResults);
  if (prepared.consumerOp && prepared.consumerReplacement) {
    rewriter.replaceOp(prepared.consumerOp, prepared.consumerReplacement);
  } else {
    if (!prepared.outerReplacements.empty()) {
      rewriter.replaceOp(prepared.originalConv, prepared.outerReplacements);
    } else {
      rewriter.replaceOp(prepared.originalConv, finalIcResults);
    }
  }

  for (int i = (int)prepared.outerLoops.size() - 1; i >= 0; --i) {
    auto loopOp = cast<scf::ForOp>(prepared.outerLoops[i].getOperation());
    scf::ForOp partialLoop;
    if (succeeded(scf::peelForLoopAndSimplifyBounds(
            rewriter, loopOp, partialLoop))) {
      inheritNpuAttributes(loopOp, partialLoop);
    }
  }

  return success();
}

} // namespace

SmallVector<int64_t> npux::inferConvActivationTileShape(
    linalg::GenericOp convOp, ArrayRef<int64_t> seedTileSizes) {
  return inferConvActivationTileShapeImpl(convOp, seedTileSizes);
}

SmallVector<int64_t> npux::inferConvWeightTileShape(
    linalg::GenericOp convOp, ArrayRef<int64_t> seedTileSizes) {
  return inferConvWeightTileShapeImpl(convOp, seedTileSizes);
}

LogicalResult npux::tileConvWithRoot(
    const FusionCursor &cursor, PatternRewriter &rewriter) {
  auto seedBase = cursor.seed;
  auto root = cursor.tail;
  auto seed = cast<linalg::GenericOp>(seedBase);
  if (!seed || !root)
    return failure();

  auto seedLibCall = seed->getAttrOfType<StringAttr>("library_call");
  if (!seedLibCall || seedLibCall.getValue() != "npu_conv")
    return failure();

  SmallVector<int64_t> seedTileSizes = cursor.seedTileSizes;
  SmallVector<int64_t> tailTileSizes = cursor.tailTileSizes;
  if (seedTileSizes.empty())
    seedTileSizes = tailTileSizes;
  if (seedTileSizes.size() < 5)
    return failure();

  SmallVector<int64_t> spatialTileSizes;
  if (root == seed.getOperation()) {
    spatialTileSizes = seedTileSizes;
    spatialTileSizes[4] = 0;
  } else {
    spatialTileSizes = tailTileSizes;
    if (spatialTileSizes.empty())
      return failure();
    if (spatialTileSizes.size() >= 5)
      spatialTileSizes.back() = 0;
  }

  SmallVector<int64_t> icTileSizes = {0, 0, 0, 0, seedTileSizes[4]};

  FailureOr<PreparedConvTiling> prepared =
      (root == seed.getOperation())
      ? prepareStandaloneConv(
            seed, spatialTileSizes, seedTileSizes, tailTileSizes,
            cursor.seedInputDmaAnalyses, rewriter)
      : prepareFusedConv(
            seed, root, spatialTileSizes, seedTileSizes, tailTileSizes,
            cursor.seedInputDmaAnalyses, cursor.rootOutputDmaAnalysis,
            rewriter);
  if (failed(prepared))
    return failure();

  return finalizePreparedConv(*prepared, icTileSizes, rewriter);
}
