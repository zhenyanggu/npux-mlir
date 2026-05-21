#include "src/Conversion/NpuTiling/Strategy/Gemm.hpp"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

#include <cmath>

using namespace mlir;

namespace {

static constexpr StringLiteral kFusionLookupAttr = "__npux.fusion_lookup_id";

static void tagInnerComputeOp(
    Operation *containerOp, StringRef phase, RewriterBase &rewriter) {
  containerOp->walk([&](Operation *op) {
    if (op->hasAttr("npu.tiled"))
      op->setAttr("npu.loop_stage", rewriter.getStringAttr(phase));
  });
}

static int64_t getStaticTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (lb && ub && step)
    return (int64_t)std::ceil((double)(*ub - *lb) / *step);
  return -1;
}

static void inheritNpuAttributes(scf::ForOp source, scf::ForOp target) {
  if (auto attr = source->getAttr("npu.target"))
    target->setAttr("npu.target", attr);
  if (auto attr = source->getAttr("npu.loop_dim"))
    target->setAttr("npu.loop_dim", attr);
}

static StringRef getLoopLabelForGemmSpatialDim(int64_t rank, size_t dimIdx) {
  if (rank == 3) {
    if (dimIdx == 0)
      return "Batch";
    if (dimIdx == 1)
      return "N";
    return "M";
  }

  if (dimIdx == 0)
    return "N";
  return "M";
}

static SmallVector<int64_t> getLoopTileSizesFromResultTileShape(
    Operation *op, ArrayRef<int64_t> resultTileShape) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp)
    return {};

  ArrayRef<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
  unsigned resultMapIdx = genericOp.getNumDpsInputs();
  if (resultMapIdx >= indexingMaps.size())
    return {};

  AffineMap resultMap = indexingMaps[resultMapIdx];
  if (resultMap.getNumResults() != resultTileShape.size())
    return {};

  SmallVector<int64_t> loopTileSizes(resultMap.getNumDims(), 0);
  for (auto [idx, expr] : llvm::enumerate(resultMap.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return {};
    unsigned pos = dimExpr.getPosition();
    if (pos >= loopTileSizes.size())
      return {};
    loopTileSizes[pos] = resultTileShape[idx];
  }

  SmallVector<int64_t> loopRanges = genericOp.getStaticLoopRanges();
  if (loopRanges.size() != loopTileSizes.size())
    return {};
  for (auto [idx, size] : llvm::enumerate(loopTileSizes)) {
    if (size == 0)
      loopTileSizes[idx] = loopRanges[idx];
  }
  return loopTileSizes;
}

static linalg::GenericOp findTiledGenericOpByMarker(
    scf::SCFTileAndFuseResult &fuseResult, int64_t marker) {
  for (Operation *op : fuseResult.tiledAndFusedOps) {
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    if (!genericOp)
      continue;
    auto markerAttr =
        genericOp->getAttrOfType<IntegerAttr>(kFusionLookupAttr);
    if (markerAttr && markerAttr.getInt() == marker)
      return genericOp;
  }
  return nullptr;
}

// Gets the full tiling configuration [Batch..., N, M, K] for gemm-like seeds.
static SmallVector<int64_t> getGemmTileSizes(linalg::GenericOp op) {
  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  int64_t rank = loopRanges.size();
  if (rank != 3 && rank != 4)
    return {};

  auto &config = npux::NPUConfig::getInstance();
  std::vector<int64_t> manualSizes = config.getMatMulTileSize();
  if (!manualSizes.empty() && manualSizes.size() >= 3) {
    SmallVector<int64_t> finalTileSizes(rank, 0);
    int64_t tm = manualSizes[0];
    int64_t tn = manualSizes[1];
    int64_t tk = manualSizes[2];
    if (rank == 4) {
      finalTileSizes[0] = 1;
      finalTileSizes[1] = tn;
      finalTileSizes[2] = tm;
      finalTileSizes[3] = tk;
    } else {
      finalTileSizes[0] = tn;
      finalTileSizes[1] = tm;
      finalTileSizes[2] = tk;
    }
    return finalTileSizes;
  }

  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  return costModel.getOptimalTileSizes(op);
}

} // namespace

LogicalResult npux::tileGemmWithRoot(
    const FusionCursor &cursor, PatternRewriter &rewriter) {
  auto seedBase = cursor.seed;
  auto root = cursor.tail;
  auto seed = cast<linalg::GenericOp>(seedBase);
  auto seedLibCall = seed->getAttrOfType<StringAttr>("library_call");
  if (seedLibCall.getValue() != "npu_gemm" &&
      seedLibCall.getValue() != "npu_matmul")
    return failure();

  SmallVector<int64_t> tileSizes = cursor.seedTileSizes;
  if (tileSizes.empty())
    tileSizes = getGemmTileSizes(seed);

  SmallVector<int64_t> rootTileSizes;
  SmallVector<int64_t> spatialTileSizes = tileSizes;
  if (root == seed.getOperation()) {
    spatialTileSizes.back() = 0;
    rootTileSizes = spatialTileSizes;
  } else {
    rootTileSizes = getLoopTileSizesFromResultTileShape(
        root, cursor.tailTileSizes);
    if (rootTileSizes.empty())
      return failure();
  }
  SmallVector<int64_t> kTileSizes(tileSizes.size(), 0);
  kTileSizes.back() = tileSizes.back();

  auto rootTilingInterface = cast<TilingInterface>(root);
  scf::SCFTileAndFuseOptions fuseOptions;
  fuseOptions.tilingOptions.setTileSizes(
      getAsOpFoldResult(rewriter.getI64ArrayAttr(rootTileSizes)));

  llvm::SmallPtrSet<Operation *, 8> chainOps;
  for (Operation *op : cursor.chainOps)
    chainOps.insert(op);
  Operation *targetOpPtr = seed.getOperation();
  auto stopAfterSeed = std::make_shared<bool>(false);
  fuseOptions.setFusionControlFn(
      [targetOpPtr, chainOps, stopAfterSeed](
          tensor::ExtractSliceOp, OpResult originalProducer, bool)
          -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
        if (*stopAfterSeed)
          return std::nullopt;
        if (!chainOps.contains(originalProducer.getOwner()))
          return std::nullopt;
        if (originalProducer.getOwner() == targetOpPtr)
          *stopAfterSeed = true;
        return scf::SCFTileAndFuseOptions::ControlFnResult{};
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

  auto spatialLoops = fuseResult->loops;
  int currentLoopIdx = 0;
  for (size_t dimIdx = 0; dimIdx < rootTileSizes.size(); ++dimIdx) {
    if (rootTileSizes[dimIdx] == 0 || currentLoopIdx >= spatialLoops.size())
      continue;

    spatialLoops[currentLoopIdx]->setAttr(
        "npu.loop_dim",
        rewriter.getStringAttr(
            getLoopLabelForGemmSpatialDim(rootTileSizes.size(), dimIdx)));
    spatialLoops[currentLoopIdx]->setAttr(
        "npu.target", rewriter.getStringAttr("npu"));
    currentLoopIdx++;
  }

  auto tiledRoot = findTiledGenericOpByMarker(*fuseResult, 1);
  if (!tiledRoot)
    return failure();
  tiledRoot->setAttr("npu.tiled", rewriter.getUnitAttr());
  tiledRoot->removeAttr(kFusionLookupAttr);

  Value originalRootResult =
      root->getNumResults() > 0 ? root->getResult(0) : Value{};
  Value rootReplacement;
  if (originalRootResult &&
      fuseResult->replacements.count(originalRootResult)) {
    rootReplacement = fuseResult->replacements.lookup(originalRootResult);
  }

  linalg::GenericOp fusedGemmOp = findTiledGenericOpByMarker(*fuseResult, 0);
  bool isStandaloneGemm = (root == seed.getOperation());
  if (!fusedGemmOp)
    return failure();
  fusedGemmOp->setAttr("npu.tiled", rewriter.getUnitAttr());
  fusedGemmOp->removeAttr(kFusionLookupAttr);

  Value rootExternalDest = tiledRoot.getDpsInitOperand(0)->get();

  if (!isStandaloneGemm) {
    auto relocatedRoot =
        npux::cloneGenericOpToMemorySpace(tiledRoot, 2, rewriter);
    if (failed(relocatedRoot))
      return failure();
    tiledRoot = *relocatedRoot;
    tiledRoot->setAttr("npu.tiled", rewriter.getUnitAttr());
  }

  auto relocatedGemm =
      npux::cloneGenericOpToMemorySpace(fusedGemmOp, 3, rewriter);
  if (failed(relocatedGemm))
    return failure();
  fusedGemmOp = *relocatedGemm;
  fusedGemmOp->setAttr("npu.tiled", rewriter.getUnitAttr());
  if (isStandaloneGemm)
    tiledRoot = fusedGemmOp;

  SmallVector<Value> dmaInputs;
  dmaInputs.reserve(fusedGemmOp.getInputs().size());
  for (auto [idx, operand] : llvm::enumerate(fusedGemmOp.getInputs())) {
    // Match the legacy DMA insertion behavior: only lhs/rhs use mvin. Extra
    // operands such as bias remain as external inputs to the compute op.
    if (idx >= 2) {
      dmaInputs.push_back(operand);
      continue;
    }

    rewriter.setInsertionPoint(fusedGemmOp);
    auto mvinOp = npux::createDmaGenericOp(
        rewriter, fusedGemmOp.getLoc(), operand, "npu_dma_mvin", 2);
    FailureOr<Value> dmaResult = mvinOp.getResult(0);
    if (idx < cursor.seedInputDmaAnalyses.size()) {
      dmaResult = npux::maybeSplitDmaOp(
          mvinOp, cursor.seedInputDmaAnalyses[idx], rewriter);
    } else if (auto inputAnalysis = buildFullTensorDmaAnalysis(operand)) {
      dmaResult = npux::maybeSplitDmaOp(mvinOp, *inputAnalysis, rewriter);
    }
    if (failed(dmaResult))
      return failure();
    dmaInputs.push_back(*dmaResult);
  }
  rewriter.modifyOpInPlace(fusedGemmOp, [&]() {
    fusedGemmOp.getInputsMutable().assign(dmaInputs);
  });

  if (!isStandaloneGemm) {
    rewriter.setInsertionPointAfter(tiledRoot);
    auto mvoutOp = npux::createDmaGenericOp(rewriter, tiledRoot.getLoc(),
        tiledRoot.getResult(0), "npu_dma_mvout", 1, rootExternalDest);
    if (cursor.rootOutputDmaAnalysis) {
      if (failed(npux::maybeSplitDmaOp(
              mvoutOp, *cursor.rootOutputDmaAnalysis, rewriter)))
        return failure();
    } else if (auto outputAnalysis = buildFullTensorDmaAnalysis(rootExternalDest)) {
      if (failed(npux::maybeSplitDmaOp(mvoutOp, *outputAnalysis, rewriter)))
        return failure();
    }
    rewriter.replaceAllUsesExcept(
        tiledRoot.getResult(0), mvoutOp.getResult(0), mvoutOp.getOperation());
  }

  scf::SCFTilingOptions kOptions;
  kOptions.setTileSizes(getAsOpFoldResult(rewriter.getI64ArrayAttr(kTileSizes)));
  FailureOr<scf::SCFTilingResult> kTilingResult = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(fusedGemmOp.getOperation()), kOptions);
  if (failed(kTilingResult))
    return failure();

  if (!kTilingResult->loops.empty()) {
    kTilingResult->loops.front()->setAttr(
        "npu.loop_dim", rewriter.getStringAttr("K"));
    kTilingResult->loops.front()->setAttr(
        "npu.target", rewriter.getStringAttr("npu"));
  }

  SmallVector<Value> finalKResults = kTilingResult->replacements;
  if (!kTilingResult->loops.empty()) {
    auto loopOp = cast<scf::ForOp>(kTilingResult->loops.back().getOperation());
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
        finalKResults = restLoop->getResults();
      } else {
        scf::ForOp tailLoop;
        if (succeeded(scf::peelForLoopAndSimplifyBounds(
                rewriter, restLoop, tailLoop)) ||
            succeeded(peelForLoopLastIteration(rewriter, restLoop, tailLoop))) {
          inheritNpuAttributes(restLoop, tailLoop);
          tagInnerComputeOp(tailLoop, "tail", rewriter);
          tagInnerComputeOp(restLoop, "body", rewriter);
          finalKResults = tailLoop->getResults();
        } else {
          tagInnerComputeOp(restLoop, "body", rewriter);
          finalKResults = restLoop->getResults();
        }
      }
    }
  }

  if (isStandaloneGemm) {
    if (finalKResults.empty())
      return failure();
    Value finalComputeResult = finalKResults.front();
    Operation *finalOwner = finalComputeResult.getDefiningOp();
    if (!finalOwner)
      return failure();

    rewriter.setInsertionPointAfter(finalOwner);
    auto mvoutOp = npux::createDmaGenericOp(rewriter, fusedGemmOp.getLoc(),
        finalComputeResult, "npu_dma_mvout", 1, rootExternalDest);
    if (cursor.rootOutputDmaAnalysis) {
      if (failed(npux::maybeSplitDmaOp(
              mvoutOp, *cursor.rootOutputDmaAnalysis, rewriter)))
        return failure();
    } else if (auto outputAnalysis = buildFullTensorDmaAnalysis(rootExternalDest)) {
      if (failed(npux::maybeSplitDmaOp(mvoutOp, *outputAnalysis, rewriter)))
        return failure();
    }
    rewriter.replaceOp(fusedGemmOp, mvoutOp.getResults());
  } else {
    rewriter.replaceOp(fusedGemmOp, finalKResults);
    if (rootReplacement)
      rewriter.replaceOp(root, rootReplacement);
  }

  for (int i = (int)spatialLoops.size() - 1; i >= 0; --i) {
    auto loopOp = cast<scf::ForOp>(spatialLoops[i].getOperation());
    scf::ForOp partialLoop;
    if (succeeded(scf::peelForLoopAndSimplifyBounds(
            rewriter, loopOp, partialLoop)))
      inheritNpuAttributes(loopOp, partialLoop);
  }

  return success();
}
