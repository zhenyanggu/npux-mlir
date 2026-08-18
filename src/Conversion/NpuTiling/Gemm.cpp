//==============================================
// src/Conversion/NpuTiling/Gemm.cpp
// This file implements the tiling strategy of
// gemm op (converted linalg.generic)
//==============================================

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include <cmath> // for ceil
#include <limits>

using namespace mlir;
using namespace npux;

namespace {

// -----------------------------------------------------------------------------
// Helper: 给 Loop 内部打了 "npu.tiled" 标记的 Op 添加 Phase 标签
// -----------------------------------------------------------------------------
static void tagInnerComputeOp(
    Operation *containerOp, StringRef phase, RewriterBase &rewriter) {
  containerOp->walk([&](Operation *op) {
    if (op->hasAttr("npu.tiled")) {
      op->setAttr("npu.loop_stage", rewriter.getStringAttr(phase));
    }
  });
}

// -----------------------------------------------------------------------------
// Helper: 估算静态 Loop 的迭代次数
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Helper: Peel Last Iteration
// -----------------------------------------------------------------------------
static LogicalResult peelForLoopLastIteration(
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

// Select descriptor tiles, not the RTL's 32x32 execution microtiles. The SA
// scheduler walks each descriptor across its full M/N extent. Every legal M/N
// pair is evaluated against the physical 64 KiB A/W/O banks (2048 x 32-byte
// words) so the result is driven by external traffic and reuse rather than a
// fixed M/N/K ordering.
static SmallVector<int64_t, 3> calculateAutoGemmTile(
    int64_t M, int64_t N, int64_t K, bool outputIsInt8) {
  constexpr int64_t kBankWords = 2048;
  constexpr int64_t kWordBytes = 32;
  constexpr int64_t kMaxSaK = 4096;
  constexpr int64_t kArrayDimension = 32;
  const int64_t outputLanes = outputIsInt8 ? 32 : 8;

  if (M <= 0 || N <= 0 || K <= 0)
    return {};

  struct Score {
    int64_t transferWords;
    int64_t descriptorTiles;
    int64_t microTileWaste;
    int64_t payload;
  };
  auto ceilDiv = [](int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
  };
  auto better = [](const Score &candidate, const Score &best) {
    if (candidate.transferWords != best.transferWords)
      return candidate.transferWords < best.transferWords;
    if (candidate.descriptorTiles != best.descriptorTiles)
      return candidate.descriptorTiles < best.descriptorTiles;
    if (candidate.microTileWaste != best.microTileWaste)
      return candidate.microTileWaste < best.microTileWaste;
    return candidate.payload > best.payload;
  };
  auto sumKWords = [&](int64_t tileK, int64_t tileCount) {
    const int64_t tail = K - (tileCount - 1) * tileK;
    return (tileCount - 1) * ceilDiv(tileK, kWordBytes) +
           ceilDiv(tail, kWordBytes);
  };
  auto sumNOutputWords = [&](int64_t tileN, int64_t tileCount) {
    int64_t words = 0;
    for (int64_t index = 0; index < tileCount; ++index)
      words += ceilDiv(std::min(tileN, N - index * tileN), outputLanes);
    return words;
  };
  auto sumMicroTileBlocks = [&](int64_t total, int64_t tile) {
    int64_t blocks = 0;
    for (int64_t offset = 0; offset < total; offset += tile)
      blocks += ceilDiv(std::min(tile, total - offset), kArrayDimension);
    return blocks;
  };

  SmallVector<int64_t, 3> bestTile;
  Score bestScore{std::numeric_limits<int64_t>::max(), 0, 0, 0};
  const int64_t maxM = std::min(M, kBankWords);
  const int64_t maxN = std::min(N, kBankWords);
  SmallVector<int64_t> outputWordsByN(maxN + 1);
  SmallVector<int64_t> microBlocksByM(maxM + 1);
  SmallVector<int64_t> microBlocksByN(maxN + 1);
  for (int64_t tileM = 1; tileM <= maxM; ++tileM)
    microBlocksByM[tileM] = sumMicroTileBlocks(M, tileM);
  for (int64_t tileN = 1; tileN <= maxN; ++tileN) {
    outputWordsByN[tileN] =
        sumNOutputWords(tileN, ceilDiv(N, tileN));
    microBlocksByN[tileN] = sumMicroTileBlocks(N, tileN);
  }
  for (int64_t tileM = 1; tileM <= maxM; ++tileM) {
    for (int64_t tileN = 1; tileN <= maxN; ++tileN) {
      if (tileM * ceilDiv(tileN, outputLanes) > kBankWords)
        continue;

      const int64_t maxKWords = std::min(kBankWords / tileM,
          kBankWords / tileN);
      if (maxKWords == 0)
        continue;
      const int64_t tileK = std::min(K,
          std::min(kMaxSaK, maxKWords * kWordBytes));
      const int64_t mTiles = ceilDiv(M, tileM);
      const int64_t nTiles = ceilDiv(N, tileN);
      const int64_t kTiles = ceilDiv(K, tileK);
      const int64_t descriptorTiles = mTiles * nTiles * kTiles;
      const int64_t aWords = M * sumKWords(tileK, kTiles) * nTiles;
      const int64_t wWords = N * sumKWords(tileK, kTiles) * mTiles;
      const int64_t oWords = M * outputWordsByN[tileN];
      const int64_t microTiles =
          microBlocksByM[tileM] * microBlocksByN[tileN] * kTiles;
      const Score score{aWords + wWords + oWords, descriptorTiles,
          microTiles * kArrayDimension * kArrayDimension - M * N * kTiles,
          tileM * tileN * tileK};
      if (bestTile.empty() || better(score, bestScore)) {
        bestTile = {tileM, tileN, tileK};
        bestScore = score;
      }
    }
  }
  return bestTile;
}

SmallVector<int64_t> getGemmTileSizes(linalg::GenericOp op, bool requiresOutputInSpm = true) { 
  auto &config = npux::NPUConfig::getInstance();

  // 1. 获取手动配置 (这里假设用户的配置还是按 Tm, Tn, Tk 填写的)
  std::vector<int64_t> manualSizes = config.getMatMulTileSize();

  // 2. 获取 Loop Ranges
  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  int64_t rank = loopRanges.size(); // 2D为3, 3D为4, 4D为5

  if (rank != 3 && rank != 4 && rank != 5) {
    return {};
  }

  // 根据迭代器顺序提取维度
  int64_t K = loopRanges[rank - 1];
  int64_t M = loopRanges[rank - 2];
  int64_t N = loopRanges[rank - 3];
  
  // 仅用于日志打印，把多维 Batch 乘起来作为一个整体指标
  int64_t B = (rank >= 4) ? loopRanges[0] : 1; 
  if (rank == 5) B *= loopRanges[1];

  SmallVector<int64_t, 3> computedSizes;
  bool isManual = false;

  if (!manualSizes.empty() && manualSizes.size() >= 3) {
    // An explicit user configuration intentionally overrides automatic
    // traffic-based tiling. Descriptor legality remains checked downstream.
    computedSizes = {manualSizes[0], manualSizes[1], manualSizes[2]};
    isManual = true;
  } else {
    // mv_acc_to_spm receives INT8 O-bank data. Standalone integer matmul
    // retains raw INT32 results and therefore has eight lanes per O-bank word.
    computedSizes = calculateAutoGemmTile(M, N, K, requiresOutputInSpm);
  }

  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "Tiling [Gemm] (" << (isManual ? "Manual" : "Auto") << "): "
     << "Problem=[B:" << B << ", N:" << N << ", M:" << M << ", K:" << K << "] "
     << "-> Tile=[Tm:" << computedSizes[0] << ", Tn:" << computedSizes[1]
     << ", Tk:" << computedSizes[2] << "]\n";
  llvm::errs() << os.str();

  // 3. 构建最终的 Tile Sizes 数组
  SmallVector<int64_t> finalTileSizes(rank, 0);

  // 按照 Linalg Generic Iterator 的顺序填充分块大小
  if (rank == 5) {
    finalTileSizes[0] = 1;                // B1 按 1 分块
    finalTileSizes[1] = 1;                // B2 按 1 分块
    finalTileSizes[2] = computedSizes[1]; // N -> tn
    finalTileSizes[3] = computedSizes[0]; // M -> tm
    finalTileSizes[4] = computedSizes[2]; // K -> tk
  } else if (rank == 4) {
    finalTileSizes[0] = 1;                // Batch 永远按 1 分块
    finalTileSizes[1] = computedSizes[1]; // N -> tn
    finalTileSizes[2] = computedSizes[0]; // M -> tm
    finalTileSizes[3] = computedSizes[2]; // K -> tk
  } else {
    finalTileSizes[0] = computedSizes[1]; // N -> tn
    finalTileSizes[1] = computedSizes[0]; // M -> tm
    finalTileSizes[2] = computedSizes[2]; // K -> tk
  }

  return finalTileSizes;
}

// -----------------------------------------------------------------------------
// Main Pattern for Gemm
// -----------------------------------------------------------------------------
struct NpuGemmTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {

    // 1. 匹配 Consumer (mv_acc_to_spm)
    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (!libCall || libCall.getValue() != "mv_acc_to_spm") {
      return failure();
    }

    // 防止重复 Tiling
    if (op->hasAttr("npu.tiled")) {
      return failure();
    }

    // 2. 向上追溯寻找 Producer (npu_gemm / npu_matmul)
    Value input = op->getOperand(0);
    auto gemmOp = input.getDefiningOp<linalg::GenericOp>();
    if (!gemmOp || !gemmOp->hasAttr("library_call")) {
      return failure();
    }
    StringRef prodOpName =
        gemmOp->getAttrOfType<StringAttr>("library_call").getValue();
    if (prodOpName != "npu_gemm" && prodOpName != "npu_matmul") {
      return failure();
    }

    // 3. 获取 npu_gemm 的完整切分配置 [Batch..., M, N, K]
    SmallVector<int64_t> tileSizes =
        getGemmTileSizes(gemmOp); // 假设 Helper 中有此函数
    if (tileSizes.empty())
      return failure();

    int64_t rank = tileSizes.size();
    if (rank < 2)
      return failure(); // 至少要有 M, K

    // 拆分 Tile Sizes：空间维度 (给 mv) 和 规约维度 (给 gemm 的 K)
    SmallVector<int64_t> spatialTileSizes = tileSizes;
    spatialTileSizes.back() = 0; // 最后一个维度 K 设为 0，不在这里切

    SmallVector<int64_t> kTileSizes(rank, 0);
    kTileSizes.back() = tileSizes.back(); // 只有 K 维度有值

    // =================================================================
    // Phase 1: 沿着空间维度对 mv_acc_to_spm 进行 Tiling，并融合 npu_gemm
    // =================================================================
    auto consumerTilingInterface = cast<TilingInterface>(op.getOperation());
    scf::SCFTileAndFuseOptions fuseOptions;
    fuseOptions.tilingOptions.setTileSizes(
        getAsOpFoldResult(rewriter.getI64ArrayAttr(spatialTileSizes)));

    Value inputVal = op.getInputs()[0];

    auto targetProducerOp = inputVal.getDefiningOp<linalg::GenericOp>();
    if (!targetProducerOp || !targetProducerOp->hasAttr("library_call")) {
      return failure();
    }

    // 2. 校验名字
    StringRef producerName =
        targetProducerOp->getAttrOfType<StringAttr>("library_call").getValue();
    if (producerName != "npu_gemm" && producerName != "npu_matmul" &&
        producerName != "npu_conv") {
      return failure();
    }

    Operation *targetOpPtr = targetProducerOp.getOperation();

    // 捕获裸指针 targetOpPtr
    fuseOptions.setFusionControlFn(
        [targetOpPtr](tensor::ExtractSliceOp candidateSliceOp,
            OpResult originalProducer, bool isDestinationOperand)
            -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          // 直接做指针地址比对，简单高效，且没有 const 问题
          if (originalProducer.getOwner() == targetOpPtr) {
            return scf::SCFTileAndFuseOptions::ControlFnResult{};
          }
          return std::nullopt;
        });

    FailureOr<scf::SCFTileAndFuseResult> fuseResult =
        scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, consumerTilingInterface, fuseOptions);

    if (failed(fuseResult)) {
      return failure();
    }

    // 给空间循环打上 Label
    auto spatialLoops = fuseResult->loops;
    int currentLoopIdx = 0;
    for (size_t dimIdx = 0; dimIdx < rank - 1; ++dimIdx) {
      if (spatialTileSizes[dimIdx] == 0)
        continue;
      if (currentLoopIdx >= spatialLoops.size())
        break;

      StringRef label;
      if (rank == 5) { // [B1, B2, N, M, K] 空间维度没有 K
        if (dimIdx == 0) label = "Batch1";
        else if (dimIdx == 1) label = "Batch2";
        else if (dimIdx == 2) label = "N";
        else if (dimIdx == 3) label = "M";
      } else if (rank == 4) { // [Batch, N, M, K]
        if (dimIdx == 0) label = "Batch";
        else if (dimIdx == 1) label = "N";
        else if (dimIdx == 2) label = "M";
      } else { // rank == 3, [N, M, K]
        if (dimIdx == 0) label = "N";
        else if (dimIdx == 1) label = "M";
      }

      spatialLoops[currentLoopIdx]->setAttr(
          "npu.loop_dim", rewriter.getStringAttr(label));
      spatialLoops[currentLoopIdx]->setAttr(
          "npu.target", rewriter.getStringAttr("npu"));
      currentLoopIdx++;
    }

    // =================================================================
    // Phase 2: 在最内层空间循环中，找到被融合进来的局部 npu_gemm，并处理 ACC
    // 内存
    // =================================================================
    auto tiledConsumer = fuseResult->tiledAndFusedOps.front();
    tiledConsumer->setAttr("npu.tiled", rewriter.getUnitAttr());

    auto fusedGemmOp =
        tiledConsumer->getOperand(0).getDefiningOp<linalg::GenericOp>();
    if (!fusedGemmOp)
      return failure();

    fusedGemmOp->setAttr("npu.tiled", rewriter.getUnitAttr());

    Location loc = fusedGemmOp.getLoc();
    rewriter.setInsertionPoint(fusedGemmOp);

    // 1. 分配 ACC 内存 (Space 3)
    OpOperand *gemmOutOperand = fusedGemmOp.getDpsInitOperand(0);
    Value gemmOutVal = gemmOutOperand->get();
    auto gemmOutType = cast<RankedTensorType>(gemmOutVal.getType());

    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < gemmOutType.getRank(); ++i) {
      if (gemmOutType.isDynamicDim(i)) {
        dynamicSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, gemmOutVal, i));
      }
    }

    // 将 memory_space 编码进 Tensor 的 Type
    auto memSpaceAttr = rewriter.getI32IntegerAttr(3);
    auto accTensorType = RankedTensorType::get(
        gemmOutType.getShape(), gemmOutType.getElementType(), memSpaceAttr);

    auto accAlloc = rewriter.create<bufferization::AllocTensorOp>(
        loc, accTensorType, dynamicSizes);

    auto newGemmOp = rewriter.create<linalg::GenericOp>(loc,
        TypeRange{accTensorType}, fusedGemmOp.getInputs(), ValueRange{accAlloc},
        fusedGemmOp.getIndexingMapsArray(),
        fusedGemmOp.getIteratorTypesArray());

    rewriter.inlineRegionBefore(fusedGemmOp.getRegion(), newGemmOp.getRegion(),
        newGemmOp.getRegion().begin());
    newGemmOp->setAttrs(fusedGemmOp->getAttrs());
    rewriter.replaceOp(fusedGemmOp, newGemmOp.getResults());

    fusedGemmOp = newGemmOp; // 更新指针供后续使用

    // =================================================================
    // Phase 3: 对局部的 npu_gemm 沿着 K (Reduction) 维度进行 Tiling
    // =================================================================
    scf::SCFTilingOptions kOptions;
    kOptions.setTileSizes(
        getAsOpFoldResult(rewriter.getI64ArrayAttr(kTileSizes)));

    FailureOr<scf::SCFTilingResult> kTilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(fusedGemmOp.getOperation()), kOptions);

    if (failed(kTilingResult))
      return failure();

    // 给 K 循环打标签
    if (!kTilingResult->loops.empty()) {
      kTilingResult->loops.front()->setAttr(
          "npu.loop_dim", rewriter.getStringAttr("K"));
      kTilingResult->loops.front()->setAttr(
          "npu.target", rewriter.getStringAttr("npu"));
    }

    // =================================================================
    // Phase 4: K 维度的 Head -> Body -> Tail Peeling
    // =================================================================
    auto kLoops = kTilingResult->loops;
    SmallVector<Value> finalKResults = kTilingResult->replacements;

    if (!kLoops.empty()) {
      auto loopOp = cast<scf::ForOp>(kLoops.back().getOperation());

      int64_t tripCount = getStaticTripCount(loopOp);
      if (tripCount == 1) {
        tagInnerComputeOp(loopOp, "single", rewriter);
      } else {
        scf::ForOp restLoop = loopOp;

        // --- Phase 4.1: Peel Head ---
        scf::ForOp headLoop;
        if (succeeded(peelForLoopFirstIteration(rewriter, loopOp, headLoop))) {
          inheritNpuAttributes(restLoop, headLoop);
          tagInnerComputeOp(headLoop, "head", rewriter);
          restLoop = loopOp;
        }

        int64_t restTripCount = getStaticTripCount(restLoop);

        // --- Phase 4.2: Peel Tail ---
        if (restTripCount == 1) {
          tagInnerComputeOp(restLoop, "tail", rewriter);
          finalKResults = restLoop->getResults();
        } else {
          scf::ForOp tailLoop;
          bool hasTail = false;

          scf::ForOp partialLoop;
          if (succeeded(scf::peelForLoopAndSimplifyBounds(
                  rewriter, restLoop, partialLoop))) {
            tailLoop = partialLoop;
            hasTail = true;
          } else {
            scf::ForOp forceTail;
            if (succeeded(
                    peelForLoopLastIteration(rewriter, restLoop, forceTail))) {
              tailLoop = forceTail;
              hasTail = true;
            }
          }

          if (hasTail) {
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
    // =================================================================
    // Phase 6: 链接数据流并替换原 Op
    // =================================================================
    // 1. 替换融合内部计算的产物
    rewriter.replaceOp(fusedGemmOp, finalKResults);

    // 2. 将外层原始的 mv_acc_to_spm 替换为 Tile & Fuse 流程的最终产物
    Value originalResult = op->getResult(0);
    if (fuseResult->replacements.count(originalResult)) {
      rewriter.replaceOp(op, fuseResult->replacements[originalResult]);
    } else {
      return failure();
    }

    // =================================================================
    // Phase 5: 空间循环维度的常规尾部剥离 (Tail Peeling)
    // =================================================================
    for (int i = (int)spatialLoops.size() - 1; i >= 0; --i) {
      auto loopOp = cast<scf::ForOp>(spatialLoops[i].getOperation());
      scf::ForOp partialLoop;
      if (succeeded(scf::peelForLoopAndSimplifyBounds(
              rewriter, loopOp, partialLoop))) {
        inheritNpuAttributes(loopOp, partialLoop);
      }
    }

    return success();
  }
};
// -----------------------------------------------------------------------------
// Pattern for Standalone Matmul (e.g., npu_matmul_integer without mv_acc_to_spm)
// -----------------------------------------------------------------------------
struct NpuStandaloneMatmulTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {

    // 1. 匹配目标算子 npu_matmul_integer
    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (!libCall || libCall.getValue() != "npu_matmul_integer") {
      return failure();
    }

    // 防止重复 Tiling
    if (op->hasAttr("npu.tiled")) {
      return failure();
    }

    // 2. 获取切分配置 [Batch..., M, N, K]
    SmallVector<int64_t> tileSizes = getGemmTileSizes(op, /*requiresOutputInSpm=*/false);
    if (tileSizes.empty())
      return failure();

    int64_t rank = tileSizes.size();
    if (rank < 2)
      return failure(); // 至少要有 M, K

    // 拆分 Tile Sizes：空间维度 (M, N) 和 规约维度 (K)
    SmallVector<int64_t> spatialTileSizes = tileSizes;
    spatialTileSizes.back() = 0; // 最后一个维度 K 设为 0，空间循环不切分 K

    SmallVector<int64_t> kTileSizes(rank, 0);
    kTileSizes.back() = tileSizes.back(); // 只有 K 维度有值

    // =================================================================
    // Phase 1: 沿着空间维度对 npu_matmul_integer 进行 Tiling
    // 注意：这里不再使用 Fuse，而是直接 tileUsingSCF
    // =================================================================
    auto tilingInterface = cast<TilingInterface>(op.getOperation());
    scf::SCFTilingOptions spatialOptions;
    spatialOptions.setTileSizes(
        getAsOpFoldResult(rewriter.getI64ArrayAttr(spatialTileSizes)));

    FailureOr<scf::SCFTilingResult> spatialTilingResult =
        scf::tileUsingSCF(rewriter, tilingInterface, spatialOptions);

    if (failed(spatialTilingResult)) {
      return failure();
    }

    // 给空间循环打上 Label
    auto spatialLoops = spatialTilingResult->loops;
    int currentLoopIdx = 0;
    for (size_t dimIdx = 0; dimIdx < rank - 1; ++dimIdx) {
      if (spatialTileSizes[dimIdx] == 0)
        continue;
      if (currentLoopIdx >= spatialLoops.size())
        break;

      StringRef label;
      if (rank == 5) {
        if (dimIdx == 0) label = "Batch1";
        else if (dimIdx == 1) label = "Batch2";
        else if (dimIdx == 2) label = "N";
        else if (dimIdx == 3) label = "M";
      } else if (rank == 4) {
        if (dimIdx == 0) label = "Batch";
        else if (dimIdx == 1) label = "N";
        else if (dimIdx == 2) label = "M";
      } else {
        if (dimIdx == 0) label = "N";
        else if (dimIdx == 1) label = "M";
      }

      spatialLoops[currentLoopIdx]->setAttr(
          "npu.loop_dim", rewriter.getStringAttr(label));
      spatialLoops[currentLoopIdx]->setAttr(
          "npu.target", rewriter.getStringAttr("npu"));
      currentLoopIdx++;
    }

    // =================================================================
    // Phase 2: 获取局部 Matmul 并沿着 K (Reduction) 维度进行 Tiling
    // =================================================================
    auto tiledMatmulOp = cast<linalg::GenericOp>(spatialTilingResult->tiledOps.front());
    tiledMatmulOp->setAttr("npu.tiled", rewriter.getUnitAttr());

    scf::SCFTilingOptions kOptions;
    kOptions.setTileSizes(
        getAsOpFoldResult(rewriter.getI64ArrayAttr(kTileSizes)));

    FailureOr<scf::SCFTilingResult> kTilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(tiledMatmulOp.getOperation()), kOptions);

    if (failed(kTilingResult))
      return failure();

    // 给 K 循环打标签
    if (!kTilingResult->loops.empty()) {
      kTilingResult->loops.front()->setAttr(
          "npu.loop_dim", rewriter.getStringAttr("K"));
      kTilingResult->loops.front()->setAttr(
          "npu.target", rewriter.getStringAttr("npu"));
    }

    // =================================================================
    // Phase 3: K 维度的 Head -> Body -> Tail Peeling (复用现有逻辑)
    // =================================================================
    auto kLoops = kTilingResult->loops;
    SmallVector<Value> finalKResults = kTilingResult->replacements;

    if (!kLoops.empty()) {
      auto loopOp = cast<scf::ForOp>(kLoops.back().getOperation());

      int64_t tripCount = getStaticTripCount(loopOp);
      if (tripCount == 1) {
        tagInnerComputeOp(loopOp, "single", rewriter);
      } else {
        scf::ForOp restLoop = loopOp;

        // --- Phase 3.1: Peel Head ---
        scf::ForOp headLoop;
        if (succeeded(peelForLoopFirstIteration(rewriter, loopOp, headLoop))) {
          inheritNpuAttributes(restLoop, headLoop);
          tagInnerComputeOp(headLoop, "head", rewriter);
          restLoop = loopOp;
        }

        int64_t restTripCount = getStaticTripCount(restLoop);

        // --- Phase 3.2: Peel Tail ---
        if (restTripCount == 1) {
          tagInnerComputeOp(restLoop, "tail", rewriter);
          finalKResults = restLoop->getResults();
        } else {
          scf::ForOp tailLoop;
          bool hasTail = false;

          scf::ForOp partialLoop;
          if (succeeded(scf::peelForLoopAndSimplifyBounds(
                  rewriter, restLoop, partialLoop))) {
            tailLoop = partialLoop;
            hasTail = true;
          } else {
            scf::ForOp forceTail;
            if (succeeded(
                    peelForLoopLastIteration(rewriter, restLoop, forceTail))) {
              tailLoop = forceTail;
              hasTail = true;
            }
          }

          if (hasTail) {
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

    // =================================================================
    // Phase 4: 链接数据流并替换原 Op
    // =================================================================
    // 1. 将空间 Tiling 生成的内部原始 Matmul 替换为 K 维度 Tiling 的结果
    rewriter.replaceOp(tiledMatmulOp, finalKResults);

    // 2. 将外层的原始 npu_matmul_integer 替换为空间 Tiling 的最终结果
    rewriter.replaceOp(op, spatialTilingResult->replacements);

    // =================================================================
    // Phase 5: 空间循环维度的常规尾部剥离 (Tail Peeling)
    // =================================================================
    for (int i = (int)spatialLoops.size() - 1; i >= 0; --i) {
      auto loopOp = cast<scf::ForOp>(spatialLoops[i].getOperation());
      scf::ForOp partialLoop;
      if (succeeded(scf::peelForLoopAndSimplifyBounds(
              rewriter, loopOp, partialLoop))) {
        inheritNpuAttributes(loopOp, partialLoop);
      }
    }

    return success();
  }
};

} // namespace

void npux::populateGemmTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuGemmTilingPattern,NpuStandaloneMatmulTilingPattern>(context);
}
