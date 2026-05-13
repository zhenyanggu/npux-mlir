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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"

#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include <algorithm>
#include <cmath> // for ceil
#include <limits>
#include <optional>

using namespace mlir;
using namespace npux;

namespace {

llvm::cl::opt<std::string> gemmTilingStrategy(
    "gemm-tiling-strategy",
    llvm::cl::desc(
        "GEMM tiling strategy: legacy (current heuristic) or costmodel"),
    llvm::cl::init("legacy"));

llvm::cl::opt<std::string> gemmFusionStrategy(
    "gemm-fusion-strategy",
    llvm::cl::desc(
        "GEMM tiling fusion strategy: legacy (fuse first compute only) or "
        "extended (fuse until the second NPU compute op)"),
    llvm::cl::init("extended"));

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

static bool isNpuComputeBoundary(Operation *op) {
  auto genericOp = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!genericOp)
    return false;

  auto libCall = genericOp->getAttrOfType<StringAttr>("library_call");
  if (!libCall)
    return false;

  StringRef name = libCall.getValue();
  return name == "npu_conv" || name == "npu_gemm" || name == "npu_matmul" || name == "npu_maxpool";
}

static bool useLegacyFusionStrategy() {
  std::string strategy = gemmFusionStrategy;
  strategy = llvm::StringRef(strategy).trim().lower();
  return strategy == "legacy";
}

static Operation *getSingleUser(Value value) {
  Operation *singleUser = nullptr;
  for (Operation *user : value.getUsers()) {
    if (singleUser)
      return nullptr;
    singleUser = user;
  }
  return singleUser;
}

static linalg::GenericOp findDownstreamFusionConsumer(
    linalg::GenericOp firstConsumer) {
  if (useLegacyFusionStrategy())
    return firstConsumer;

  linalg::GenericOp consumer = firstConsumer;
  while (consumer->getNumResults() == 1) {
    Operation *user = getSingleUser(consumer->getResult(0));
    auto nextConsumer = dyn_cast_or_null<linalg::GenericOp>(user);
    if (!nextConsumer || nextConsumer->hasAttr("npu.tiled"))
      break;
    if (isNpuComputeBoundary(nextConsumer))
      break;
    consumer = nextConsumer;
  }

  return consumer;
}

static linalg::GenericOp findFusedNpuCompute(
    Operation *root, llvm::function_ref<bool(StringRef)> isTargetCompute) {
  llvm::SmallPtrSet<Operation *, 16> visited;
  SmallVector<Operation *, 8> worklist;

  auto addDefiningGenericOp = [&](Value value) {
    if (auto genericOp = value.getDefiningOp<linalg::GenericOp>())
      worklist.push_back(genericOp.getOperation());
  };

  for (Value operand : root->getOperands())
    addDefiningGenericOp(operand);

  while (!worklist.empty()) {
    Operation *candidate = worklist.pop_back_val();
    if (!visited.insert(candidate).second)
      continue;

    auto genericOp = cast<linalg::GenericOp>(candidate);
    if (auto libCall = genericOp->getAttrOfType<StringAttr>("library_call")) {
      if (isTargetCompute(libCall.getValue()))
        return genericOp;
    }

    for (Value operand : candidate->getOperands())
      addDefiningGenericOp(operand);
  }

  return nullptr;
}

static std::optional<scf::SCFTileAndFuseOptions::ControlFnResult>
fuseBackToCurrentNpuCompute(Operation *currentComputeOp,
    OpResult originalProducer) {
  Operation *producerOp = originalProducer.getOwner();
  if (!isa<linalg::GenericOp>(producerOp))
    return std::nullopt;

  if (producerOp == currentComputeOp)
    return scf::SCFTileAndFuseOptions::ControlFnResult{};

  if (useLegacyFusionStrategy())
    return std::nullopt;

  if (isNpuComputeBoundary(producerOp))
    return std::nullopt;

  return scf::SCFTileAndFuseOptions::ControlFnResult{};
}

static void markAllFusedOpsAsTiled(
    const scf::SCFTileAndFuseResult &fuseResult, RewriterBase &rewriter) {
  for (Operation *fusedOp : fuseResult.tiledAndFusedOps) {
    if (!fusedOp)
      continue;
    fusedOp->setAttr("npu.tiled", rewriter.getUnitAttr());
  }
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

static SmallVector<int64_t, 3> calculateAutoGemmTile(
    int64_t M, int64_t N, int64_t K, int64_t spmSize, int64_t accSize) {

  const int64_t arraySizeH = 32;
  const int64_t arraySizeW = 32;
  const int64_t inputDtypeBytes = 1; 
  const int64_t outputDtypeBytes = 1;
  const int64_t accDtypeBytes = 4;   

  auto alignDownToArray = [&](int64_t val) -> int64_t {
    if (val <= 0)
      return 1;
    if (val < arraySizeH)
      return val;
    return std::max<int64_t>(arraySizeH, (val / arraySizeH) * arraySizeH);
  };

  // Legacy baseline: keep the output tile at one systolic-array footprint and
  // only stretch K as far as the local memories allow. This intentionally avoids
  // the previous capacity-maximizing heuristic so that "legacy" is a simple
  // non-cost-model baseline for ablation.
  int64_t t_m = std::min<int64_t>(std::max<int64_t>(1, M), arraySizeH);
  int64_t t_n = std::min<int64_t>(std::max<int64_t>(1, N), arraySizeW);

  int64_t maxTkSpm = 1;
  int64_t baseSpm = t_m * t_n * outputDtypeBytes;
  int64_t kBytesPerStep = (t_m + t_n) * inputDtypeBytes;
  if (spmSize > baseSpm && kBytesPerStep > 0)
    maxTkSpm = (spmSize - baseSpm) / kBytesPerStep;

  int64_t maxTkAcc = std::numeric_limits<int64_t>::max();
  int64_t accBytes = t_m * t_n * accDtypeBytes;
  if (accBytes > accSize) {
    int64_t maxTnByAcc = std::max<int64_t>(1, accSize / (t_m * accDtypeBytes));
    t_n = std::min(t_n, maxTnByAcc);
    baseSpm = t_m * t_n * outputDtypeBytes;
    kBytesPerStep = (t_m + t_n) * inputDtypeBytes;
    maxTkSpm = (spmSize > baseSpm && kBytesPerStep > 0)
                   ? (spmSize - baseSpm) / kBytesPerStep
                   : 1;
  }

  int64_t t_k = std::max<int64_t>(1, std::min({K, maxTkSpm, maxTkAcc}));
  t_k = alignDownToArray(t_k);

  return {t_m, t_n, t_k};
}

struct GemmCostResult {
  int64_t tN = 0;
  int64_t tM = 0;
  int64_t tK = 0;
  double latencyMs = std::numeric_limits<double>::infinity();
  double spmUtil = 0.0;
  double accUtil = 0.0;
  int64_t stage1KTiles = 0;
  int64_t stage2KTiles = 0;
  int64_t dmaTrafficBytes = 0;
};

static int64_t ceilDiv(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

static int64_t alignGemmTile(int64_t value, int64_t arraySize = 32) {
  if (value <= 0)
    return 0;
  if (value < arraySize)
    return value;
  return std::max<int64_t>(arraySize, (value / arraySize) * arraySize);
}

static SmallVector<int64_t> buildGemmTileCandidates(
    int64_t dim, int64_t step = 32) {
  SmallVector<int64_t> candidates;
  if (dim <= 0)
    return candidates;
  if (dim < step) {
    candidates.push_back(dim);
    return candidates;
  }

  for (int64_t v = step; v <= dim; v += step)
    candidates.push_back(v);

  int64_t alignedDim = alignGemmTile(dim, step);
  if (alignedDim > 0 &&
      std::find(candidates.begin(), candidates.end(), alignedDim) ==
          candidates.end())
    candidates.push_back(alignedDim);

  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
      candidates.end());
  return candidates;
}

static std::optional<GemmCostResult> evaluateGemmTileCost(int64_t B, int64_t N,
    int64_t M, int64_t K, bool hasBias, int64_t spmSize, int64_t accSize,
    int64_t tN, int64_t tM, int64_t tK) {
  constexpr int64_t dtypeInputBytes = 1;
  constexpr int64_t dtypeAccBytes = 4;
  constexpr int64_t gemmNmSplit = 32;
  constexpr int64_t gemmKSplitLimit = 2048;
  constexpr double measuredBandwidthMbps = 1300.0;
  constexpr double measuredGemmUs = 5.0;
  constexpr double instrOverheadUs = 1.54;

  if (std::min({tN, tM, tK}) <= 0)
    return std::nullopt;

  int64_t sizeATile = tN * tK * dtypeInputBytes;
  int64_t sizeBTile = tK * tM * dtypeInputBytes;
  int64_t sizeOutTile = tN * tM * dtypeInputBytes;
  int64_t totalSpmNeeded = sizeATile + sizeBTile + sizeOutTile;
  int64_t sizeAccNeeded = tN * tM * dtypeAccBytes;

  if (totalSpmNeeded > spmSize || sizeAccNeeded > accSize)
    return std::nullopt;

  int64_t nNTiles = ceilDiv(N, tN);
  int64_t nMTiles = ceilDiv(M, tM);
  int64_t nKStage1 = ceilDiv(K, tK);
  int64_t outputTiles = B * nNTiles * nMTiles;
  int64_t totalStage1Tiles = outputTiles * nKStage1;

  int64_t effectiveK = std::min(tK, gemmKSplitLimit);
  int64_t nKStage2 = ceilDiv(K, std::max<int64_t>(1, effectiveK));

  int64_t nNMicro = ceilDiv(tN, gemmNmSplit);
  int64_t nMMicro = ceilDiv(tM, gemmNmSplit);
  int64_t nKMicro = ceilDiv(effectiveK, gemmNmSplit);
  int64_t opsPerComputeCall = nNMicro * nMMicro * nKMicro;
  int64_t totalComputeCalls = outputTiles * nKStage2;
  int64_t totalAtomicOps = totalComputeCalls * opsPerComputeCall;

  int64_t totalDmaIfmBytes = totalStage1Tiles * sizeATile;
  int64_t totalDmaWgtBytes = totalStage1Tiles * sizeBTile;
  int64_t totalDmaBiasBytes = hasBias ? outputTiles * (tM * dtypeAccBytes) : 0;
  int64_t totalDmaOfmBytes = outputTiles * sizeOutTile;
  int64_t totalTrafficBytes =
      totalDmaIfmBytes + totalDmaWgtBytes + totalDmaBiasBytes + totalDmaOfmBytes;

  int64_t numDmaInstructions = totalStage1Tiles * 2 + outputTiles;
  if (hasBias)
    numDmaInstructions += outputTiles;

  double latDmaUs =
      numDmaInstructions * instrOverheadUs +
      (static_cast<double>(totalTrafficBytes) / measuredBandwidthMbps);
  double latComputeUs = totalAtomicOps * measuredGemmUs;

  GemmCostResult result;
  result.tN = tN;
  result.tM = tM;
  result.tK = tK;
  result.latencyMs = (latDmaUs + latComputeUs) / 1000.0;
  result.spmUtil = static_cast<double>(totalSpmNeeded) / spmSize;
  result.accUtil = static_cast<double>(sizeAccNeeded) / accSize;
  result.stage1KTiles = nKStage1;
  result.stage2KTiles = nKStage2;
  result.dmaTrafficBytes = totalTrafficBytes;
  return result;
}

static SmallVector<int64_t, 3> calculateCostModelGemmTile(int64_t B, int64_t N,
    int64_t M, int64_t K, bool hasBias, int64_t spmSize, int64_t accSize,
    std::optional<GemmCostResult> &bestResult) {
  SmallVector<int64_t> nCandidates = buildGemmTileCandidates(N);
  SmallVector<int64_t> mCandidates = buildGemmTileCandidates(M);
  SmallVector<int64_t> kCandidates = buildGemmTileCandidates(K);

  std::optional<GemmCostResult> best;
  for (int64_t tN : nCandidates) {
    for (int64_t tM : mCandidates) {
      for (int64_t tK : kCandidates) {
        auto result = evaluateGemmTileCost(
            B, N, M, K, hasBias, spmSize, accSize, tN, tM, tK);
        if (!result)
          continue;
        if (!best || result->latencyMs < best->latencyMs ||
            (result->latencyMs == best->latencyMs &&
                result->dmaTrafficBytes < best->dmaTrafficBytes)) {
          best = *result;
        }
      }
    }
  }

  bestResult = best;
  if (best)
    return {best->tM, best->tN, best->tK};

  return calculateAutoGemmTile(M, N, K, spmSize, accSize);
}

SmallVector<int64_t> getGemmTileSizes(linalg::GenericOp op) {
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
  std::optional<GemmCostResult> costModelResult;
  bool isManual = false;
  bool isCostModel = false;

  if (!manualSizes.empty() && manualSizes.size() >= 3) {
    // 第一阶段仅按容量分块，硬件 2048 限制在第二阶段 (npu-op-splitting) 处理。
    computedSizes = {manualSizes[0], manualSizes[1], manualSizes[2]};
    isManual = true;
  } else {
    int64_t spmSize = config.getSpmSize();
    int64_t accSize = config.getAccSize();
    std::string strategy = gemmTilingStrategy;
    strategy = llvm::StringRef(strategy).trim().lower();
    if (strategy == "costmodel") {
      auto withBiasAttr = op->getAttrOfType<IntegerAttr>("with_bias");
      bool hasBias = withBiasAttr && withBiasAttr.getInt() != 0;
      computedSizes = calculateCostModelGemmTile(
          B, N, M, K, hasBias, spmSize, accSize, costModelResult);
      isCostModel = costModelResult.has_value();
    } else {
      // 自动分块逻辑依然基于 M, N, K 计算 Tm, Tn, Tk，无需改变
      computedSizes = calculateAutoGemmTile(M, N, K, spmSize, accSize);
    }
  }

  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "Tiling [Gemm] ("
     << (isManual ? "Manual" : (isCostModel ? "CostModel" : "LegacyAuto"))
     << "): "
     << "Problem=[B:" << B << ", N:" << N << ", M:" << M << ", K:" << K << "] "
     << "-> Tile=[Tm:" << computedSizes[0] << ", Tn:" << computedSizes[1]
     << ", Tk:" << computedSizes[2] << "]";
  if (costModelResult) {
    os << " cost=[latency_ms:" << costModelResult->latencyMs
       << ", spm_util:" << costModelResult->spmUtil
       << ", acc_util:" << costModelResult->accUtil
       << ", k_stage1_tiles:" << costModelResult->stage1KTiles
       << ", k_stage2_tiles:" << costModelResult->stage2KTiles
       << ", dma_bytes:" << costModelResult->dmaTrafficBytes << "]";
  }
  os << "\n";
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
    // Phase 1: 从当前 GEMM 后的 mv_acc_to_spm 开始向下找到融合终点，
    // 再沿着空间维度 Tiling 该终点，并向上融合回当前 npu_gemm。
    // =================================================================
    linalg::GenericOp fusionConsumerOp = findDownstreamFusionConsumer(op);
    auto consumerTilingInterface =
        cast<TilingInterface>(fusionConsumerOp.getOperation());
    scf::SCFTileAndFuseOptions fuseOptions;
    fuseOptions.tilingOptions.setTileSizes(
        getAsOpFoldResult(rewriter.getI64ArrayAttr(spatialTileSizes)));

    Operation *targetOpPtr = gemmOp.getOperation();
    fuseOptions.setFusionControlFn(
        [targetOpPtr](tensor::ExtractSliceOp candidateSliceOp,
            OpResult originalProducer, bool isDestinationOperand)
            -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          return fuseBackToCurrentNpuCompute(targetOpPtr, originalProducer);
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
    markAllFusedOpsAsTiled(*fuseResult, rewriter);
    auto tiledConsumer = fuseResult->tiledAndFusedOps.front();

    auto fusedGemmOp = findFusedNpuCompute(tiledConsumer,
        [](StringRef name) {
          return name == "npu_gemm" || name == "npu_matmul";
        });
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

    // 2. 将外层原始融合终点替换为 Tile & Fuse 流程的最终产物。
    Value originalResult = fusionConsumerOp->getResult(0);
    if (fuseResult->replacements.count(originalResult)) {
      rewriter.replaceOp(
          fusionConsumerOp, fuseResult->replacements[originalResult]);
      if (fusionConsumerOp != op && op->use_empty())
        rewriter.eraseOp(op);
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

} // namespace

void npux::populateGemmTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuGemmTilingPattern>(context);
}
