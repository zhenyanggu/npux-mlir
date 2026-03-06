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
  if (!source || !target) return;
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


static SmallVector<int64_t, 3> calculateAutoGemmTile(
    int64_t M, int64_t N, int64_t K, int64_t spmSize, int64_t accSize) {

  // 硬件对齐参数
  const int64_t arraySizeH = 32;
  const int64_t arraySizeW = 32;
  const int64_t inputDtypeBytes = 1;  // int8 (输入)
  const int64_t outputDtypeBytes = 1; // int8 (输出到SPM也是int8)
  const int64_t accDtypeBytes = 4;    // int32 (ACC累加)

  // ---------------------------------------------------------
  // Step 1: 初始估计 Tm 和 Tn (基于 ACC 容量)
  // ---------------------------------------------------------
  int64_t maxAccElem = accSize / accDtypeBytes;

  int64_t targetDim = std::floor(std::sqrt(maxAccElem));

  // M 维度初步分块
  int64_t t_m = std::min(M, targetDim);
  if (t_m >= arraySizeH)
    t_m = (t_m / arraySizeH) * arraySizeH;
  else
    t_m = arraySizeH;

  // N 维度初步分块
  int64_t t_n = std::min(N, maxAccElem / t_m);
  if (t_n >= arraySizeW)
    t_n = (t_n / arraySizeW) * arraySizeW;
  else
    t_n = arraySizeW;

  // ---------------------------------------------------------
  // Step 2: 联合调整 Tm, Tn, Tk (基于 ACC 和 SPM 容量)
  // ---------------------------------------------------------
  // 这里需要循环，因为如果 SPM 放不下 (Input + Output)，
  // 我们需要缩小 Tm/Tn 来腾出空间。
  int64_t t_k = arraySizeH; // 初始设为最小对齐单位

  while (true) {
    // 1. 检查 ACC 限制 (Accumulator overflow check)
    // ---------------------------------------------
    bool accFits = (t_m * t_n * accDtypeBytes) <= accSize;

    // 2. 检查 SPM 限制 (SPM overflow check)
    // ---------------------------------------------
    // Output 占用: Tm * Tn * 1 byte
    int64_t outputSpmBytes = t_m * t_n * outputDtypeBytes;

    // Input 单位 K 占用: (Tm + Tn) * 1 byte
    int64_t inputBytesPerK = (t_m + t_n) * inputDtypeBytes;

    // 计算 SPM 中剩余给 Input 的空间
    int64_t remainingSpmForInput = spmSize - outputSpmBytes;

    // 至少要能放下一个最小单位的 Tk (arraySizeH)
    bool spmFits = (remainingSpmForInput >= (inputBytesPerK * arraySizeH));

    // 3. 如果 ACC 或 SPM 爆了，缩小 Tm/Tn
    // ---------------------------------------------
    if (!accFits || !spmFits) {
      t_n -= arraySizeW; // 优先缩减 N
      if (t_n < arraySizeW) {
        t_n = arraySizeW;
        t_m -= arraySizeH; // N 缩无可缩，缩 M
      }

      // 保护机制：如果连最小块都放不下（极少见），强制退出
      if (t_m < arraySizeH) {
        t_m = arraySizeH;
        t_n = arraySizeW;
        break;
      }
      continue; // 重新检查新的 Tm/Tn
    }

    // 4. 计算最终的 Tk
    // ---------------------------------------------
    // 到这里说明 Tm, Tn 既符合 ACC，也给 SPM 留出了至少 32*K 的空间
    int64_t maxTk = remainingSpmForInput / inputBytesPerK;
    t_k = std::min(K, maxTk);

    // Tk 对齐
    if (t_k >= arraySizeH)
      t_k = (t_k / arraySizeH) * arraySizeH;
    else
      t_k = arraySizeH;

    // 成功找到合适的分块
    break;
  }

  return {t_m, t_n, t_k};
}

SmallVector<int64_t> getGemmTileSizes(linalg::GenericOp op) {
  auto &config = npux::NPUConfig::getInstance();

  // 1. 获取手动配置
  std::vector<int64_t> manualSizes = config.getMatMulTileSize();

  // 2. 获取 Loop Ranges
  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  int64_t rank = loopRanges.size();

  if (rank < 3) {
    return {};
  }

  // 提取 M, N, K
  int64_t K = loopRanges[rank - 1];
  int64_t N = loopRanges[rank - 2];
  int64_t M = loopRanges[rank - 3];

  SmallVector<int64_t, 3> computedSizes;
  bool isManual = false;

  if (!manualSizes.empty() && manualSizes.size() >= 3) {
    computedSizes = {manualSizes[0], manualSizes[1], manualSizes[2]};
    isManual = true;
  } else {
    int64_t spmSize = config.getSpmSize();
    int64_t accSize = config.getAccSize();
    computedSizes = calculateAutoGemmTile(M, N, K, spmSize, accSize);
  }

  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "Tiling [Gemm] (" << (isManual ? "Manual" : "Auto") << "): "
     << "Problem=[M:" << M << ", N:" << N << ", K:" << K << "] "
     << "-> Tile=[Tm:" << computedSizes[0] << ", Tn:" << computedSizes[1]
     << ", Tk:" << computedSizes[2] << "]\n";
  llvm::errs() << os.str();

  // 3. 构建最终的 Tile Sizes 数组
  SmallVector<int64_t> finalTileSizes(rank, 0);

  // [..., tm, tn, tk]
  finalTileSizes[rank - 3] = computedSizes[0]; // M -> tm
  finalTileSizes[rank - 2] = computedSizes[1]; // N -> tn
  finalTileSizes[rank - 1] = computedSizes[2]; // K -> tk

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

      StringRef label = "Batch";
      if (dimIdx == rank - 2)
        label = "N";
      else if (dimIdx == rank - 3)
        label = "M";

      spatialLoops[currentLoopIdx]->setAttr(
          "npu.loop_dim", rewriter.getStringAttr(label));
      spatialLoops[currentLoopIdx]->setAttr("npu.target", rewriter.getStringAttr("npu"));
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
        loc, accTensorType, dynamicSizes, Value{}, memSpaceAttr);

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
      kTilingResult->loops.front()->setAttr("npu.target", rewriter.getStringAttr("npu"));
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

} // namespace

void npux::populateGemmTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuGemmTilingPattern>(context);
}