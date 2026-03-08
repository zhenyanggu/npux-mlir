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

  // 硬件参数
  const int64_t arraySizeH = 32;
  const int64_t arraySizeW = 32;
  const int64_t inputDtypeBytes = 1;  // int8 (A 和 B 输入)
  const int64_t outputDtypeBytes = 1; // int8 (输出到 SPM)
  const int64_t accDtypeBytes = 4;    // int32 (ACC 累加器)

  // 辅助 Lambda：向下取整到 32 的倍数，但最小不低于 32
  auto align_down_32 = [&](int64_t val) -> int64_t {
    return std::max<int64_t>(32, (val / 32) * 32);
  };

  int64_t m_aligned = align_down_32(M);
  int64_t n_aligned = align_down_32(N);

  // ---------------------------------------------------------
  // 核心策略: 优先最大化 K，然后 M，最后 N 
  // 目标: 尽可能将 A 矩阵 (M x K) 完整驻留在 SPM 中
  // ---------------------------------------------------------

  // ==========================================
  // Step 1: 最大化 Tk (K 维度无 32 对齐约束)
  // ==========================================
  // 为了探求 Tk 的绝对物理上限，我们假设 Tm 和 Tn 取硬件支持的最小值 (32x32)
  // SPM 约束公式: Tm*Tn*1 + Tk*(Tm+Tn)*1 <= spmSize
  int64_t min_tm = arraySizeH;
  int64_t min_tn = arraySizeW;
  int64_t base_out_spm = min_tm * min_tn * outputDtypeBytes;
  
  int64_t max_tk_spm = 1;
  if (spmSize > base_out_spm) {
    max_tk_spm = (spmSize - base_out_spm) / ((min_tm + min_tn) * inputDtypeBytes);
  }
  // Tk 尽可能取到 K，且无需对齐
  int64_t t_k = std::max<int64_t>(1, std::min(K, max_tk_spm));

  // ==========================================
  // Step 2: 在固定 Tk 的前提下，最大化 Tm (必须是 32 的倍数)
  // ==========================================
  // 同样，为了让 Tm 最大，我们假设优先级最低的 Tn 取最小值 (32)
  // 1. ACC 约束: Tm * 32 * 4 <= accSize
  int64_t max_tm_acc = accSize / (min_tn * accDtypeBytes);
  
  // 2. SPM 约束: Tm*32*out + Tk*(Tm+32)*in <= spmSize
  // 推导 -> Tm * (32*out + Tk*in) <= spmSize - 32 * Tk * in
  int64_t max_tm_spm = arraySizeH; 
  int64_t spm_rem_for_m = spmSize - min_tn * t_k * inputDtypeBytes; 
  if (spm_rem_for_m > 0) {
    max_tm_spm = spm_rem_for_m / (min_tn * outputDtypeBytes + t_k * inputDtypeBytes);
  }
  
  // 综合 M 自身大小、ACC 限制和 SPM 限制
  int64_t t_m = std::min({m_aligned, max_tm_acc, max_tm_spm});
  t_m = align_down_32(t_m);

  // ==========================================
  // Step 3: 在固定 Tk 和 Tm 的前提下，计算剩余的 Tn (必须是 32 的倍数)
  // ==========================================
  // 1. ACC 约束: Tm * Tn * 4 <= accSize
  int64_t max_tn_acc = accSize / (t_m * accDtypeBytes);
  
  // 2. SPM 约束: Tm*Tn*out + Tk*(Tm+Tn)*in <= spmSize
  // 推导 -> Tn * (Tm*out + Tk*in) <= spmSize - Tk * Tm * in
  int64_t max_tn_spm = arraySizeW;
  int64_t spm_rem_for_n = spmSize - t_k * t_m * inputDtypeBytes;
  if (spm_rem_for_n > 0) {
    max_tn_spm = spm_rem_for_n / (t_m * outputDtypeBytes + t_k * inputDtypeBytes);
  }

  int64_t t_n = std::min({n_aligned, max_tn_acc, max_tn_spm});
  t_n = align_down_32(t_n);

  return {t_m, t_n, t_k};
}

SmallVector<int64_t> getGemmTileSizes(linalg::GenericOp op) {
  auto &config = npux::NPUConfig::getInstance();

  // 1. 获取手动配置 (这里假设用户的配置还是按 Tm, Tn, Tk 填写的)
  std::vector<int64_t> manualSizes = config.getMatMulTileSize();

  // 2. 获取 Loop Ranges
  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  int64_t rank = loopRanges.size(); // 2D Gemm 为 3，3D Batched Gemm 为 4

  if (rank != 3 && rank != 4) {
    return {};
  }

  // 根据新的迭代器顺序 [Batch, N, M, K] 或 [N, M, K] 提取维度
  int64_t K = loopRanges[rank - 1];
  int64_t M = loopRanges[rank - 2];
  int64_t N = loopRanges[rank - 3];
  int64_t B = (rank == 4) ? loopRanges[0] : 1; // 如果是 3D 的，提取 Batch

  SmallVector<int64_t, 3> computedSizes;
  bool isManual = false;

  if (!manualSizes.empty() && manualSizes.size() >= 3) {
    computedSizes = {manualSizes[0], manualSizes[1], manualSizes[2]};
    isManual = true;
  } else {
    int64_t spmSize = config.getSpmSize();
    int64_t accSize = config.getAccSize();
    // 自动分块逻辑依然基于 M, N, K 计算 Tm, Tn, Tk，无需改变
    computedSizes = calculateAutoGemmTile(M, N, K, spmSize, accSize);
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
  if (rank == 4) {
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
      if (rank == 4) { // [Batch, N, M, K]
        if (dimIdx == 0) label = "Batch";
        else if (dimIdx == 1) label = "N";
        else if (dimIdx == 2) label = "M";
      } else { // rank == 3, [N, M, K]
        if (dimIdx == 0) label = "N";
        else if (dimIdx == 1) label = "M";
      }

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