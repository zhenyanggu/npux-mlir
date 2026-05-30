//=============================================================================
// src/Conversion/NpuOpSplitting/NpuOpSplitting.cpp
// Implements tiling for Conv-related DMAs and Conv computation.
//=============================================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/StringRef.h"
#include <algorithm>

using namespace mlir;

namespace {
static int64_t getStaticTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());

  if (lb && ub && step) {
    return (int64_t)std::ceil((double)(*ub - *lb) / *step);
  }
  return -1;
}

static void tagInnerComputeOp(
    Operation *containerOp, StringRef phase, RewriterBase &rewriter) {
  containerOp->walk([&](Operation *op) {
    if (op->hasAttr("npu.split_done")) {
      // 按照要求修改了 Tag 名字
      op->setAttr("npu.split_stage", rewriter.getStringAttr(phase));
    }
  });
}

static StringRef getDmaKindName(npucore::DmaMvinOp) { return "npu_dma_mvin"; }

static StringRef getDmaKindName(npucore::DmaMvoutOp) { return "npu_dma_mvout"; }

static Value getDmaInputValue(npucore::DmaMvinOp op) {
  return op.getInputs().front();
}

static Value getDmaInputValue(npucore::DmaMvoutOp op) {
  return op.getInputs().front();
}

static Value getDmaOutputValue(npucore::DmaMvinOp op) {
  return op.getOutputs().front();
}

static Value getDmaOutputValue(npucore::DmaMvoutOp op) {
  return op.getOutputs().front();
}

//=============================================================================
// Pattern 1: NpuDmaTilingPattern
// Only target Conv-related Activation DMAs (5D, Dim 1 is Channel)
//=============================================================================
template <typename DmaOpTy>
struct NpuDmaTilingPattern : public OpRewritePattern<DmaOpTy> {
  using OpRewritePattern<DmaOpTy>::OpRewritePattern;

  // ============================================================================
  // 核心分析函数：analyzeAndTileDmaOp
  // 1. 推导 Slice 链条，记录历史切分维度
  // 2. 检测 SRAM Stride 溢出 (16-bit 限制)
  // 3. 拦截从未被切分但总大小超限的巨型 Tensor
  // 4. 返回精确的 Tile Sizes 和 连续块起始维度 (colDimIdx)
  // ============================================================================
  template <typename OpTy>
  static SmallVector<int64_t> analyzeAndTileDmaOp(
      OpTy op, StringRef libName, int &colDimIdx) {

    // 1. 尝试提取 SliceOp (不再强制要求必须有)
    tensor::ExtractSliceOp finalSliceOp = nullptr;
    if (libName == "npu_dma_mvin") {
      auto input = getDmaInputValue(op);
      finalSliceOp = input.template getDefiningOp<tensor::ExtractSliceOp>();
    } else if (libName == "npu_dma_mvout") {
      auto output = getDmaOutputValue(op);
      finalSliceOp = output.template getDefiningOp<tensor::ExtractSliceOp>();
    }

    auto inputType = cast<RankedTensorType>(getDmaInputValue(op).getType());
    ArrayRef<int64_t> currentShape = inputType.getShape();
    int rank = currentShape.size();

    // 2. 追溯 ExtractSlice 链条 (如果没有 SliceOp，isSplit 默认全 false)
    SmallVector<bool> isSplit(rank, false);
    if (finalSliceOp) {
      tensor::ExtractSliceOp currentSlice = finalSliceOp;
      while (currentSlice) {
        auto sourceType = cast<RankedTensorType>(currentSlice.getSourceType());
        int currentRank = sourceType.getRank();

        for (int i = 0; i < currentRank && i < rank; ++i) {
          if (isSplit[i])
            continue;

          bool split = false;
          if (currentSlice.isDynamicSize(i)) {
            split = true;
          } else {
            int64_t sliceSize = currentSlice.getStaticSize(i);
            int64_t sSize = sourceType.getShape()[i];
            if (sSize != ShapedType::kDynamic && sliceSize != sSize) {
              split = true;
            }
          }

          if (!split) {
            if (currentSlice.isDynamicOffset(i))
              split = true;
            else if (currentSlice.getStaticOffset(i) != 0)
              split = true;
          }

          if (split)
            isSplit[i] = true;
        }
        currentSlice =
            currentSlice.getSource().getDefiningOp<tensor::ExtractSliceOp>();
      }
    }

    int splitCount = 0;
    int idx1 = -1, idx2 = -1;
    for (int i = rank - 1; i >= 0; --i) {
      if (isSplit[i]) {
        splitCount++;
        if (splitCount == 1)
          idx1 = i;
        if (splitCount == 2) {
          idx2 = i;
          break;
        }
      }
    }

    SmallVector<int64_t> tileSizes(rank, 0);

    const int64_t LIMIT = 65535;

    if (idx1 != -1) {
      // 【情况 A】有上游切分：走 SRAM Stride 检测逻辑
      int64_t trailingElements = 1;
      bool isStatic = true;
      for (int i = idx1 + 1; i < rank; ++i) {
        if (currentShape[i] == ShapedType::kDynamic) {
          isStatic = false;
          break;
        }
        trailingElements *= currentShape[i];
      }

      if (isStatic && currentShape[idx1] != ShapedType::kDynamic) {
        int64_t sramStride = currentShape[idx1] * trailingElements;

        if (sramStride > LIMIT) {
          idx2 = idx1;
          idx1++;
        }
      }
    } else {
      // 【情况 B】没有任何切分 (完整 Tensor)：检查总容量是否超限
      int64_t totalSize = 1;
      bool isStatic = true;
      for (int i = 0; i < rank; ++i) {
        if (currentShape[i] == ShapedType::kDynamic) {
          isStatic = false;
          break;
        }
        totalSize *= currentShape[i];
      }

      if (isStatic && totalSize > LIMIT) {
        int64_t acc = 1;
        for (int i = rank - 1; i >= 0; --i) {
          if (acc * currentShape[i] > LIMIT) {
            idx1 = i;
            if (idx1 == rank - 1) {
              tileSizes[i] = LIMIT;
            } else {
              idx1++;
            }
            break;
          }
          acc *= currentShape[i];
        }
      }
    }

    for(int i = idx2-1; i>=0; --i) {
      tileSizes[i] = 1;
    }
    // 4. 打印信息与传出正确坐标
    llvm::errs() << "[Tiling] DMA " << libName << " Split Size: [";
    for (size_t i = 0; i < tileSizes.size(); ++i) {
      llvm::errs() << tileSizes[i] << (i == tileSizes.size() - 1 ? "" : ", ");
    }
    llvm::errs() << "]"<< "\n";
    colDimIdx = idx1;
    return tileSizes;
  }

  LogicalResult matchAndRewrite(
      DmaOpTy op, PatternRewriter &rewriter) const override {
    StringRef libName = getDmaKindName(op);
    if (libName.empty())
      return failure();

    if (!(libName == "npu_dma_mvin") && !(libName == "npu_dma_mvout"))
      return failure();

    // 1. 防止递归
    if (op->hasAttr("npu.split_done"))
      return failure();

    // 2. 将核心分析过程全部交给辅助函数
    int colDimIdx = -1;
    SmallVector<int64_t> splitSize =
        analyzeAndTileDmaOp(op, libName, colDimIdx);

    op->setAttr("npu.dma_col_dim", rewriter.getI32IntegerAttr(colDimIdx));

    // 检查是否需要分块 (空数组，或全为 0，则直接标记跳过)
    bool needSplit = llvm::any_of(splitSize, [](int64_t s) { return s > 0; });
    if (!needSplit) {
      // 保险起见，只有 DMA 才打标签退出
      if (libName.starts_with("npu_dma_mvin") ||
          libName.starts_with("npu_dma_mvout")) {
        op->setAttr("npu.split_done", rewriter.getUnitAttr());
      }
      return failure();
    }

    // ==============================================================
    // 3. 执行分块 (Tiling)
    // ==============================================================
    SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(rewriter.getContext(), splitSize);

    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(op.getOperation()), options);

    if (failed(tilingResult))
      return failure();

    // 在 Peel 之前打上完成标签，防止死循环
    for (auto *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.split_done", rewriter.getUnitAttr());
    }

    // ==============================================================
    // 4. 处理尾部 Peel (应对 Col 截断产生的余数)
    // ==============================================================
    auto loops = tilingResult->loops;
    SmallVector<Value> finalResults = tilingResult->replacements;

    for (int i = (int)loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp)
        continue;

      scf::ForOp partialIteration;
      LogicalResult status =
          scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
        if (i == 0) {
          finalResults = partialIteration->getResults();
        }
      }
    }

    // 5. 最终替换
    rewriter.replaceOp(op, finalResults);

    return success();
  }
};

struct NpuGemmTilingPattern : public RewritePattern {
  explicit NpuGemmTilingPattern(MLIRContext *context)
      : RewritePattern(Pattern::MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(
      Operation *operation, PatternRewriter &rewriter) const override {
    if (auto moveOp = dyn_cast<npucore::MvAccToSpmOp>(operation)) {
      if (moveOp->hasAttr("npu.split_done"))
        return failure();
      auto producerOp = moveOp.getInputs()[0].getDefiningOp<npucore::MatMulOp>();
      if (!producerOp)
        return failure();
      return handleTailFusion(moveOp, producerOp, rewriter);
    }

    auto matmulOp = dyn_cast<npucore::MatMulOp>(operation);
    if (!matmulOp || matmulOp->hasAttr("npu.split_done"))
      return failure();

    for (Operation *user : matmulOp.getResultTensors().front().getUsers()) {
      if (isa<npucore::MvAccToSpmOp>(user))
        return failure();
    }
    return handleSimpleGemmTiling(matmulOp, rewriter);
  }

private:
  // ----------------------------------------------------------------------------
  // 辅助函数：继承 Loop 属性，防止 Peel 出来的 Tail 循环丢失 Label
  // ----------------------------------------------------------------------------
  void inheritNpuAttributes(scf::ForOp source, scf::ForOp target) const {
    if (!source || !target)
      return;
    if (auto attr = source->getAttr("npu.split_dim"))
      target->setAttr("npu.split_dim", attr);
    if (auto attr = source->getAttr("npu.target"))
      target->setAttr("npu.target", attr);
  }

  // ----------------------------------------------------------------------------
  // 辅助函数：按照 mv_acc_to_spm 输出的物理维度顺序给空间循环打标签。
  // 2D 输出顺序为 [M, N]，3D 输出顺序为 [Batch, M, N]。
  // ----------------------------------------------------------------------------
  void labelMvAccToSpmSpatialLoops(ArrayRef<LoopLikeOpInterface> loops,
      npucore::MvAccToSpmOp moveOp, PatternRewriter &rewriter) const {
    int64_t outRank =
        cast<RankedTensorType>(moveOp.getOutputs().front().getType()).getRank();

    if (outRank == 2) {
      if (loops.size() >= 1)
        loops[0]->setAttr("npu.split_dim", rewriter.getStringAttr("M"));
      if (loops.size() >= 2)
        loops[1]->setAttr("npu.split_dim", rewriter.getStringAttr("N"));
      return;
    }

    if (outRank == 3) {
      if (loops.size() >= 1)
        loops[0]->setAttr("npu.split_dim", rewriter.getStringAttr("Batch"));
      if (loops.size() >= 2)
        loops[1]->setAttr("npu.split_dim", rewriter.getStringAttr("M"));
      if (loops.size() >= 3)
        loops[2]->setAttr("npu.split_dim", rewriter.getStringAttr("N"));
    }
  }

  // ----------------------------------------------------------------------------
  // 辅助函数：动态生成 [N, M, K] 分块大小并输出 Split 信息
  // ----------------------------------------------------------------------------
  SmallVector<int64_t> getGemmSplitSizes(Operation *op) const {
    int64_t rank = 0;
    SmallVector<int64_t> sizes(rank, 0);

    if (auto moveOp = dyn_cast<npucore::MvAccToSpmOp>(op)) {
      rank =
          cast<RankedTensorType>(moveOp.getOutputs().front().getType()).getRank();
      sizes.assign(rank, 0);
      if (rank == 2) {
        sizes[0] = 32;
        sizes[1] = 32;
      } else if (rank == 3) {
        sizes[1] = 32;
        sizes[2] = 32;
      }
      return sizes;
    }

    auto matmulOp = cast<npucore::MatMulOp>(op);
    rank = matmulOp.getLogicalLoopRank();
    sizes.assign(rank, 0);
    if (rank == 3) {
      sizes[0] = 32;
      sizes[1] = 32;
      sizes[2] = 2048;
    } else if (rank == 4) {
      sizes[1] = 32;
      sizes[2] = 32;
      sizes[3] = 2048;
    }

    {
      int64_t n_val = (rank == 4) ? sizes[1] : ((rank == 3) ? sizes[0] : 0);
      int64_t m_val = (rank == 4) ? sizes[2] : ((rank == 3) ? sizes[1] : 0);
      int64_t k_val = (rank == 4) ? sizes[3] : ((rank == 3) ? sizes[2] : 0);
      llvm::errs() << "[Spliting] Gemm: Tile=[N:" << n_val << ", M:" << m_val
                   << ", K:" << k_val << "]\n";
    }

    return sizes;
  }

  // ----------------------------------------------------------------------------
  // 场景 B 的实现：仅对 npu_gemm / npu_matmul 进行分块并 Peel
  // ----------------------------------------------------------------------------
  LogicalResult handleSimpleGemmTiling(
      npucore::MatMulOp op, PatternRewriter &rewriter) const {
    SmallVector<int64_t> staticTileSizes = getGemmSplitSizes(op.getOperation());

    SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(rewriter.getContext(), staticTileSizes);
    auto tilingOptions = scf::SCFTilingOptions().setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(op.getOperation()), tilingOptions);

    if (failed(tilingResult))
      return failure();

    auto loops = tilingResult->loops;
    if (loops.size() >= 1) {
      loops[0]->setAttr("npu.split_dim", rewriter.getStringAttr("N"));
    }
    if (loops.size() >= 2) {
      loops[1]->setAttr("npu.split_dim", rewriter.getStringAttr("M"));
    }
    if (loops.size() >= 3) {
      loops[2]->setAttr("npu.split_dim", rewriter.getStringAttr("K"));
    }

    rewriter.replaceOp(op, tilingResult->loops.front()->getResults());
    for (auto *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.split_done", rewriter.getUnitAttr());
    }

    if (loops.empty())
      return success();

    // 最内层(K)循环做 Head/Body/Tail 三段剥离，与 TailFusion 路径对齐。
    scf::ForOp innerLoop = cast<scf::ForOp>(loops.back().getOperation());
    int64_t tripCount = getStaticTripCount(innerLoop);
    if (tripCount == 1) {
      tagInnerComputeOp(innerLoop, "single", rewriter);
    } else {
      scf::ForOp restLoop = innerLoop;
      scf::ForOp headLoop;

      if (succeeded(peelForLoopFirstIteration(rewriter, innerLoop, headLoop))) {
        inheritNpuAttributes(restLoop, headLoop);
        tagInnerComputeOp(headLoop, "head", rewriter);
      }

      int64_t restTripCount = getStaticTripCount(restLoop);
      if (restTripCount == 1) {
        tagInnerComputeOp(restLoop, "tail", rewriter);
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
        } else {
          tagInnerComputeOp(restLoop, "body", rewriter);
        }
      }
    }

    // 对除最内层以外的循环进行边界剥离，消除动态维度 '?'。
    for (int i = (int)loops.size() - 2; i >= 0; --i) {
      auto loopOp = cast<scf::ForOp>(loops[i].getOperation());
      scf::ForOp partialLoop;
      if (succeeded(scf::peelForLoopAndSimplifyBounds(
              rewriter, loopOp, partialLoop))) {
        inheritNpuAttributes(loopOp, partialLoop);
      }
    }

    return success();
  }

  // ----------------------------------------------------------------------------
  // 场景 A 的实现：融合分块 + 内层K分块 + 全方位 Peel (含 Head/Body/Tail)
  // ----------------------------------------------------------------------------
  LogicalResult handleTailFusion(npucore::MvAccToSpmOp consumerOp,
      npucore::MatMulOp producerOp, PatternRewriter &rewriter) const {
    SmallVector<int64_t> staticTileSizes =
        getGemmSplitSizes(consumerOp.getOperation());

    SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(rewriter.getContext(), staticTileSizes);

    auto consumerTilingInterface =
        cast<TilingInterface>(consumerOp.getOperation());
    scf::SCFTileAndFuseOptions fuseOptions;
    fuseOptions.tilingOptions.setTileSizes(tileSizes);

    Operation *targetOpPtr = producerOp.getOperation();
    fuseOptions.setFusionControlFn(
        [targetOpPtr](tensor::ExtractSliceOp candidateSliceOp,
            OpResult originalProducer, bool isDestinationOperand)
            -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          if (originalProducer.getOwner() == targetOpPtr) {
            scf::SCFTileAndFuseOptions::ControlFnResult ctrlResult;
            ctrlResult.yieldProducerReplacement = true;
            return ctrlResult;
          }
          return std::nullopt;
        });

    FailureOr<scf::SCFTileAndFuseResult> fuseResult =
        scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, consumerTilingInterface, fuseOptions);

    if (failed(fuseResult))
      return failure();

    auto loops = fuseResult->loops;
    labelMvAccToSpmSpatialLoops(loops, consumerOp, rewriter);

    for (auto fusedOp : fuseResult->tiledAndFusedOps) {
      if (auto matmulOp = dyn_cast<npucore::MatMulOp>(fusedOp)) {
        SmallVector<int64_t> gemmSizes = getGemmSplitSizes(matmulOp.getOperation());
        for (size_t i = 0; i < gemmSizes.size() - 1; ++i)
          gemmSizes[i] = 0;

        SmallVector<OpFoldResult> kTileSizes =
            getAsIndexOpFoldResult(rewriter.getContext(), gemmSizes);
        auto kTilingOptions = scf::SCFTilingOptions().setTileSizes(kTileSizes);

        FailureOr<scf::SCFTilingResult> kTilingResult = scf::tileUsingSCF(
            rewriter, cast<TilingInterface>(matmulOp.getOperation()),
            kTilingOptions);
        if (failed(kTilingResult) || kTilingResult->loops.empty()) {
          matmulOp->setAttr("npu.split_done", rewriter.getUnitAttr());
          continue;
        }

        for (auto *kTiledOp : kTilingResult->tiledOps)
          kTiledOp->setAttr("npu.split_done", rewriter.getUnitAttr());

        scf::ForOp kLoop =
            cast<scf::ForOp>(kTilingResult->loops.front().getOperation());
        kLoop->setAttr("npu.split_dim", rewriter.getStringAttr("K"));

        SmallVector<Value> finalKResults = kTilingResult->replacements;
        int64_t tripCount = getStaticTripCount(kLoop);

        if (tripCount == 1) {
          tagInnerComputeOp(kLoop, "single", rewriter);
        } else {
          scf::ForOp restLoop = kLoop;
          scf::ForOp headLoop;

          if (succeeded(
                  peelForLoopFirstIteration(rewriter, kLoop, headLoop))) {
            inheritNpuAttributes(restLoop, headLoop);
            tagInnerComputeOp(headLoop, "head", rewriter);
          }

          int64_t restTripCount = getStaticTripCount(restLoop);
          if (restTripCount == 1) {
            tagInnerComputeOp(restLoop, "tail", rewriter);
            finalKResults = restLoop->getResults();
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
              finalKResults = tailLoop->getResults();
            } else {
              tagInnerComputeOp(restLoop, "body", rewriter);
              finalKResults = restLoop->getResults();
            }
          }
        }

        rewriter.replaceOp(matmulOp, finalKResults);
        continue;
      }

      fusedOp->setAttr("npu.split_done", rewriter.getUnitAttr());
    }

    if (fuseResult->replacements.count(consumerOp->getResult(0)))
      rewriter.replaceOp(
          consumerOp, fuseResult->replacements[consumerOp->getResult(0)]);

    // 空间循环边界清理 (N, M 维度的 Peeling)
    auto spatialLoops = fuseResult->loops;
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

//=============================================================================
// Pass Definition
//=============================================================================
struct NpuOpSplittingPass
    : public PassWrapper<NpuOpSplittingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuOpSplittingPass)
  llvm::StringRef getArgument() const override { return "npu-op-splitting"; }
  llvm::StringRef getDescription() const override {
    return "Splits Conv DMAs and Computes into hardware-aligned micro ops.";
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<NpuDmaTilingPattern<npucore::DmaMvinOp>>(context);
    patterns.add<NpuDmaTilingPattern<npucore::DmaMvoutOp>>(context);
    patterns.add<NpuGemmTilingPattern>(context);

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);

    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuOpSplittingPass() {
  return std::make_unique<NpuOpSplittingPass>();
}
