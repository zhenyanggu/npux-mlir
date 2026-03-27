//=============================================================================
// src/Conversion/NpuOpSplitting/NpuOpSplitting.cpp
// Implements tiling for Conv-related DMAs and Conv computation.
//=============================================================================

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <limits>

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

//=============================================================================
// Pattern 1: NpuDmaTilingPattern
// Only target Conv-related Activation DMAs (5D, Dim 1 is Channel)
//=============================================================================
struct NpuDmaTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  // ============================================================================
// 核心分析函数：analyzeAndTileDmaOp
// 1. 推导 Slice 链条，记录历史切分维度
// 2. 检测 SRAM Stride 溢出 (16-bit 限制)
// 3. 拦截从未被切分但总大小超限的巨型 Tensor
// 4. 返回精确的 Tile Sizes 和 连续块起始维度 (colDimIdx)
// ============================================================================
static SmallVector<int64_t> analyzeAndTileDmaOp(
    linalg::GenericOp op, StringRef libName, int &colDimIdx) {
  
  // 1. 尝试提取 SliceOp (不再强制要求必须有)
  tensor::ExtractSliceOp finalSliceOp = nullptr;
  if (libName == "npu_dma_mvin") {
    auto input = op.getInputs()[0];
    finalSliceOp = input.getDefiningOp<tensor::ExtractSliceOp>();
  } else if (libName == "npu_dma_mvout") {
    auto output = op.getOutputs()[0];
    finalSliceOp = output.getDefiningOp<tensor::ExtractSliceOp>();
  }

  auto inputType = cast<RankedTensorType>(op.getInputs()[0].getType());
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
        if (isSplit[i]) continue;

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
          if (currentSlice.isDynamicOffset(i)) split = true;
          else if (currentSlice.getStaticOffset(i) != 0) split = true;
        }

        if (split) isSplit[i] = true;
      }
      currentSlice = currentSlice.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    }
  }

  // 3. 寻找 idx1, idx2, idx3
  int splitCount = 0;
  int idx1 = -1, idx2 = -1, idx3 = -1;
  for (int i = rank - 1; i >= 0; --i) {
    if (isSplit[i]) {
      splitCount++;
      if (splitCount == 1) idx1 = i;
      if (splitCount == 2) idx2 = i;
      if (splitCount == 3) idx3 = i;
    }
  }

  SmallVector<int64_t> tileSizes(rank, 0);

  // 规则 1：寻三打平
  if (idx3 != -1) {
    for (int i = idx3; i >= 0; --i) tileSizes[i] = 1;
  }

  bool strideOverflow = false;

  // -----------------------------------------------------------
  // 规则 2 & 3：硬件溢出处理 (分有无上游切分两种情况)
  // -----------------------------------------------------------
  const int64_t LIMIT = 65535;

  if (idx1 != -1) {
    // 【情况 A】有上游切分：走 SRAM Stride 检测逻辑
    int64_t trailingElements = 1;
    bool isStatic = true;
    for (int i = idx1 + 1; i < rank; ++i) {
      if (currentShape[i] == ShapedType::kDynamic) {
        isStatic = false; break;
      }
      trailingElements *= currentShape[i];
    }

    if (isStatic && currentShape[idx1] != ShapedType::kDynamic) {
      int64_t sramStride = currentShape[idx1] * trailingElements;

      if (sramStride > LIMIT) {
        if (idx2 != -1) {
          strideOverflow = true;
          idx1++; // 降维移位
          for (int i = idx2; i >= 0; --i) tileSizes[i] = 1;
        } else {
          // 只有 idx1 被切分，且超出了 Col 限制
          strideOverflow = true;
          tileSizes[idx1] = std::max<int64_t>(1, LIMIT / trailingElements);
        }
      }
    }
  } else {
    // 【情况 B】没有任何切分 (完整 Tensor)：检查总容量是否超限
    int64_t totalSize = 1;
    bool isStatic = true;
    for (int i = 0; i < rank; ++i) {
      if (currentShape[i] == ShapedType::kDynamic) {
        isStatic = false; break;
      }
      totalSize *= currentShape[i];
    }

    if (isStatic && totalSize > LIMIT) {
      strideOverflow = true;
      // 从右向左找，找到第一个导致累乘超过 65535 的维度
      int64_t acc = 1;
      for (int i = rank - 1; i >= 0; --i) {
        if (acc * currentShape[i] > LIMIT) {
          idx1 = i;
          if(idx1 == rank - 1) {
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

  // 4. 打印信息与传出正确坐标
  llvm::errs() << "[Tiling] DMA " << libName << " Split Size: [";
  for (size_t i = 0; i < tileSizes.size(); ++i) {
    llvm::errs() << tileSizes[i] << (i == tileSizes.size() - 1 ? "" : ", ");
  }
  llvm::errs() << "]\n";

  if (strideOverflow) {
    llvm::errs() << "  -> Warning: SRAM Stride Overflow (> 65535)! Shifting Col dim inward.\n";
  }
  colDimIdx = idx1;
  return tileSizes;
}

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {

    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr)
      return failure();
    StringRef libName = libCallAttr.getValue();

    if (!(libName == "npu_dma_mvin") && !(libName == "npu_dma_mvout"))
      return failure();

    // 1. 防止递归
    if (op->hasAttr("npu.split_done"))
      return failure();

    // 2. 将核心分析过程全部交给辅助函数
    int colDimIdx = -1;
    SmallVector<int64_t> splitSize = analyzeAndTileDmaOp(op, libName, colDimIdx);

    op->setAttr("npu.dma_col_dim", rewriter.getI32IntegerAttr(colDimIdx));

    // 检查是否需要分块 (空数组，或全为 0，则直接标记跳过)
    bool needSplit = llvm::any_of(splitSize, [](int64_t s) { return s > 0; });
    if (!needSplit) {
      // 保险起见，只有 DMA 才打标签退出
      auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
      if (libCallAttr &&
          (libCallAttr.getValue().starts_with("npu_dma_mvin") ||
              libCallAttr.getValue().starts_with("npu_dma_mvout"))) {
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

//=============================================================================
// Pattern 2: NpuConvTilingPattern
// Tiles Conv on OC, inserts Bias Mvin and Acc2Spm
//=============================================================================
struct NpuConvTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {

    if (op->hasAttr("npu.split_done"))
      return failure();

    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr)
      return failure();
    StringRef libName = libCallAttr.getValue();

    // ==============================================================================
    // 1. 逻辑分叉点
    // ==============================================================================
    if (libName == "mv_acc_to_spm") {
      OpOperand *inputOperand = &op->getOpOperand(0);
      auto producerOp = inputOperand->get().getDefiningOp<linalg::GenericOp>();
      if (!producerOp)
        return failure();
      auto prodLibCall = producerOp->getAttrOfType<StringAttr>("library_call");
      if (!prodLibCall || prodLibCall.getValue() != "npu_conv") {
        return failure();
      }
      return handleTailFusion(op, producerOp, rewriter);
    }

    if (libName == "npu_conv") {
      for (Operation *user : op.getResult(0).getUsers()) {
        if (auto genericUser = dyn_cast<linalg::GenericOp>(user)) {
          auto attr = genericUser->getAttrOfType<StringAttr>("library_call");
          if (attr && attr.getValue() == "mv_acc_to_spm")
            return failure();
        }
      }
      return handleSimpleConvTiling(op, rewriter);
    }

    return failure();
  }

private:
  // ----------------------------------------------------------------------------
  // 辅助函数：根据 H * W <= 32 规则，动态生成切分大小
  // ----------------------------------------------------------------------------
  SmallVector<int64_t> getDynamicTileSizes(
      linalg::GenericOp op, bool isSpatialOnly) const {
    auto loopRanges = op.getStaticLoopRanges();
    int64_t rank = loopRanges.size();
    SmallVector<int64_t> sizes(rank, 0);

    if (rank < 4)
      return sizes;

    int64_t h_orig = loopRanges[2];
    int64_t w_orig = loopRanges[3];

    int64_t h_tile = h_orig;
    int64_t w_tile = w_orig;

    if (h_orig * w_orig > 32) {
      if (w_orig <= 32) {
        h_tile = 32 / w_orig; // 优先切碎 H
        w_tile = w_orig;      // 保持 W 不变
        if (h_tile == 0)
          h_tile = 1;
      } else {
        h_tile = 1;
        w_tile = 32; // 被迫切 W
      }
    }

    sizes[1] = 1;      // Cout_c
    sizes[2] = h_tile; // H
    sizes[3] = w_tile; // W

    // 如果不是仅空间切分(即切分9D Conv)，则带上 Cin_c
    if (!isSpatialOnly && rank > 4) {
      sizes[4] = 1; // Cin_c
    }
    llvm::errs() << "[Spliting] Conv: Tile=[OH:" << sizes[2]
                 << ", OW:" << sizes[3] << "]\n";

    return sizes;
  }

  // ----------------------------------------------------------------------------
  // 辅助函数：统一打标签
  // ----------------------------------------------------------------------------
  void labelGeneratedLoops(ArrayRef<LoopLikeOpInterface> loops,
      StringRef libCall, PatternRewriter &rewriter) const {
    SmallVector<StringRef> labels = {"cout", "H", "W"};
    if (libCall == "npu_conv") {
      labels.push_back("cin");
    }
    for (size_t i = 0; i < loops.size() && i < labels.size(); ++i) {
      loops[i]->setAttr("npu.split_dim", rewriter.getStringAttr(labels[i]));
    }
  }

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
  // 场景 B 的实现：直接切分 npu_conv (4维全切)
  // ----------------------------------------------------------------------------
  LogicalResult handleSimpleConvTiling(
      linalg::GenericOp op, PatternRewriter &rewriter) const {
    SmallVector<int64_t> staticTileSizes = getDynamicTileSizes(op, false);

    SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(rewriter.getContext(), staticTileSizes);
    auto tilingOptions = scf::SCFTilingOptions().setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(op.getOperation()), tilingOptions);

    if (failed(tilingResult))
      return failure();

    labelGeneratedLoops(tilingResult->loops, "npu_conv", rewriter);

    rewriter.replaceOp(op, tilingResult->loops.front()->getResults());
    for (auto *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.split_done", rewriter.getUnitAttr());
    }

    auto loops = tilingResult->loops;
    if (loops.empty())
      return success();

    // 最内层循环做 Head/Body/Tail 三段剥离，与 TailFusion 路径对齐。
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
  // 场景 A 的实现：两段式切分 + Cin 尾块剥离 + 空间尾块剥离
  // ----------------------------------------------------------------------------
  LogicalResult handleTailFusion(linalg::GenericOp consumerOp,
      linalg::GenericOp producerOp, PatternRewriter &rewriter) const {

    // Phase 1: 外层空间切分 (Cout, H, W)
    SmallVector<int64_t> spatialTileSizes =
        getDynamicTileSizes(consumerOp, true);

    SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(rewriter.getContext(), spatialTileSizes);

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

    labelGeneratedLoops(fuseResult->loops, "mv_acc_to_spm", rewriter);

    // Phase 2: 获取融合进内层的 npu_conv
    linalg::GenericOp fusedConv = nullptr;
    for (auto op : fuseResult->tiledAndFusedOps) {
      if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        auto libCall = genericOp->getAttrOfType<StringAttr>("library_call");
        if (libCall && libCall.getValue() == "npu_conv") {
          fusedConv = genericOp;
        } else {
          genericOp->setAttr("npu.split_done", rewriter.getUnitAttr());
        }
      }
    }

    if (!fusedConv)
      return failure();

    // Phase 3: 内层单独切分 Cin
    SmallVector<int64_t> cinTileSizes = {0, 0, 0, 0, 1};
    SmallVector<OpFoldResult> cinTileSizesOFR =
        getAsIndexOpFoldResult(rewriter.getContext(), cinTileSizes);
    auto cinTilingOptions =
        scf::SCFTilingOptions().setTileSizes(cinTileSizesOFR);

    rewriter.setInsertionPoint(fusedConv);
    FailureOr<scf::SCFTilingResult> cinTilingResult =
        scf::tileUsingSCF(rewriter,
            cast<TilingInterface>(fusedConv.getOperation()), cinTilingOptions);

    if (failed(cinTilingResult) || cinTilingResult->loops.empty()) {
      fusedConv->setAttr("npu.split_done", rewriter.getUnitAttr());
      return success();
    }

    for (auto *op : cinTilingResult->tiledOps) {
      op->setAttr("npu.split_done", rewriter.getUnitAttr());
    }

    scf::ForOp cinLoop =
        cast<scf::ForOp>(cinTilingResult->loops.front().getOperation());
    cinLoop->setAttr("npu.split_dim", rewriter.getStringAttr("cin"));

    // Phase 4: Cin Peeling (Head/Body/Tail) - 同步 Conv.cpp 逻辑
    SmallVector<Value> finalCinResults = cinTilingResult->replacements;
    int64_t tripCount = getStaticTripCount(cinLoop);

    if (tripCount == 1) {
      tagInnerComputeOp(cinLoop, "single", rewriter);
    } else {
      scf::ForOp restLoop = cinLoop;
      scf::ForOp headLoop;

      // 剥离 Head
      if (succeeded(peelForLoopFirstIteration(rewriter, cinLoop, headLoop))) {
        inheritNpuAttributes(restLoop, headLoop);
        tagInnerComputeOp(headLoop, "head", rewriter);
      }

      int64_t restTripCount = getStaticTripCount(restLoop);
      if (restTripCount == 1) {
        tagInnerComputeOp(restLoop, "tail", rewriter);
        finalCinResults = restLoop->getResults();
      } else {
        scf::ForOp tailLoop;
        bool hasTail = false;

        // 尝试剥离 Tail
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
          finalCinResults = tailLoop->getResults();
        } else {
          tagInnerComputeOp(restLoop, "body", rewriter);
          finalCinResults = restLoop->getResults();
        }
      }
    }

    // Phase 5: Final Replacement (使用更新后的 finalCinResults 替换内部 Conv)
    rewriter.replaceOp(fusedConv, finalCinResults);

    // 替换外层 consumer (mv_acc_to_spm) 的输出
    if (fuseResult->replacements.count(consumerOp->getResult(0))) {
      rewriter.replaceOp(
          consumerOp, fuseResult->replacements[consumerOp->getResult(0)]);
    }

    // Phase 6: 空间循环边界清理 (Spatial Peeling)
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

struct NpuGemmTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {

    if (op->hasAttr("npu.split_done"))
      return failure();

    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr)
      return failure();
    StringRef libName = libCallAttr.getValue();

    // ==============================================================================
    // 1. 逻辑分叉点
    // ==============================================================================
    if (libName == "mv_acc_to_spm") {
      OpOperand *inputOperand = &op->getOpOperand(0);
      auto producerOp = inputOperand->get().getDefiningOp<linalg::GenericOp>();
      if (!producerOp)
        return failure();
      auto prodLibCall = producerOp->getAttrOfType<StringAttr>("library_call");
      if (!prodLibCall || (prodLibCall.getValue() != "npu_gemm" &&
                              prodLibCall.getValue() != "npu_matmul")) {
        return failure();
      }
      // -----------------------------------------------------------
      // 场景 A: Tail / Single 阶段 (进行融合分块 + Peeling)
      // -----------------------------------------------------------
      return handleTailFusion(op, producerOp, rewriter);
    }

    if (libName == "npu_gemm" || libName == "npu_matmul") {
      // -----------------------------------------------------------
      // 场景 B: Head / Body 阶段 (仅对 Gemm 分块 + Peeling)
      // -----------------------------------------------------------
      for (Operation *user : op.getResult(0).getUsers()) {
        if (auto genericUser = dyn_cast<linalg::GenericOp>(user)) {
          auto attr = genericUser->getAttrOfType<StringAttr>("library_call");
          if (attr && attr.getValue() == "mv_acc_to_spm")
            return failure();
        }
      }
      return handleSimpleGemmTiling(op, rewriter);
    }

    return failure();
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
  // 辅助函数：动态生成 [N, M, K] 分块大小并输出 Split 信息
  // ----------------------------------------------------------------------------
  SmallVector<int64_t> getGemmSplitSizes(linalg::GenericOp op) const {
    int64_t rank = op.getNumLoops();
    SmallVector<int64_t> sizes(rank, 0);

    auto libCall = op->getAttrOfType<StringAttr>("library_call").getValue();

    if (libCall == "mv_acc_to_spm") {
      if (rank == 2) {
        sizes[0] = 32; // N
        sizes[1] = 32; // M
      } else if (rank == 3) {
        sizes[1] = 32; // N
        sizes[2] = 32; // M
      }
    } else {
      if (rank == 3) {
        sizes[0] = 32;   // N
        sizes[1] = 32;   // M
        sizes[2] = 2048; // K
      } else if (rank == 4) {
        sizes[1] = 32;   // N
        sizes[2] = 32;   // M
        sizes[3] = 2048; // K
      }
    }

    // [新增]: 像 Conv 一样输出 Split 信息
    if (libCall != "mv_acc_to_spm") {
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
      linalg::GenericOp op, PatternRewriter &rewriter) const {
    SmallVector<int64_t> staticTileSizes = getGemmSplitSizes(op);

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
  LogicalResult handleTailFusion(linalg::GenericOp consumerOp,
      linalg::GenericOp producerOp, PatternRewriter &rewriter) const {
    SmallVector<int64_t> staticTileSizes = getGemmSplitSizes(consumerOp);

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
    if (loops.size() >= 1) {
      loops[0]->setAttr("npu.split_dim", rewriter.getStringAttr("N"));
    }
    if (loops.size() >= 2) {
      loops[1]->setAttr("npu.split_dim", rewriter.getStringAttr("M"));
    }

    for (auto op : fuseResult->tiledAndFusedOps) {
      if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        auto attr = genericOp->getAttrOfType<StringAttr>("library_call");
        bool isGemm = attr && (attr.getValue() == "npu_gemm" ||
                                  attr.getValue() == "npu_matmul");

        if (isGemm) {
          SmallVector<int64_t> gemmSizes = getGemmSplitSizes(genericOp);
          for (size_t i = 0; i < gemmSizes.size() - 1; ++i) {
            gemmSizes[i] = 0; // 仅保留 K 维度的切分
          }

          SmallVector<OpFoldResult> kTileSizes =
              getAsIndexOpFoldResult(rewriter.getContext(), gemmSizes);
          auto kTilingOptions =
              scf::SCFTilingOptions().setTileSizes(kTileSizes);

          FailureOr<scf::SCFTilingResult> kTilingResult = scf::tileUsingSCF(
              rewriter, cast<TilingInterface>(genericOp.getOperation()),
              kTilingOptions);

          // [新增]: K 维度切分失败或为空的提早退出
          if (failed(kTilingResult) || kTilingResult->loops.empty()) {
            genericOp->setAttr("npu.split_done", rewriter.getUnitAttr());
            continue;
          }

          // 打上 split_done tag
          for (auto *kTiledOp : kTilingResult->tiledOps) {
            kTiledOp->setAttr("npu.split_done", rewriter.getUnitAttr());
          }

          scf::ForOp kLoop =
              cast<scf::ForOp>(kTilingResult->loops.front().getOperation());
          kLoop->setAttr("npu.split_dim", rewriter.getStringAttr("K"));

          // [新增]: K Dimension Peeling (Head/Body/Tail) - 对齐 Conv 的 Cin
          SmallVector<Value> finalKResults = kTilingResult->replacements;
          int64_t tripCount = getStaticTripCount(kLoop);

          if (tripCount == 1) {
            tagInnerComputeOp(kLoop, "single", rewriter);
          } else {
            scf::ForOp restLoop = kLoop;
            scf::ForOp headLoop;

            // 剥离 Head
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

              // 尝试剥离 Tail
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

          // Final Replacement (使用更新后的 finalKResults 替换内部 Gemm)
          rewriter.replaceOp(genericOp, finalKResults);
          continue;
        }
        genericOp->setAttr("npu.split_done", rewriter.getUnitAttr());
      }
    }

    if (fuseResult->replacements.count(consumerOp->getResult(0))) {
      rewriter.replaceOp(
          consumerOp, fuseResult->replacements[consumerOp->getResult(0)]);
    }

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

    patterns.add<NpuDmaTilingPattern>(context);
    patterns.add<NpuConvTilingPattern>(context);
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
