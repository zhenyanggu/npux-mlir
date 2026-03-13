//=============================================================================
// src/Conversion/NpuOpSplitting/NpuOpSplitting.cpp
// Implements tiling for Conv-related DMAs and Conv computation.
//=============================================================================

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"

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

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {

    // 1. 防止递归
    if (op->hasAttr("npu.split_done"))
      return failure();

    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr)
      return failure();
    StringRef libName = libCallAttr.getValue();

    bool isMvin = libName.starts_with("npu_dma_mvin");
    bool isMvout = libName.starts_with("npu_dma_mvout");

    if (!isMvin && !isMvout)
      return failure();

    // 2. 获取 DMA 类型属性
    StringRef dmaType = "";
    if (auto typeAttr = op->getAttrOfType<StringAttr>("npu.dma_type")) {
      dmaType = typeAttr.getValue();
    }

    // ==============================================================
    // 逻辑：分块条件过滤
    // ==============================================================

    // 条件 1: 如果是 weight，不需要分块
    if (isMvin && dmaType == "weight") {
      op->setAttr("npu.split_done", rewriter.getUnitAttr());
      return failure();
    }

    // 条件 2: 对于 MVIN (Input)，检查其输入源是否为 tensor.extract_slice
    if (isMvin && dmaType == "input") {
      Value mvinSource = op.getInputs()[0];
      if (!mvinSource.getDefiningOp<tensor::ExtractSliceOp>()) {
        op->setAttr("npu.split_done", rewriter.getUnitAttr());
        return failure();
      }
    }

    // 条件 3: 对于 MVOUT，检查其输出目标是否为 tensor.extract_slice
    if (isMvout) {
      Value mvoutDest = op.getOutputs()[0];
      if (!mvoutDest.getDefiningOp<tensor::ExtractSliceOp>()) {
        op->setAttr("npu.split_done", rewriter.getUnitAttr());
        return failure();
      }
    }

    // ==============================================================
    // 修改点：根据维度 Rank 选择不同的 splitDim
    // ==============================================================
    auto inputType = cast<RankedTensorType>(op.getInputs()[0].getType());
    int rank = inputType.getRank();

    int splitDim = -1;

    if (rank == 4) {
      // 四维：对最高维（第0维）分块
      splitDim = 1;
    } else if (rank == 5) {
      // 五维：对第二位（第1维）分块
      splitDim = 1;
    } else {
      // 其他维度暂不处理
      op->setAttr("npu.split_done", rewriter.getUnitAttr());
      return failure();
    }

    // 3. 执行分块 (Tiling)
    // 只有选中的 splitDim 设置为 1，其余为 0（代表不在此维度切分）
    SmallVector<OpFoldResult> tileSizes(rank, rewriter.getIndexAttr(0));
    tileSizes[splitDim] = rewriter.getIndexAttr(1);

    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(op.getOperation()), options);

    if (failed(tilingResult))
      return failure();

    // 4. 替换并标记完成
    // 注意：linalg::GenericOp 通常只有一个输出结果
    rewriter.replaceOp(op, tilingResult->loops.front()->getResults());

    for (auto *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.split_done", rewriter.getUnitAttr());
    }

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
  SmallVector<int64_t> getDynamicTileSizes(linalg::GenericOp op, bool isSpatialOnly) const {
    auto loopRanges = op.getStaticLoopRanges();
    int64_t rank = loopRanges.size();
    SmallVector<int64_t> sizes(rank, 0);

    if (rank < 4) return sizes;

    int64_t h_orig = loopRanges[2];
    int64_t w_orig = loopRanges[3];

    int64_t h_tile = h_orig;
    int64_t w_tile = w_orig;

    if (h_orig * w_orig > 32) {
      if (w_orig <= 32) {
        h_tile = 32 / w_orig; // 优先切碎 H
        w_tile = w_orig;      // 保持 W 不变
        if (h_tile == 0) h_tile = 1; 
      } else {
        h_tile = 1;
        w_tile = 32;          // 被迫切 W
      }
    }

    sizes[1] = 1;       // Cout_c
    sizes[2] = h_tile;  // H
    sizes[3] = w_tile;  // W

    // 如果不是仅空间切分(即切分9D Conv)，则带上 Cin_c
    if (!isSpatialOnly && rank > 4) {
      sizes[4] = 1;     // Cin_c
    }
    llvm::errs() << "[Spliting] Conv: Tile=[OH:" << sizes[2] 
                 << ", OW:" << sizes[3] << "]\n";

    return sizes;
  }

  // ----------------------------------------------------------------------------
  // 辅助函数：统一打标签
  // ----------------------------------------------------------------------------
  void labelGeneratedLoops(ArrayRef<LoopLikeOpInterface> loops, StringRef libCall, 
                           PatternRewriter &rewriter) const {
    SmallVector<StringRef> labels = {"cout", "H", "W"};
    if (libCall == "npu_conv") {
      labels.push_back("cin");
    }
    for (size_t i = 0; i < loops.size() && i < labels.size(); ++i) {
      loops[i]->setAttr("npu.split_dim", rewriter.getStringAttr(labels[i]));
    }
  }

  bool needsSplit(linalg::GenericOp op, ArrayRef<int64_t> tileSizes) const {
    auto loopRanges = op.getStaticLoopRanges();
    for (size_t i = 0; i < loopRanges.size(); ++i) {
      if (tileSizes[i] > 0 && loopRanges[i] > tileSizes[i])
        return true;
    }
    return false;
  }

  // ----------------------------------------------------------------------------
  // 辅助函数：继承 Loop 属性，防止 Peel 出来的 Tail 循环丢失 Label
  // ----------------------------------------------------------------------------
  void inheritNpuAttributes(scf::ForOp source, scf::ForOp target) const {
    if (!source || !target) return;
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

    if (!needsSplit(op, staticTileSizes)) {
      op->setAttr("npu.split_done", rewriter.getUnitAttr());
      return failure();
    }

    SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(rewriter.getContext(), staticTileSizes);
    auto tilingOptions = scf::SCFTilingOptions().setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(op.getOperation()), tilingOptions);

    if (failed(tilingResult)) return failure();

    labelGeneratedLoops(tilingResult->loops, "npu_conv", rewriter);

    rewriter.replaceOp(op, tilingResult->loops.front()->getResults());
    for (auto *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.split_done", rewriter.getUnitAttr());
    }

    // [新增]: 对生成的空间循环进行边界剥离，消除动态维度 '?'
    auto loops = tilingResult->loops;
    for (int i = (int)loops.size() - 1; i >= 0; --i) {
      auto loopOp = cast<scf::ForOp>(loops[i].getOperation());
      scf::ForOp partialLoop;
      if (succeeded(scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialLoop))) {
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
    SmallVector<int64_t> spatialTileSizes = getDynamicTileSizes(consumerOp, true);

    if (!needsSplit(consumerOp, spatialTileSizes)) {
      consumerOp->setAttr("npu.split_done", rewriter.getUnitAttr());
      return failure();
    }

    SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(rewriter.getContext(), spatialTileSizes);

    auto consumerTilingInterface = cast<TilingInterface>(consumerOp.getOperation());
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

    if (failed(fuseResult)) return failure();

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

    if (!fusedConv) return failure();

    // Phase 3: 内层单独切分 Cin
    SmallVector<int64_t> cinTileSizes = {0, 0, 0, 0, 1}; 
    SmallVector<OpFoldResult> cinTileSizesOFR = 
        getAsIndexOpFoldResult(rewriter.getContext(), cinTileSizes);
    auto cinTilingOptions = scf::SCFTilingOptions().setTileSizes(cinTileSizesOFR);

    rewriter.setInsertionPoint(fusedConv);
    FailureOr<scf::SCFTilingResult> cinTilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(fusedConv.getOperation()), cinTilingOptions);

    if (failed(cinTilingResult) || cinTilingResult->loops.empty()) {
        fusedConv->setAttr("npu.split_done", rewriter.getUnitAttr());
        return success();
    }

    for (auto *op : cinTilingResult->tiledOps) {
         op->setAttr("npu.split_done", rewriter.getUnitAttr());
    }

    scf::ForOp cinLoop = cast<scf::ForOp>(cinTilingResult->loops.front().getOperation());
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
        if (succeeded(scf::peelForLoopAndSimplifyBounds(rewriter, restLoop, tailLoop))) {
          hasTail = true;
        } else if (succeeded(peelForLoopLastIteration(rewriter, restLoop, tailLoop))) {
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
      // 检查其 Producer 是否是 gemm/matmul
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
      // 场景 A: Tail / Single 阶段 (进行融合分块)
      // -----------------------------------------------------------
      return handleTailFusion(op, producerOp, rewriter);
    }

    if (libName == "npu_gemm" || libName == "npu_matmul") {
      // -----------------------------------------------------------
      // 场景 B: Head / Body 阶段 (仅对 Gemm 分块)
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
  static constexpr int64_t kMaxGemmKTile = 2048;

  bool isGemmLikeOp(linalg::GenericOp op) const {
    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr)
      return false;
    StringRef libName = libCallAttr.getValue();
    return libName == "npu_gemm" || libName == "npu_matmul";
  }

  void inheritLoopAttrs(scf::ForOp source, scf::ForOp target) const {
    if (!source || !target)
      return;
    if (auto attr = source->getAttr("npu.split_dim"))
      target->setAttr("npu.split_dim", attr);
    if (auto attr = source->getAttr("npu.target"))
      target->setAttr("npu.target", attr);
  }

  LogicalResult splitGemmOnKIfNeeded(
      linalg::GenericOp gemmOp, PatternRewriter &rewriter, bool &didSplit) const {
    didSplit = false;
    if (!isGemmLikeOp(gemmOp))
      return success();

    SmallVector<int64_t> loopRanges = gemmOp.getStaticLoopRanges();
    if (loopRanges.empty())
      return success();

    int64_t staticK = loopRanges.back();
    if (staticK == ShapedType::kDynamic || staticK <= kMaxGemmKTile)
      return success();

    int64_t rank = gemmOp.getNumLoops();
    SmallVector<int64_t> kTileSizes(rank, 0);
    kTileSizes.back() = kMaxGemmKTile;

    SmallVector<OpFoldResult> kTileSizesOfr =
        getAsIndexOpFoldResult(rewriter.getContext(), kTileSizes);
    auto kTilingOptions = scf::SCFTilingOptions().setTileSizes(kTileSizesOfr);

    rewriter.setInsertionPoint(gemmOp);
    FailureOr<scf::SCFTilingResult> kTilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(gemmOp.getOperation()), kTilingOptions);
    if (failed(kTilingResult))
      return failure();

    for (Operation *tiledOp : kTilingResult->tiledOps) {
      tiledOp->setAttr("npu.split_done", rewriter.getUnitAttr());
    }

    auto kLoops = kTilingResult->loops;
    if (!kLoops.empty()) {
      auto kLoop = cast<scf::ForOp>(kLoops.back().getOperation());
      kLoop->setAttr("npu.split_dim", rewriter.getStringAttr("K"));
      if (auto targetAttr = gemmOp->getAttr("npu.target"))
        kLoop->setAttr("npu.target", targetAttr);
    }

    SmallVector<Value> finalKResults = kTilingResult->replacements;
    if (!kLoops.empty()) {
      auto loopOp = cast<scf::ForOp>(kLoops.back().getOperation());
      int64_t tripCount = getStaticTripCount(loopOp);

      if (tripCount == 1) {
        tagInnerComputeOp(loopOp, "single", rewriter);
      } else {
        scf::ForOp restLoop = loopOp;

        scf::ForOp headLoop;
        if (succeeded(peelForLoopFirstIteration(rewriter, loopOp, headLoop))) {
          inheritLoopAttrs(restLoop, headLoop);
          tagInnerComputeOp(headLoop, "head", rewriter);
          restLoop = loopOp;
        }

        int64_t restTripCount = getStaticTripCount(restLoop);
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
            if (succeeded(peelForLoopLastIteration(
                    rewriter, restLoop, forceTail))) {
              tailLoop = forceTail;
              hasTail = true;
            }
          }

          if (hasTail) {
            inheritLoopAttrs(restLoop, tailLoop);
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

    rewriter.replaceOp(gemmOp, finalKResults);
    didSplit = true;
    return success();
  }

  // ----------------------------------------------------------------------------
  // 辅助函数：根据算子的 Rank 动态生成 [N, M] 分块大小
  // ----------------------------------------------------------------------------
  SmallVector<int64_t> getGemmSplitSizes(linalg::GenericOp op) const {
    int64_t rank = op.getNumLoops();
    SmallVector<int64_t> sizes(rank, 0);

    auto libCall = op->getAttrOfType<StringAttr>("library_call").getValue();

    if (libCall == "mv_acc_to_spm") {
      // mv_acc_to_spm 迭代空间是 [N, M] (Rank 2) 或 [B, N, M] (Rank 3)
      if (rank == 2) {
        sizes[0] = 32; // N
        sizes[1] = 32; // M
      } else if (rank == 3) {
        sizes[1] = 32; // N
        sizes[2] = 32; // M
      }
    } else {
      // npu_gemm 迭代空间是 [N, M, K] (Rank 3) 或 [B, N, M, K] (Rank 4)
      if (rank == 3) {
        sizes[0] = 32; // N
        sizes[1] = 32; // M
      } else if (rank == 4) {
        sizes[1] = 32; // N
        sizes[2] = 32; // M
      }
    }
    return sizes;
  }

  // ----------------------------------------------------------------------------
  // 辅助函数：检查是否真的需要切分 (如果 N 和 M 都 <= 32，就不切)
  // ----------------------------------------------------------------------------
  bool needsSplit(linalg::GenericOp op, ArrayRef<int64_t> tileSizes) const {
    auto loopRanges = op.getStaticLoopRanges();
    for (size_t i = 0; i < loopRanges.size(); ++i) {
      if (tileSizes[i] > 0 && loopRanges[i] > tileSizes[i])
        return true;
    }
    return false;
  }

  // ----------------------------------------------------------------------------
  // 场景 B 的实现：仅对 npu_gemm / npu_matmul 进行分块
  // ----------------------------------------------------------------------------
  LogicalResult handleSimpleGemmTiling(
      linalg::GenericOp op, PatternRewriter &rewriter) const {
    SmallVector<int64_t> staticTileSizes = getGemmSplitSizes(op);

    if (!needsSplit(op, staticTileSizes)) {
      bool didKSplit = false;
      if (failed(splitGemmOnKIfNeeded(op, rewriter, didKSplit)))
        return failure();
      if (didKSplit)
        return success();
      op->setAttr("npu.split_done", rewriter.getUnitAttr());
      return failure();
    }

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

    for (auto *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.split_done", rewriter.getUnitAttr());
    }

    linalg::GenericOp tiledGemmOp = nullptr;
    for (Operation *tiledOp : tilingResult->tiledOps) {
      auto genericOp = dyn_cast<linalg::GenericOp>(tiledOp);
      if (genericOp && isGemmLikeOp(genericOp)) {
        tiledGemmOp = genericOp;
        break;
      }
    }

    if (tiledGemmOp) {
      bool didKSplit = false;
      if (failed(splitGemmOnKIfNeeded(tiledGemmOp, rewriter, didKSplit)))
        return failure();
    }

    // 对 N/M 空间循环做边界剥离，尽量把尾块静态化（例如 40 -> 32 + 8）。
    // 这样后续 lowering 不会因为动态 memref 维度退化为 `?`。
    for (int i = (int)loops.size() - 1; i >= 0; --i) {
      auto loopOp = cast<scf::ForOp>(loops[i].getOperation());
      scf::ForOp partialLoop;
      if (succeeded(
              scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialLoop))) {
        inheritLoopAttrs(loopOp, partialLoop);
      }
    }

    rewriter.replaceOp(op, tilingResult->loops.front()->getResults());
    return success();
  }

  // ----------------------------------------------------------------------------
  // 场景 A 的实现：mv_acc_to_spm + gemm 融合分块
  // ----------------------------------------------------------------------------
  LogicalResult handleTailFusion(linalg::GenericOp consumerOp,
      linalg::GenericOp producerOp, PatternRewriter &rewriter) const {
    SmallVector<int64_t> staticTileSizes = getGemmSplitSizes(consumerOp);

    // 如果维度已经满足 <= 32，打上标签并跳过
    if (!needsSplit(consumerOp, staticTileSizes)) {
      bool didKSplit = false;
      if (failed(splitGemmOnKIfNeeded(producerOp, rewriter, didKSplit)))
        return failure();
      consumerOp->setAttr("npu.split_done", rewriter.getUnitAttr());
      if (!didKSplit)
        producerOp->setAttr("npu.split_done", rewriter.getUnitAttr());
      return didKSplit ? success() : failure();
    }

    SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(rewriter.getContext(), staticTileSizes);

    // 1. 初始化 Tile & Fuse 选项
    auto consumerTilingInterface =
        cast<TilingInterface>(consumerOp.getOperation());
    scf::SCFTileAndFuseOptions fuseOptions;
    fuseOptions.tilingOptions.setTileSizes(tileSizes);

    // 2. 告诉 MLIR 引擎：只要遇到我们的 producerOp (npu_gemm)，就把它融合进循环
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

    // 3. 一键执行：自动生成 2 层
    // scf.for，自动执行融合，完美处理所有数据流传递！
    FailureOr<scf::SCFTileAndFuseResult> fuseResult =
        scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, consumerTilingInterface, fuseOptions);

    if (failed(fuseResult))
      return failure();

    auto loops = fuseResult->loops;
    if (loops.size() >= 1) {
      // 最外层生成的一定是 N 的切分循环
      loops[0]->setAttr("npu.split_dim", rewriter.getStringAttr("N"));
    }
    if (loops.size() >= 2) {
      // 内层生成的一定是 M 的切分循环
      loops[1]->setAttr("npu.split_dim", rewriter.getStringAttr("M"));
    }

    // 4. 给新生成的算子打上完成标签
    linalg::GenericOp fusedGemmOp = nullptr;
    for (auto op : fuseResult->tiledAndFusedOps) {
      if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        if (isGemmLikeOp(genericOp))
          fusedGemmOp = genericOp;
        genericOp->setAttr("npu.split_done", rewriter.getUnitAttr());
      }
    }

    if (fusedGemmOp) {
      bool didKSplit = false;
      if (failed(splitGemmOnKIfNeeded(fusedGemmOp, rewriter, didKSplit)))
        return failure();
    }

    // 对融合后生成的 N/M 循环执行边界剥离，避免尾块以动态维度传播到后续阶段。
    for (int i = (int)loops.size() - 1; i >= 0; --i) {
      auto loopOp = cast<scf::ForOp>(loops[i].getOperation());
      scf::ForOp partialLoop;
      if (succeeded(
              scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialLoop))) {
        inheritLoopAttrs(loopOp, partialLoop);
      }
    }

    // 5. 替换外层的 Consumer 输出
    if (fuseResult->replacements.count(consumerOp->getResult(0))) {
      rewriter.replaceOp(
          consumerOp, fuseResult->replacements[consumerOp->getResult(0)]);
    }

    // 注: Producer(npu_gemm) 如果没有其他下游算子使用了，MLIR
    // 稍后会自动把它作为死代码(DCE)清理掉
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
