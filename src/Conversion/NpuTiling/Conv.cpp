//=============================================================
// src/Conversion/NpuTiling/Conv.cpp
// this file is for Conv op tiling pattern
//=============================================================

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include <cmath> // for ceil

using namespace mlir;
using namespace npux;
namespace {

// 1. 更新维度标签：现在是 5 维 (NCHWc32 中的外层 5 维)
// d0:N, d1:OC, d2:OH, d3:OW, d4:IC
static const SmallVector<StringRef> kDimensionLabels = {
    "N", "OC", "OH", "OW", "IC"};

    
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
  // 复制你关心的所有属性
  if (auto attr = source->getAttr("npu.target"))
    target->setAttr("npu.target", attr);
  if (auto attr = source->getAttr("npu.loop_dim"))
    target->setAttr("npu.loop_dim", attr);
}

SmallVector<int64_t> getConvTileSizes(npucore::ConvOp op) {
  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  return costModel.getOptimalTileSizes(op);
}

struct NpuConvTilingPattern : public OpRewritePattern<npucore::MvAccToSpmOp> {
  using OpRewritePattern<npucore::MvAccToSpmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::MvAccToSpmOp op, PatternRewriter &rewriter) const override {

    if (op->hasAttr("npu.tiled"))
      return failure();
    if (op->getParentOfType<scf::ForOp>())
      return failure();

    Value input = op->getOperand(0);
    auto convOp = input.getDefiningOp<npucore::ConvOp>();
    if (!convOp) {
      return failure();
    }

    // 2. 获取 TileSizes，现在长度应为 5 [N, OC, OH, OW, IC]
    SmallVector<int64_t> tileSizes = getConvTileSizes(convOp);
    if (tileSizes.size() < 5)
      return failure();

    // 3. 重新划分 Spatial (N, OC, OH, OW) 和 IC 维度
    SmallVector<int64_t> spatialTileSizes = {
        tileSizes[0], tileSizes[1], tileSizes[2], tileSizes[3],
        0, 0, 0, 0, 0};
    SmallVector<int64_t> icTileSizes = {
        0, 0, 0, 0, tileSizes[4], 0, 0, 0, 0};

    // Phase 1: Spatial Tiling & Fusion
    auto consumerTilingInterface = cast<TilingInterface>(op.getOperation());
    scf::SCFTileAndFuseOptions fuseOptions;
    fuseOptions.tilingOptions.setTileSizes(
        getAsOpFoldResult(rewriter.getI64ArrayAttr(spatialTileSizes)));

    Operation *targetOpPtr = convOp.getOperation();
    fuseOptions.setFusionControlFn(
        [targetOpPtr](tensor::ExtractSliceOp candidateSliceOp,
            OpResult originalProducer, bool isDestinationOperand)
            -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          if (originalProducer.getOwner() == targetOpPtr) {
            return scf::SCFTileAndFuseOptions::ControlFnResult{};
          }
          return std::nullopt;
        });

    FailureOr<scf::SCFTileAndFuseResult> fuseResult =
        scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, consumerTilingInterface, fuseOptions);

    if (failed(fuseResult))
      return failure();

    // 4. 给空间循环打标签 (增加到 4 层循环: N, OC, OH, OW)
    auto spatialLoops = fuseResult->loops;
    for (size_t i = 0; i < std::min(spatialLoops.size(), (size_t)4); ++i) {
      spatialLoops[i]->setAttr(
          "npu.loop_dim", rewriter.getStringAttr(kDimensionLabels[i]));
      spatialLoops[i]->setAttr("npu.target", rewriter.getStringAttr("npu"));
    }

    // Phase 2: 处理融合后的 Conv
    auto tiledConsumer = fuseResult->tiledAndFusedOps.front();
    tiledConsumer->setAttr("npu.tiled", rewriter.getUnitAttr());

    auto fusedConvOp =
        tiledConsumer->getOperand(0).getDefiningOp<npucore::ConvOp>();
    if (!fusedConvOp)
      return failure();
    fusedConvOp->setAttr("npu.tiled", rewriter.getUnitAttr());

    Location loc = fusedConvOp.getLoc();
    rewriter.setInsertionPoint(fusedConvOp);

    // 分配 ACC 内存 (Space 3)，rank 5 自动处理
    OpOperand *convOutOperand = fusedConvOp.getDpsInitOperand(0);
    Value convOutVal = convOutOperand->get();
    auto convOutType = cast<RankedTensorType>(convOutVal.getType());

    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < convOutType.getRank(); ++i) {
      if (convOutType.isDynamicDim(i)) {
        dynamicSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, convOutVal, i));
      }
    }

    auto memSpaceAttr = rewriter.getI32IntegerAttr(3);
    auto accTensorType = RankedTensorType::get(
        convOutType.getShape(), convOutType.getElementType(), memSpaceAttr);

    auto accAlloc = rewriter.create<bufferization::AllocTensorOp>(
        loc, accTensorType, dynamicSizes);

    auto newConvOp = rewriter.create<npucore::ConvOp>(loc,
        TypeRange{accTensorType}, fusedConvOp.getInputs(), ValueRange{accAlloc},
        fusedConvOp.getInScaleAttr(), fusedConvOp.getInZpAttr(),
        fusedConvOp.getWScaleAttr(), fusedConvOp.getWZpAttr(),
        fusedConvOp.getOutScaleAttr(), fusedConvOp.getOutZpAttr(),
        fusedConvOp.getPadsAttr(), fusedConvOp.getStridesAttr(),
        fusedConvOp.getDilationsAttr(), fusedConvOp.getGroupAttr(),
        fusedConvOp.getDoReluAttr(), fusedConvOp.getReluTypeAttr());
    for (NamedAttribute attr : fusedConvOp->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (name == "operandSegmentSizes" || name == "in_scale" ||
          name == "in_zp" || name == "w_scale" || name == "w_zp" ||
          name == "out_scale" || name == "out_zp" || name == "pads" ||
          name == "strides" || name == "dilations" || name == "group" ||
          name == "do_relu" || name == "relu_type")
        continue;
      newConvOp->setAttr(attr.getName(), attr.getValue());
    }
    rewriter.replaceOp(fusedConvOp, newConvOp.getResults());
    fusedConvOp = newConvOp;

    // Phase 3: IC Tiling
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

    // Phase 4: IC Peeling (Head/Body/Tail)
    auto icLoops = icTilingResult->loops;
    SmallVector<Value> finalIcResults = icTilingResult->replacements;

    if (!icLoops.empty()) {
      auto loopOp = cast<scf::ForOp>(icLoops.back().getOperation());
      // 确保原始 IC Loop 有属性 (如果 Phase 3 没给它打 npu.target，这里补一下)
      loopOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

      int64_t tripCount = getStaticTripCount(loopOp);
      if (tripCount == 1) {
        tagInnerComputeOp(loopOp, "single", rewriter);
      } else {
        scf::ForOp restLoop = loopOp;
        scf::ForOp headLoop;
        if (succeeded(peelForLoopFirstIteration(rewriter, loopOp, headLoop))) {
          // 【关键】同步属性到 headLoop
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
            // 【关键】同步属性到 tailLoop
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

    // Phase 5: Final Replacement
    rewriter.replaceOp(fusedConvOp, finalIcResults);
    Value originalResult = op->getResult(0);
    if (!fuseResult->replacements.count(originalResult))
      return rewriter.notifyMatchFailure(
          op, "missing replacement for tiled npucore.mv_acc_to_spm");
    rewriter.replaceOp(op, fuseResult->replacements[originalResult]);

    // 对空间循环进行边界清理 (Peeling)
    for (int i = (int)spatialLoops.size() - 1; i >= 0; --i) {
      auto loopOp = cast<scf::ForOp>(spatialLoops[i].getOperation());
      scf::ForOp partialLoop;
      if (succeeded(scf::peelForLoopAndSimplifyBounds(
              rewriter, loopOp, partialLoop))) {
        // 【重要】如果不在这里 inheritNpuAttributes，Spatial 的 Tail
        // 部分也会丢失 npu.target
        inheritNpuAttributes(loopOp, partialLoop);
      }
    }

    return success();
  }
};

} // namespace

void npux::populateConvTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuConvTilingPattern>(context);
};
