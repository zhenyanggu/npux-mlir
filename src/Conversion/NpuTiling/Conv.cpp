//=============================================================
// src/Conversion/NpuTiling/Conv.cpp
// this file is for Conv op tiling pattern
//=============================================================

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
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

SmallVector<int64_t> getConvTileSizes(linalg::GenericOp op) {
  unsigned rank = 0;
  if (!op.getOutputs().empty()) {
    if (auto type = dyn_cast<RankedTensorType>(op.getOutputs()[0].getType())) {
      rank = type.getRank();
    }
  }
  // Conv 在 NCHWc32 下通常是 5 维
  if (rank < 5) return {};

  auto &config = npux::NPUConfig::getInstance();
  
  // 1. 优先级最高：CLI/JSON 传入的强制手工配置
  std::vector<int64_t> manualSizes = config.getConvTileSize();
  if (!manualSizes.empty() && manualSizes.size() >= 4) {
    SmallVector<int64_t> sizes(rank, 0);
    int64_t t_oh = manualSizes[0];
    int64_t t_ow = manualSizes[1];
    int64_t t_ic = manualSizes[2];
    int64_t t_oc = manualSizes[3];

    // 映射到 NCHWc32: [N, OC_outer, OH, OW, IC_outer]
    sizes[0] = 1;                                 
    sizes[1] = (t_oc > 32) ? (t_oc / 32) : 1;     
    sizes[2] = t_oh;                              
    sizes[3] = t_ow;                              
    sizes[4] = (t_ic > 32) ? (t_ic / 32) : 1;     

    llvm::errs() << "[Tiling] Conv (Manual): Tile=[OH:" << t_oh << ", OW:" << t_ow 
                 << ", IC_blk:" << t_ic << ", OC_blk:" << t_oc << "]\n";
    return sizes;
  }

  // 2. 使用内置 Cost-Model 动态计算
  npux::HardwareConfig hwConfig;

  npux::NPUCostModel costModel(hwConfig);
  SmallVector<int64_t> optimalSizes = costModel.getOptimalTileSizes(op);
  return optimalSizes;
}

struct NpuConvTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {

    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (!libCall || libCall.getValue() != "mv_acc_to_spm")
      return failure();
    if (op->hasAttr("npu.tiled"))
      return failure();

    Value input = op->getOperand(0);
    auto convOp = input.getDefiningOp<linalg::GenericOp>();
    if (!convOp || !convOp->hasAttr("library_call") ||
        convOp->getAttrOfType<StringAttr>("library_call").getValue() !=
            "npu_conv") {
      return failure();
    }

    // 2. 获取 TileSizes，现在长度应为 5 [N, OC, OH, OW, IC]
    SmallVector<int64_t> tileSizes = getConvTileSizes(convOp);
    if (tileSizes.size() < 5)
      return failure();

    // 3. 重新划分 Spatial (N, OC, OH, OW) 和 IC 维度
    SmallVector<int64_t> spatialTileSizes = {
        tileSizes[0], tileSizes[1], tileSizes[2], tileSizes[3]};
    SmallVector<int64_t> icTileSizes = {
        0, 0, 0, 0, tileSizes[4]}; // IC 维度在第 5 位

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
        tiledConsumer->getOperand(0).getDefiningOp<linalg::GenericOp>();
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

    auto newConvOp = rewriter.create<linalg::GenericOp>(loc,
        TypeRange{accTensorType}, fusedConvOp.getInputs(), ValueRange{accAlloc},
        fusedConvOp.getIndexingMapsArray(),
        fusedConvOp.getIteratorTypesArray());

    rewriter.inlineRegionBefore(fusedConvOp.getRegion(), newConvOp.getRegion(),
        newConvOp.getRegion().begin());
    newConvOp->setAttrs(fusedConvOp->getAttrs());
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
    if (fuseResult->replacements.count(originalResult)) {
      rewriter.replaceOp(op, fuseResult->replacements[originalResult]);
    }

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