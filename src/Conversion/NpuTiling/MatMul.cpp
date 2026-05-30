//==============================================
// src/Conversion/NpuTiling/Gemm.cpp
// This file implements the tiling strategy of
// npucore.matmul + npucore.mv_acc_to_spm.
//==============================================

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include <cmath>

using namespace mlir;
using namespace npux;

namespace {

// -----------------------------------------------------------------------------
// Helper: 给 Loop 内部打了 "npu.tiled" 标记的 Op 添加 Phase 标签
// -----------------------------------------------------------------------------
static void tagInnerComputeOp(
    Operation *containerOp, StringRef phase, RewriterBase &rewriter) {
  containerOp->walk([&](Operation *op) {
    if (op->hasAttr("npu.tiled"))
      op->setAttr("npu.loop_stage", rewriter.getStringAttr(phase));
  });
}

// -----------------------------------------------------------------------------
// Helper: 估算静态 Loop 的迭代次数
// -----------------------------------------------------------------------------
static int64_t getStaticTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());

  if (lb && ub && step)
    return (int64_t)std::ceil((double)(*ub - *lb) / *step);
  return -1;
}

// -----------------------------------------------------------------------------
// Helper: 同步 Loop 属性，避免 Peel 后 Tail/Head 丢标签
// -----------------------------------------------------------------------------
static void inheritNpuAttributes(scf::ForOp source, scf::ForOp target) {
  if (!source || !target)
    return;
  if (auto attr = source->getAttr("npu.target"))
    target->setAttr("npu.target", attr);
  if (auto attr = source->getAttr("npu.loop_dim"))
    target->setAttr("npu.loop_dim", attr);
}

// -----------------------------------------------------------------------------
// Helper: 从 npucore.matmul 提取静态 M/N/K 维度并计算 Tile
// -----------------------------------------------------------------------------
static SmallVector<int64_t> getGemmTileSizes(npucore::MatMulOp op) {
  npux::HardwareConfig hwConfig;
  npux::NPUCostModel costModel(hwConfig);
  return costModel.getOptimalTileSizes(op);
}

// -----------------------------------------------------------------------------
// Helper: 将 MatMul 的逻辑 Tile [Batch, N, M, K] / [N, M, K] 映射为
// mv_acc_to_spm 输出的物理空间 Tile [Batch, M, N] / [M, N]
// -----------------------------------------------------------------------------
static SmallVector<int64_t> getMvAccToSpmSpatialTileSizes(
    ArrayRef<int64_t> logicalTileSizes) {
  if (logicalTileSizes.size() == 4)
    return {logicalTileSizes[0], logicalTileSizes[2], logicalTileSizes[1]};
  if (logicalTileSizes.size() == 3)
    return {logicalTileSizes[1], logicalTileSizes[0]};
  return {};
}

// -----------------------------------------------------------------------------
// Main Pattern for Gemm
// -----------------------------------------------------------------------------
struct NpuGemmTilingPattern : public OpRewritePattern<npucore::MvAccToSpmOp> {
  using OpRewritePattern<npucore::MvAccToSpmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::MvAccToSpmOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("npu.tiled"))
      return failure();
    if (op->getParentOfType<scf::ForOp>())
      return failure();

    Value input = op.getOperand(0);
    auto matmulOp = input.getDefiningOp<npucore::MatMulOp>();
    if (!matmulOp)
      return failure();

    SmallVector<int64_t> tileSizes = getGemmTileSizes(matmulOp);
    if (tileSizes.empty())
      return failure();

    int64_t rank = tileSizes.size();
    SmallVector<int64_t> spatialTileSizes =
        getMvAccToSpmSpatialTileSizes(tileSizes);
    if (spatialTileSizes.empty())
      return failure();
    SmallVector<int64_t> kTileSizes(rank, 0);
    if (rank == 4) {
      kTileSizes[3] = tileSizes[3];
    } else if (rank == 3) {
      kTileSizes[2] = tileSizes[2];
    } else {
      return failure();
    }

    auto consumerTilingInterface = cast<TilingInterface>(op.getOperation());
    scf::SCFTileAndFuseOptions fuseOptions;
    fuseOptions.tilingOptions.setTileSizes(
        getAsOpFoldResult(rewriter.getI64ArrayAttr(spatialTileSizes)));

    Operation *targetOpPtr = matmulOp.getOperation();
    fuseOptions.setFusionControlFn(
        [targetOpPtr](tensor::ExtractSliceOp candidateSliceOp,
            OpResult originalProducer, bool isDestinationOperand)
            -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          if (originalProducer.getOwner() == targetOpPtr)
            return scf::SCFTileAndFuseOptions::ControlFnResult{};
          return std::nullopt;
        });

    FailureOr<scf::SCFTileAndFuseResult> fuseResult =
        scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, consumerTilingInterface, fuseOptions);
    if (failed(fuseResult))
      return failure();

    auto spatialLoops = fuseResult->loops;
    int currentLoopIdx = 0;
    for (int64_t dimIdx = 0; dimIdx < rank - 1; ++dimIdx) {
      if (currentLoopIdx >= (int)spatialLoops.size())
        break;

      StringRef label;
      if (rank == 4) {
        if (dimIdx == 0)
          label = "Batch";
        else if (dimIdx == 1)
          label = "M";
        else
          label = "N";
      } else {
        label = dimIdx == 0 ? "M" : "N";
      }

      spatialLoops[currentLoopIdx]->setAttr(
          "npu.loop_dim", rewriter.getStringAttr(label));
      spatialLoops[currentLoopIdx]->setAttr(
          "npu.target", rewriter.getStringAttr("npu"));
      currentLoopIdx++;
    }

    auto tiledConsumer = fuseResult->tiledAndFusedOps.front();
    tiledConsumer->setAttr("npu.tiled", rewriter.getUnitAttr());

    auto fusedMatmulOp =
        tiledConsumer->getOperand(0).getDefiningOp<npucore::MatMulOp>();
    if (!fusedMatmulOp)
      return failure();
    fusedMatmulOp->setAttr("npu.tiled", rewriter.getUnitAttr());

    Location loc = fusedMatmulOp.getLoc();
    rewriter.setInsertionPoint(fusedMatmulOp);

    OpOperand *matmulOutOperand = fusedMatmulOp.getDpsInitOperand(0);
    Value matmulOutVal = matmulOutOperand->get();
    auto matmulOutType = cast<RankedTensorType>(matmulOutVal.getType());

    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < matmulOutType.getRank(); ++i) {
      if (matmulOutType.isDynamicDim(i))
        dynamicSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, matmulOutVal, i));
    }

    auto memSpaceAttr = rewriter.getI32IntegerAttr(3);
    auto accTensorType = RankedTensorType::get(
        matmulOutType.getShape(), matmulOutType.getElementType(), memSpaceAttr);

    auto accAlloc = rewriter.create<bufferization::AllocTensorOp>(
        loc, accTensorType, dynamicSizes);

    auto newMatmulOp = rewriter.create<npucore::MatMulOp>(loc,
        TypeRange{accTensorType}, fusedMatmulOp.getInputs(), ValueRange{accAlloc},
        fusedMatmulOp.getLhsScaleAttr(), fusedMatmulOp.getLhsZpAttr(),
        fusedMatmulOp.getRhsScaleAttr(), fusedMatmulOp.getRhsZpAttr(),
        fusedMatmulOp.getOutScaleAttr(), fusedMatmulOp.getOutZpAttr(),
        fusedMatmulOp.getWithBiasAttr(), fusedMatmulOp.getDoReluAttr(),
        fusedMatmulOp.getReluTypeAttr());
    for (NamedAttribute attr : fusedMatmulOp->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (name == "operandSegmentSizes" || name == "lhs_scale" ||
          name == "lhs_zp" || name == "rhs_scale" || name == "rhs_zp" ||
          name == "out_scale" || name == "out_zp" || name == "with_bias" ||
          name == "do_relu" || name == "relu_type")
        continue;
      newMatmulOp->setAttr(attr.getName(), attr.getValue());
    }
    rewriter.replaceOp(fusedMatmulOp, newMatmulOp.getResults());
    fusedMatmulOp = newMatmulOp;

    scf::SCFTilingOptions kOptions;
    kOptions.setTileSizes(
        getAsOpFoldResult(rewriter.getI64ArrayAttr(kTileSizes)));

    FailureOr<scf::SCFTilingResult> kTilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(fusedMatmulOp.getOperation()), kOptions);
    if (failed(kTilingResult))
      return failure();

    if (!kTilingResult->loops.empty()) {
      kTilingResult->loops.front()->setAttr(
          "npu.loop_dim", rewriter.getStringAttr("K"));
      kTilingResult->loops.front()->setAttr(
          "npu.target", rewriter.getStringAttr("npu"));
    }

    auto kLoops = kTilingResult->loops;
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
          inheritNpuAttributes(restLoop, headLoop);
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

    rewriter.replaceOp(fusedMatmulOp, finalKResults);

    Value originalResult = op->getResult(0);
    if (!fuseResult->replacements.count(originalResult))
      return failure();
    rewriter.replaceOp(op, fuseResult->replacements[originalResult]);

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
