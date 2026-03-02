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
#include <cmath> // for ceil

using namespace mlir;
using namespace npux;

namespace {

// 定义维度名称映射
// d0:N, d1:OC, d2:OH, d3:OW, d4:IC
static const SmallVector<StringRef> kDimensionLabels = {"OC", "OH", "OW", "IC"};

// -----------------------------------------------------------------------------
// Helper: 给 Loop 内部打了 "npu.tiled" 标记的 Op 添加 Phase 标签
// 这会深入 Loop Body 找到计算 Op 并打标
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
// 返回 -1 表示动态或无法确定
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

// -----------------------------------------------------------------------------
// Custom Helper: Peel Last Iteration
// 逻辑: 克隆 Loop 作为 Tail (Last)，修改原 Loop 的 UB 作为 Body
// Body -> Tail
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

  // 计算分割点: Split = UB - Step
  // Body: [LB, Split)
  // Tail: [Split, UB)
  auto splitMap = AffineMap::get(0, 2, {ubSymbol - stepSymbol});
  b.setInsertionPoint(forOp);
  auto loc = forOp.getLoc();
  Value splitBound = b.createOrFold<affine::AffineApplyOp>(
      loc, splitMap, ValueRange{forOp.getUpperBound(), forOp.getStep()});

  // 1. 创建 Tail Loop (克隆原 Op)
  // 修改 Tail 的 LB 为 splitBound
  IRMapping map;
  map.map(forOp.getLowerBound(), splitBound);

  // 插入到原 Loop 之后
  b.setInsertionPointAfter(forOp);
  lastIteration = cast<scf::ForOp>(b.clone(*forOp.getOperation(), map));

  // 2. 修改原 Loop (Body)
  // 更新 UB 为 splitBound
  b.modifyOpInPlace(
      forOp, [&]() { forOp.getUpperBoundMutable().assign(splitBound); });

  // 3. 链接数据流
  // Body 的输出 -> Tail 的输入
  if (forOp.getNumResults() > 0) {
    b.modifyOpInPlace(lastIteration, [&]() {
      lastIteration.getInitArgsMutable().assign(forOp.getResults());
    });
  }

  // 4. 替换外部引用
  // 外部使用的是 Tail 的结果
  b.replaceOpUsesWithIf(forOp, lastIteration->getResults(),
      [&](OpOperand &use) { return use.getOwner() != lastIteration; });

  return success();
}

// -----------------------------------------------------------------------------
// Main Pattern
// -----------------------------------------------------------------------------
struct NpuConvTilingPattern : public OpRewritePattern<linalg::GenericOp> {
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

    // 2. 向上追溯寻找 Producer (npu_conv)
    Value input = op->getOperand(0);
    auto convOp = input.getDefiningOp<linalg::GenericOp>();
    if (!convOp || !convOp->hasAttr("library_call") ||
        convOp->getAttrOfType<StringAttr>("library_call").getValue() !=
            "npu_conv") {
      return failure();
    }

    // 3. 获取 npu_conv 的完整切分配置 [OC, OH, OW, IC]
    SmallVector<int64_t> tileSizes = getNpuTileSizes(convOp);
    if (tileSizes.size() < 4)
      return failure();

    // 拆分 Tile Sizes：空间维度 (给 mv) 和 规约维度 (给 conv 的 IC)
    SmallVector<int64_t> spatialTileSizes = {
        tileSizes[0], tileSizes[1], tileSizes[2]};
    SmallVector<int64_t> icTileSizes = {
        0, 0, 0, tileSizes[3]}; // 只有 IC 维度有值

    // =================================================================
    // Phase 1: 沿着空间维度对 mv_acc_to_spm 进行 Tiling，并融合 npu_conv
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

    // ==========================================
    // 修复点：在进入 Lambda 之前，提取出裸指针 (Operation *)
    // ==========================================
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

    // 给空间循环打上 Label (OC, OH, OW)
    auto spatialLoops = fuseResult->loops;
    for (size_t i = 0; i < std::min(spatialLoops.size(), (size_t)3); ++i) {
      spatialLoops[i]->setAttr(
          "npu.loop_dim", rewriter.getStringAttr(kDimensionLabels[i]));
      spatialLoops[i]->setAttr("npu.target", rewriter.getStringAttr("npu"));
    }

    // =================================================================
    // Phase 2: 在最内层空间循环 (OW) 中，找到被融合进来的局部 npu_conv
    // =================================================================
    auto tiledConsumer = fuseResult->tiledAndFusedOps.front();
    tiledConsumer->setAttr(
        "npu.tiled", rewriter.getUnitAttr()); // 标记 mv 已经处理过

    auto fusedConvOp =
        tiledConsumer->getOperand(0).getDefiningOp<linalg::GenericOp>();
    if (!fusedConvOp)
      return failure();

    fusedConvOp->setAttr("npu.tiled", rewriter.getUnitAttr());

    Location loc = fusedConvOp.getLoc();
    rewriter.setInsertionPoint(fusedConvOp);

    // 1. 分配 ACC 内存 (Space 3)
    OpOperand *convOutOperand = fusedConvOp.getDpsInitOperand(0);
    Value convOutVal = convOutOperand->get();
    auto convOutType = cast<RankedTensorType>(convOutVal.getType());

    // 解析动态维度的运行时 size
    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < convOutType.getRank(); ++i) {
      if (convOutType.isDynamicDim(i)) {
        // 使用 tensor::DimOp 获取当前切片在第 i 维的实际大小
        dynamicSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, convOutVal, i));
      }
    }

    // ========================================================
    // 【核心修复】：将 memory_space 编码进 Tensor 的 Type 基因里
    // ========================================================
    auto memSpaceAttr = rewriter.getI32IntegerAttr(3);
    auto accTensorType = RankedTensorType::get(convOutType.getShape(),
        convOutType.getElementType(),
        memSpaceAttr); // 注入 Encoding!

    // 传入 dynamicSizes 构建 AllocTensorOp，并将 memSpaceAttr 同时传给属性和
    // Type
    auto accAlloc = rewriter.create<bufferization::AllocTensorOp>(
        loc, accTensorType, dynamicSizes, Value{}, memSpaceAttr);

    auto newConvOp = rewriter.create<linalg::GenericOp>(loc,
        TypeRange{accTensorType}, // 将返回值类型也设为 Space 3
        fusedConvOp.getInputs(),
        ValueRange{accAlloc}, // outs 使用刚才建好的 Space 3 Alloc
        fusedConvOp.getIndexingMapsArray(),
        fusedConvOp.getIteratorTypesArray());

    // 1. 拷贝计算逻辑 (Region) 到新算子
    rewriter.inlineRegionBefore(fusedConvOp.getRegion(), newConvOp.getRegion(),
        newConvOp.getRegion().begin());

    // 2. 拷贝所有外挂属性 (比如 npu.tiled, library_call 等)
    newConvOp->setAttrs(fusedConvOp->getAttrs());

    // 3. 替换掉那个“旧时代”的算子
    rewriter.replaceOp(fusedConvOp, newConvOp.getResults());

    // 4. 更新指针，让后面的 Phase 3 (IC Tiling) 对着新算子切分
    fusedConvOp = newConvOp;

    // =================================================================
    // Phase 3: 对局部的 npu_conv 沿着 IC 维度进行 Tiling
    // =================================================================
    scf::SCFTilingOptions icOptions;
    icOptions.setTileSizes(
        getAsOpFoldResult(rewriter.getI64ArrayAttr(icTileSizes)));

    FailureOr<scf::SCFTilingResult> icTilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(fusedConvOp.getOperation()), icOptions);

    if (failed(icTilingResult))
      return failure();

    // 给 IC 循环打标签
    if (!icTilingResult->loops.empty()) {
      icTilingResult->loops.front()->setAttr(
          "npu.loop_dim", rewriter.getStringAttr("IC"));
    }

    // =================================================================
    // Phase 4: IC 维度的 Head -> Body -> Tail Peeling
    // =================================================================
    auto icLoops = icTilingResult->loops;
    SmallVector<Value> finalIcResults = icTilingResult->replacements;

    if (!icLoops.empty()) {
      auto loopOp = cast<scf::ForOp>(icLoops.back().getOperation());

      // B1: 检查是否为 Single Trip
      int64_t tripCount = getStaticTripCount(loopOp);
      if (tripCount == 1) {
        tagInnerComputeOp(loopOp, "single", rewriter);
      } else {
        scf::ForOp restLoop = loopOp;

        // --- Phase 4.1: Peel Head ---
        scf::ForOp headLoop;
        if (succeeded(peelForLoopFirstIteration(rewriter, loopOp, headLoop))) {
          tagInnerComputeOp(headLoop, "head", rewriter);
          restLoop = loopOp;
        }

        int64_t restTripCount = getStaticTripCount(restLoop);

        // --- Phase 4.2: Peel Tail ---
        if (restTripCount == 1) {
          tagInnerComputeOp(restLoop, "tail", rewriter);
          finalIcResults = restLoop->getResults();
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

    for (int i = (int)spatialLoops.size() - 1; i >= 0; --i) {
      // 1. 获取底层 Operation 并转换为 scf::ForOp
      auto loopOp = cast<scf::ForOp>(spatialLoops[i].getOperation());

      // 2. 声明用于接收尾部循环的局部变量
      scf::ForOp partialLoop;

      // 3. 调用 MLIR 的标准 Peel 接口
      if (succeeded(scf::peelForLoopAndSimplifyBounds(
              rewriter, loopOp, partialLoop))) {
        // 获取成功剥离出的尾部循环
        scf::ForOp tailLoop = partialLoop;
      }
    }

    // =================================================================
    // Phase 5: 链接数据流并替换原 Op
    // =================================================================
    // 1. 将融合后的 npu_conv 替换为你 peeling 之后生成的最终结果 (Tail的输出)
    rewriter.replaceOp(fusedConvOp, finalIcResults);

    // 2. 将外层原始的 mv_acc_to_spm 替换为 Tile & Fuse 流程的最终产物
    // 此时最外层返回的已经是完整的 i8 Tensor 了
    Value originalResult = op->getResult(0);
    if (fuseResult->replacements.count(originalResult)) {
      rewriter.replaceOp(op, fuseResult->replacements[originalResult]);
    } else {
      // 防御性编程：如果没有找到替换值，说明 Tiling 过程发生异常
      return failure();
    }

    return success();
  }
};

} // namespace

void npux::populateConvTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuConvTilingPattern>(context);
};