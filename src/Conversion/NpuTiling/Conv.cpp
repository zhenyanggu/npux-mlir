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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include "src/Pass/Passes.hpp"
#include <algorithm>
#include <cctype>
#include <cmath> // for ceil
#include <optional>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace npux;
namespace {

llvm::cl::opt<std::string> convTilingStrategy(
    "conv-tiling-strategy",
    llvm::cl::desc(
        "Conv tiling strategy: legacy (simple fixed tile) or costmodel"),
    llvm::cl::init("costmodel"));

llvm::cl::opt<std::string> convFusionStrategy(
    "conv-fusion-strategy",
    llvm::cl::desc(
        "Conv tiling fusion strategy: legacy (fuse first compute only) or "
        "extended (fuse until the second NPU compute op)"),
    llvm::cl::init("extended"));

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

static bool isNpuComputeBoundary(Operation *op) {
  auto genericOp = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!genericOp)
    return false;

  auto libCall = genericOp->getAttrOfType<StringAttr>("library_call");
  if (!libCall)
    return false;

  StringRef name = libCall.getValue();
  return name == "npu_conv" || name == "npu_gemm" || name == "npu_matmul"|| name == "npu_maxpool";
}

static bool useLegacyFusionStrategy() {
  std::string strategy = convFusionStrategy;
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

static SmallVector<Operation *> collectLinearGenericChain(
    linalg::GenericOp start, linalg::GenericOp endInclusive) {
  SmallVector<Operation *> chain;
  linalg::GenericOp cur = start;
  while (cur) {
    chain.push_back(cur.getOperation());
    if (cur == endInclusive || cur->getNumResults() != 1)
      break;
    Operation *user = getSingleUser(cur->getResult(0));
    cur = dyn_cast_or_null<linalg::GenericOp>(user);
  }
  return chain;
}

static bool isLayoutNchwc32ToNchwOp(Operation *op) {
  auto genericOp = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!genericOp)
    return false;
  auto libCall = genericOp->getAttrOfType<StringAttr>("library_call");
  return libCall && libCall.getValue() == "npu_layout_nchwc32_to_nchw";
}

static linalg::GenericOp findFirstLayoutNchwc32ToNchwOnChain(
    linalg::GenericOp start, linalg::GenericOp endInclusive) {
  linalg::GenericOp cur = start;
  while (cur) {
    if (isLayoutNchwc32ToNchwOp(cur.getOperation()))
      return cur;
    if (cur == endInclusive || cur->getNumResults() != 1)
      break;
    Operation *user = getSingleUser(cur->getResult(0));
    cur = dyn_cast_or_null<linalg::GenericOp>(user);
  }
  return nullptr;
}

static SmallVector<int64_t> getSpatialTileSizesForFusionConsumer(
    linalg::GenericOp firstConsumer, linalg::GenericOp fusionConsumer,
    ArrayRef<int64_t> convSpatialTileSizes) {
  // 默认行为：沿用 Conv 的空间分块（N, OC_outer, OH, OW）
  SmallVector<int64_t> defaultTiles(convSpatialTileSizes.begin(),
      convSpatialTileSizes.end());

  linalg::GenericOp layoutOp =
      findFirstLayoutNchwc32ToNchwOnChain(firstConsumer, fusionConsumer);
  if (!layoutOp)
    return defaultTiles;

  // 只要融合链里包含了 npu_layout_nchwc32_to_nchw，就按其映射语义重算：
  // flat_C = C_chunk * tileSize + C_block
  // 其中 tileSize 不写死，取 layout 输入的最后一维。
  int64_t tileSize = 0;
  if (layoutOp->getNumOperands() > 0) {
    if (auto inType =
            dyn_cast<RankedTensorType>(layoutOp->getOperand(0).getType())) {
      if (inType.getRank() >= 5) {
        int64_t maybeTile = inType.getShape()[4];
        if (!ShapedType::isDynamic(maybeTile))
          tileSize = maybeTile;
      }
    }
  }

  unsigned loops = fusionConsumer.getNumLoops();
  SmallVector<int64_t> mappedTiles(loops, 0);

  if (loops >= 1)
    mappedTiles[0] = defaultTiles[0]; // N
  if (loops >= 2) {
    if (loops == 4) {
      // 对 4D(NCHW) consumer，C 维按 layout 语义映射。
      mappedTiles[1] = (tileSize > 0) ? defaultTiles[1] * tileSize : 0;
    } else {
      // 对 5D(N,C_chunk,H,W,C_block) consumer，保留 C_chunk 分块。
      mappedTiles[1] = defaultTiles[1];
    }
  }
  if (loops >= 3)
    mappedTiles[2] = defaultTiles[2]; // H
  if (loops >= 4)
    mappedTiles[3] = defaultTiles[3]; // W
  if (loops >= 5)
    mappedTiles[4] = tileSize; // C_block，使用真实 tileSize，避免写死 32

  return mappedTiles;
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
  // 不请求 producer yield replacement，避免为中间结果（如 mv_acc_to_spm 输出）
  // 额外生成 scf.iter_args / insert_slice 回填链，导致后续 bufferization 落成 memref.copy。
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

static Value createAllocLikeWithEncoding(
    RewriterBase &rewriter, Location loc, Value likeValue, int64_t encoding) {
  auto type = dyn_cast<RankedTensorType>(likeValue.getType());
  if (!type)
    return Value();

  SmallVector<Value> dynamicSizes;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (type.isDynamicDim(i))
      dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, likeValue, i));
  }

  auto newType = RankedTensorType::get(
      type.getShape(), type.getElementType(), rewriter.getI64IntegerAttr(encoding));
  return rewriter.create<bufferization::AllocTensorOp>(loc, newType, dynamicSizes);
}

static linalg::GenericOp retargetSingleOutputGenericToEncoding(
    linalg::GenericOp op, int64_t encoding, RewriterBase &rewriter) {
  if (!op || op.getNumResults() != 1 || op.getNumDpsInits() != 1)
    return op;

  auto outType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!outType)
    return op;
  if (auto enc = dyn_cast_or_null<IntegerAttr>(outType.getEncoding())) {
    if (enc.getInt() == encoding)
      return op;
  }

  Value oldOut = op.getDpsInitOperand(0)->get();
  Value newOut = createAllocLikeWithEncoding(rewriter, op.getLoc(), oldOut, encoding);
  if (!newOut)
    return op;

  auto newOp = cast<linalg::GenericOp>(rewriter.clone(*op.getOperation()));
  newOp.getInputsMutable().assign(op.getInputs());
  newOp.getOutputsMutable().assign(ValueRange{newOut});
  newOp.getResult(0).setType(newOut.getType());
  newOp->setAttrs(op->getAttrs());
  rewriter.replaceOp(op, newOp.getResults());
  return newOp;
}

static std::pair<linalg::GenericOp, linalg::GenericOp>
retargetLinearChainOutputsToEncoding(linalg::GenericOp start,
    linalg::GenericOp endInclusive, int64_t encoding, RewriterBase &rewriter) {
  if (!start || !endInclusive)
    return {start, endInclusive};

  // 只改链路中间结果，不改链尾 endInclusive 的输出类型。
  // 链尾通常是 layout_out，其输出可能需要回到外部可见内存空间（后续 mvout）。
  linalg::GenericOp newStart = start;
  linalg::GenericOp cur = start;
  bool first = true;
  while (cur) {
    if (cur == endInclusive)
      return {newStart, cur};

    linalg::GenericOp newCur =
        retargetSingleOutputGenericToEncoding(cur, encoding, rewriter);
    if (first) {
      newStart = newCur;
      first = false;
    }

    Operation *nextUser = getSingleUser(newCur.getResult(0));
    cur = dyn_cast_or_null<linalg::GenericOp>(nextUser);
  }

  return {newStart, endInclusive};
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


static std::vector<int64_t> getDseAttrValues(linalg::GenericOp op) {
  auto dseAttr = op->getAttrOfType<ArrayAttr>("npu.dse_tiling");
  if (!dseAttr || dseAttr.size() != 4) {
    return {};
  }
  std::vector<int64_t> values;
  for (auto val : dseAttr) {
    values.push_back(cast<IntegerAttr>(val).getInt());
  }
  return values;
}

static SmallVector<int64_t> getSimpleConvTileSizes(linalg::GenericOp op,
    unsigned rank) {
  SmallVector<int64_t> sizes(rank, 0);

  int64_t oh = 8;
  int64_t ow = 8;
  if (!op.getOutputs().empty()) {
    if (auto type = dyn_cast<RankedTensorType>(op.getOutputs()[0].getType())) {
      ArrayRef<int64_t> shape = type.getShape();
      if (shape.size() > 2 && !ShapedType::isDynamic(shape[2]))
        oh = std::min<int64_t>(oh, std::max<int64_t>(1, shape[2]));
      if (shape.size() > 3 && !ShapedType::isDynamic(shape[3]))
        ow = std::min<int64_t>(ow, std::max<int64_t>(1, shape[3]));
    }
  }

  sizes[0] = 1;  // N
  sizes[1] = 1;  // OC outer, i.e. 32 output channels
  sizes[2] = oh; // OH
  sizes[3] = ow; // OW
  sizes[4] = 1;  // IC outer, i.e. 32 input channels

  llvm::errs() << "[Tiling] Conv: Strategy=LegacyAuto"
               << " -> Tile=[OH:" << oh << ", OW:" << ow
               << ", IC_blk:32, OC_blk:32]\n";
  return sizes;
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

  std::string strategy = convTilingStrategy;
  std::transform(strategy.begin(), strategy.end(), strategy.begin(),
      [](unsigned char c) { return std::tolower(c); });

  if (strategy == "legacy") {
    return getSimpleConvTileSizes(op, rank);
  }
  if (strategy != "costmodel") {
    llvm::errs() << "[Tiling] Conv: unknown conv-tiling-strategy='"
                 << convTilingStrategy << "', fallback to costmodel\n";
  }

  SmallVector<int64_t> sizes(rank, 0);
  auto &config = npux::NPUConfig::getInstance();
  
  // 1. 优先级：CLI/JSON 配置 > IR 属性 (DSE)
  std::vector<int64_t> configSizes = config.getConvTileSize();
  if (configSizes.empty()) {
    configSizes = getDseAttrValues(op);
  }

  // 2. 维度映射逻辑
  if (configSizes.size() >= 4) {
    int64_t t_oh = configSizes[0];
    int64_t t_ow = configSizes[1];
    int64_t t_ic = configSizes[2];
    int64_t t_oc = configSizes[3];

    // 映射到 NCHWc32: [N, OC_outer, OH, OW, IC_outer]
    sizes[0] = 1;                                 // N 维度不分块
    sizes[1] = (t_oc > 32) ? (t_oc / 32) : 1;     // OC (对齐 32)
    sizes[2] = t_oh;                              // OH
    sizes[3] = t_ow;                              // OW
    sizes[4] = (t_ic > 32) ? (t_ic / 32) : 1;     // IC (对齐 32)

    // 日志记录
    llvm::errs() << "[Tiling] Conv: Tile=[OH:" << t_oh << ", OW:" << t_ow 
                 << ", IC_blk:" << t_ic << ", OC_blk:" << t_oc << "]\n";
  }

  return sizes;
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
    SmallVector<int64_t> convSpatialTileSizes = {
        tileSizes[0], tileSizes[1], tileSizes[2], tileSizes[3]};
    SmallVector<int64_t> icTileSizes = {
        0, 0, 0, 0, tileSizes[4]}; // IC 维度在第 5 位

    // Phase 1: 从当前 Conv 后的 mv_acc_to_spm 开始向下找到融合终点，
    // 再做 Spatial Tiling，并向上融合回当前 npu_conv。
    linalg::GenericOp fusionConsumerOp = findDownstreamFusionConsumer(op);
    // 只有当确实向下融合了链（不仅是 mv_acc_to_spm 本身）时，
    // 才把链上的中间结果统一放在 encoding=2，避免后续再回落到 encoding=1。
    if (fusionConsumerOp && fusionConsumerOp != op) {
      std::tie(op, fusionConsumerOp) =
          retargetLinearChainOutputsToEncoding(op, fusionConsumerOp, 2, rewriter);
    }
    SmallVector<Operation *> originalChain =
        collectLinearGenericChain(op, fusionConsumerOp);
    SmallVector<int64_t> consumerTileSizes = getSpatialTileSizesForFusionConsumer(
        op, fusionConsumerOp, convSpatialTileSizes);
    auto consumerTilingInterface =
        cast<TilingInterface>(fusionConsumerOp.getOperation());
    scf::SCFTileAndFuseOptions fuseOptions;
    fuseOptions.tilingOptions.setTileSizes(
        getAsOpFoldResult(rewriter.getI64ArrayAttr(consumerTileSizes)));

    Operation *targetOpPtr = convOp.getOperation();
    fuseOptions.setFusionControlFn(
        [targetOpPtr](tensor::ExtractSliceOp candidateSliceOp,
            OpResult originalProducer, bool isDestinationOperand)
            -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          return fuseBackToCurrentNpuCompute(targetOpPtr, originalProducer);
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
    markAllFusedOpsAsTiled(*fuseResult, rewriter);
    auto tiledConsumer = fuseResult->tiledAndFusedOps.front();

    auto fusedConvOp = findFusedNpuCompute(
        tiledConsumer, [](StringRef name) { return name == "npu_conv"; });
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

    // 应用 tile+fuse 产生的全部 value replacement，避免旧链残留继续被使用。
    for (const auto &it : fuseResult->replacements) {
      Value oldValue = it.first;
      Value newValue = it.second;
      if (oldValue != newValue)
        rewriter.replaceAllUsesWith(oldValue, newValue);
    }

    // 清理原始链上失效的 op（逆序删除，避免 use 关系干扰）。
    for (Operation *oldOp : llvm::reverse(originalChain)) {
      if (oldOp && oldOp->use_empty())
        rewriter.eraseOp(oldOp);
    }

    if (convOp && convOp->use_empty()) {
      rewriter.eraseOp(convOp);
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
