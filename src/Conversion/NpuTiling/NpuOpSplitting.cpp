//=============================================================================
// src/Conversion/NpuOpSplitting/NpuOpSplitting.cpp
// Implements tiling for Conv-related DMAs and Conv computation.
//=============================================================================

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

using namespace mlir;

namespace {

//=============================================================================
// Pattern 1: NpuDmaTilingPattern
// Only target Conv-related Activation DMAs (5D, Dim 1 is Channel)
//=============================================================================
struct NpuDmaTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    
    // 1. 防止递归
    if (op->hasAttr("npu.split_done")) return failure();

    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr) return failure();
    StringRef libName = libCallAttr.getValue();

    bool isMvin = libName.starts_with("npu_dma_mvin");
    bool isMvout = libName.starts_with("npu_dma_mvout");

    if (!isMvin && !isMvout) return failure();

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
        splitDim = 0;
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

    if (failed(tilingResult)) return failure();

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

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    
    if (op->hasAttr("npu.split_done")) return failure();

    auto libCallAttr = op->getAttrOfType<StringAttr>("library_call");
    if (!libCallAttr) return failure();
    StringRef libName = libCallAttr.getValue();

    // ==============================================================================
    // 1. 逻辑分叉点
    // ==============================================================================
    
    if (libName == "mv_acc_to_spm") {
        // -----------------------------------------------------------
        // 场景 A: Tail / Single 阶段 (进行融合分块)
        // -----------------------------------------------------------
        return handleTailFusion(op, rewriter);
    } 
    
    if (libName == "npu_conv") {
        // -----------------------------------------------------------
        // 场景 B: Head / Body 阶段 (仅对 Conv 分块)
        // -----------------------------------------------------------
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
  // 场景 B 的实现：仅对 npu_conv 进行分块
  // ----------------------------------------------------------------------------
  LogicalResult handleSimpleConvTiling(linalg::GenericOp op, PatternRewriter &rewriter) const {
    SmallVector<int64_t> staticTileSizes = {1, 0, 0, 0};
    SmallVector<OpFoldResult> tileSizes = getAsIndexOpFoldResult(rewriter.getContext(), staticTileSizes);

    auto type = cast<RankedTensorType>(op.getOutputs()[0].getType());
    if (type.getDimSize(0) <= 1) {
        op->setAttr("npu.split_done", rewriter.getUnitAttr());
        return failure();
    }

    auto tilingOptions = scf::SCFTilingOptions().setTileSizes(tileSizes);
    FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(op.getOperation()), tilingOptions);

    if (failed(tilingResult)) return failure();

    rewriter.replaceOp(op, tilingResult->loops.front()->getResults());
    for (auto *tiledOp : tilingResult->tiledOps) {
        tiledOp->setAttr("npu.split_done", rewriter.getUnitAttr());
    }
    return success();
  }

  // ----------------------------------------------------------------------------
  // 场景 A 的实现：mv_acc_to_spm + npu_conv 融合分块 (你原来的复杂逻辑)
  // ----------------------------------------------------------------------------
  LogicalResult handleTailFusion(linalg::GenericOp consumerOp, PatternRewriter &rewriter) const {
    OpOperand *inputOperand = &consumerOp->getOpOperand(0); 
    auto producerOp = inputOperand->get().getDefiningOp<linalg::GenericOp>();
    if (!producerOp) return failure();
    
    auto prodLibCall = producerOp->getAttrOfType<StringAttr>("library_call");
    if (!prodLibCall || prodLibCall.getValue() != "npu_conv") return failure();

    SmallVector<int64_t> staticTileSizes = {1, 0, 0, 0};
    SmallVector<OpFoldResult> tileSizes = getAsIndexOpFoldResult(rewriter.getContext(), staticTileSizes);

    auto type = cast<RankedTensorType>(consumerOp.getOutputs()[0].getType());
    if (type.getDimSize(0) <= 1) { 
        consumerOp->setAttr("npu.split_done", rewriter.getUnitAttr());
        return failure();
    }

    auto tilingOptions = scf::SCFTilingOptions().setTileSizes(tileSizes);

    // 5. 对 Consumer 分块
    FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(consumerOp.getOperation()), tilingOptions);
    if (failed(tilingResult)) return failure();

    // 6. 融合 Producer
    Operation *tiledConsumerOp = tilingResult->tiledOps.back();
    OpOperand &opOperandToFuse = tiledConsumerOp->getOpOperand(0);
    FailureOr<linalg::FusionInfo> fusionResult = linalg::fuseProducerOfTensor(rewriter, opOperandToFuse);
    if (failed(fusionResult)) return failure();

    // 标记 Done
    for (auto *tiledOp : tilingResult->tiledOps) {
        tiledOp->setAttr("npu.split_done", rewriter.getUnitAttr());
    }
    fusionResult->fusedProducer->setAttr("npu.split_done", rewriter.getUnitAttr());

    // 7. 重建 Loop 以添加 Accumulator (保持你原来的代码逻辑...)
    scf::ForOp oldLoop = cast<scf::ForOp>(tilingResult->loops.front());
    linalg::GenericOp fusedProducer = cast<linalg::GenericOp>(fusionResult->fusedProducer);
    
    SmallVector<Value> newInitArgs = llvm::to_vector(oldLoop.getInitArgs());
    Value accInitTensor = producerOp.getOutputs()[0]; 
    newInitArgs.push_back(accInitTensor);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(oldLoop);

    auto newLoop = rewriter.create<scf::ForOp>(
        oldLoop.getLoc(), oldLoop.getLowerBound(), oldLoop.getUpperBound(), oldLoop.getStep(), newInitArgs);

    Block *oldBody = oldLoop.getBody();
    Block *newBody = newLoop.getBody();
    SmallVector<Value> argMapping;
    argMapping.push_back(newBody->getArgument(0));
    for (size_t i = 0; i < oldLoop.getNumRegionIterArgs(); ++i) {
        argMapping.push_back(newBody->getArgument(1 + i));
    }
    rewriter.mergeBlocks(oldBody, newBody, argMapping);

    // 修正数据流
    OpOperand *outOperand = fusedProducer.getDpsInitOperand(0);
    auto accExtractOp = outOperand->get().getDefiningOp<tensor::ExtractSliceOp>();
    if (!accExtractOp) return failure();

    Value newAccIterArg = newBody->getArguments().back();
    rewriter.setInsertionPoint(accExtractOp);
    auto newAccExtractOp = rewriter.create<tensor::ExtractSliceOp>(
        accExtractOp.getLoc(), newAccIterArg, 
        accExtractOp.getMixedOffsets(), accExtractOp.getMixedSizes(), accExtractOp.getMixedStrides());
    rewriter.replaceOp(accExtractOp, newAccExtractOp.getResult());

    // 修正 Yield
    Operation *oldYield = newBody->getTerminator();
    rewriter.setInsertionPoint(oldYield);
    auto accInsertOp = rewriter.create<tensor::InsertSliceOp>(
        fusedProducer.getLoc(), fusedProducer.getResult(0), newAccIterArg,
        newAccExtractOp.getMixedOffsets(), newAccExtractOp.getMixedSizes(), newAccExtractOp.getMixedStrides());

    SmallVector<Value> newYields = llvm::to_vector(oldYield->getOperands());
    newYields.push_back(accInsertOp.getResult());
    rewriter.create<scf::YieldOp>(oldYield->getLoc(), newYields);
    rewriter.eraseOp(oldYield);

    // 替换
    rewriter.replaceOp(consumerOp, newLoop.getResult(0));
    rewriter.replaceOp(producerOp, newLoop.getResult(1));
    rewriter.eraseOp(oldLoop); 

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

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuOpSplittingPass() {
  return std::make_unique<NpuOpSplittingPass>();
}