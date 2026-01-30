//=============================================================
// src/Conversion/NpuTiling/Conv.cpp
// this file is for Conv op tiling pattern
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"

using namespace mlir;
using namespace npux;

namespace {

// ============================================================================
// 辅助函数：剥离循环的第一次迭代 (Head Peeling)
// 逻辑：
// 1. 复制 Loop Body 的内容到 Loop 之前。
// 2. 将原 Loop 的 init_args 替换为第一次迭代产生的结果。
// 3. 将原 Loop 的 LowerBound 增加一个 Step。
// ============================================================================
LogicalResult peelFirstIteration(RewriterBase &rewriter, scf::ForOp loopOp, 
                                 SmallVectorImpl<Operation*> &peeledOps) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loopOp);

  Location loc = loopOp.getLoc();
  Value lb = loopOp.getLowerBound();
  Value ub = loopOp.getUpperBound();
  Value step = loopOp.getStep();

  // 1. 简单检查：如果本身可能一次都不执行，则不剥离 (这里假设 Loop 至少执行一次)
  // 实际工程中可以加 arith.cmpi 判断，但 Tiling 通常是静态确定的
  
  // 2. 准备 Mapping：把 Loop Body 里的 BlockArgs 映射到外面
  Block *body = loopOp.getBody();
  IRMapping mapper;
  
  // 映射 Induction Variable -> LowerBound
  mapper.map(loopOp.getInductionVar(), lb);
  
  // 映射 iter_args -> init_args
  for (auto it : llvm::zip(body->getArguments().drop_front(), loopOp.getInitArgs())) {
    mapper.map(std::get<0>(it), std::get<1>(it));
  }

  // 3. Clone Body 内容
  for (auto &op : body->without_terminator()) {
    Operation *clonedOp = rewriter.clone(op, mapper);
    peeledOps.push_back(clonedOp); // 收集剥离出来的 Op，方便打标
  }

  // 4. 获取第一次迭代的 Yield 结果
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
  SmallVector<Value> firstIterResults;
  for (Value operand : yieldOp.getOperands()) {
    firstIterResults.push_back(mapper.lookupOrDefault(operand));
  }

  // 5. 更新原循环：LowerBound += Step
  Value newLb = rewriter.create<arith::AddIOp>(loc, lb, step);
  loopOp.setLowerBound(newLb);

  // 6. 更新原循环：InitArgs 变为第一次迭代的结果
  loopOp.getInitArgsMutable().assign(firstIterResults);

  return success();
}

struct NpuConvTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    
    // 1. 检查条件
    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (!libCall || libCall.getValue() != "npu_conv") return failure();
    if (op->hasAttr("npu.tiled")) return failure();

    SmallVector<int64_t> tileSizes = getNpuTileSizes(op);
    auto loopRanges = op.getStaticLoopRanges();

    // 2. 检查是否需要切分
    if (!isTilingNecessary(tileSizes, loopRanges)) {
        op->setAttr("npu.tiled", rewriter.getUnitAttr());
        op->setAttr("npu.trivial_tiling", rewriter.getUnitAttr());
        // 如果不需要切分，说明这一个 Op 就是完整的，它就是 First Chunk
        op->setAttr("npu.accumulate_mode", rewriter.getStringAttr("init")); 
        return success();
    }

    // 3. 执行 Tiling
    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizesOpFold =
        getAsOpFoldResult(rewriter.getI64ArrayAttr(tileSizes));

    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizesOpFold);

    // 注意：这里不需要 interchange，Outer Tiling 保持 [OC, H, W, IC_chunk] 
    // 或者 [H, W, OC, IC_chunk] 的顺序，只要确保 IC_chunk 在最内层即可。
    // 假设 tileSizes 对应的 Loop 顺序生成的 loops 向量，IC Loop 是最后一个。
    
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult)) return failure();

    auto loops = tilingResult->loops;
    SmallVector<Value> finalResults = tilingResult->replacements;

    // 4. [核心修改] 找到 IC Loop 并执行 Head Peeling
    // 假设 IC Loop 是 loops 里的最后一个 (Reduction loop)
    // 更严谨的做法是检查 IteratorType，这里假设 tileSizes 只有 d4 是 reduction
    
    scf::ForOp icLoopOp;
    if (!loops.empty()) {
        // 这里需要根据你的 layout 确定 IC Loop 的索引
        // 假设 tileSizes = [0, OC, H, W, IC, ...]，且 scf::tileUsingSCF 返回的 loops 
        // 仅包含非零 tile size 的维度。
        // 如果你确信 IC 是最内层循环 (loops.back())：
        if (auto loop = dyn_cast<scf::ForOp>(loops.back().getOperation())) {
             icLoopOp = loop;
        }
    }

    if (icLoopOp) {
        SmallVector<Operation*> peeledOps;
        // 执行 Head Peeling
        if (succeeded(peelFirstIteration(rewriter, icLoopOp, peeledOps))) {
            
            // A. 给剥离出来的 Op (Chunk 0) 打上 INIT 标记
            for (auto *peeledOp : peeledOps) {
                // 我们只关心 GenericOp 或者后续会被 Inner Tiling 处理的 Op
                if (isa<linalg::GenericOp>(peeledOp)) {
                    peeledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
                    peeledOp->setAttr("npu.accumulate_mode", rewriter.getStringAttr("init"));
                    // 传递 Layer Name
                    if (op->hasAttr("npu.layer_name")) {
                        peeledOp->setAttr("npu.layer_name", op->getAttr("npu.layer_name"));
                    }
                }
            }
            
            // B. 给剩余 Loop 里的 Op (Chunk 1..N) 打上 ACCUMULATE 标记
            // 此时 icLoopOp 已经是剩余部分的循环了
            icLoopOp.walk([&](linalg::GenericOp tiledOp) {
                tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
                tiledOp->setAttr("npu.accumulate_mode", rewriter.getStringAttr("accumulate"));
                 if (op->hasAttr("npu.layer_name")) {
                        tiledOp->setAttr("npu.layer_name", op->getAttr("npu.layer_name"));
                }
            });

        } else {
             // Peeling 失败 (可能是 Loop 只有 1 次迭代)，则整体标记为 Init
             // (后续逻辑处理)
        }
    }

    // 5. 处理 Tail Peeling (解决不能整除的问题)
    // 注意：Head Peeling 改变了 loops.back() 的 LowerBound，但不影响 Tail Peeling 的逻辑
    for (int i = loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp) continue;
      
      // 如果这个 Loop 已经被 Head Peeling 过了，Tail Peeling 依然有效
      // 它会处理 [LB+Step, UB] 区间的尾部
      scf::ForOp partialIteration;
      LogicalResult status = scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
        partialIteration->setAttr("npu.peeled_tail", rewriter.getUnitAttr());
        
        // 如果是 IC Loop 的 Tail，它一定属于 Accumulate 阶段 (因为 Head 肯定是 Init)
        if (loopOp == icLoopOp) {
             partialIteration.walk([&](linalg::GenericOp tailOp){
                 tailOp->setAttr("npu.accumulate_mode", rewriter.getStringAttr("accumulate"));
             });
        }

        if (i == 0) {
           finalResults = partialIteration->getResults();
        }
      }
    }

    // 6. 标记所有没有被 Peeling 特殊处理的 Op (作为默认处理)
    // 主要是为了防止上面的逻辑漏网，确保所有生成的 tiled op 都有 npu.tiled 属性
    for (Operation *tiledOp : tilingResult->tiledOps) {
        if (!tiledOp->hasAttr("npu.tiled")) {
            tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
            // 如果没被打标，说明它可能是非 IC Loop 内部的，或者 Peeling 没发生
            // 默认看作 init (根据情况调整)
            if (!tiledOp->hasAttr("npu.accumulate_mode")) {
                 // 这是一个保险策略，通常上面的逻辑应该覆盖了
                 tiledOp->setAttr("npu.accumulate_mode", rewriter.getStringAttr("init"));
            }
        }
    }

    rewriter.replaceOp(op, finalResults);
    return success();
  }
};

} // namespace

void npux::populateConvTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuConvTilingPattern>(context);
}