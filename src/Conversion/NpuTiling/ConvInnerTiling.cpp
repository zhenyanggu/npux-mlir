//=============================================================
// src/Conversion/NpuTiling/ConvInnerTiling.cpp
// Second pass: Inner tiling for Cout/Cin chunks [1, 1]
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

// 复用 helper 函数
LogicalResult peelFirstIteration(RewriterBase &rewriter, scf::ForOp loopOp, 
                                 SmallVectorImpl<Operation*> &peeledOps) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loopOp);

  Location loc = loopOp.getLoc();
  Value lb = loopOp.getLowerBound();
  Value step = loopOp.getStep();

  IRMapping mapper; 
  mapper.map(loopOp.getInductionVar(), lb);
  
  for (auto it : llvm::zip(loopOp.getBody()->getArguments().drop_front(), loopOp.getInitArgs())) {
    mapper.map(std::get<0>(it), std::get<1>(it));
  }

  for (auto &op : loopOp.getBody()->without_terminator()) {
    Operation *clonedOp = rewriter.clone(op, mapper);
    peeledOps.push_back(clonedOp);
  }

  auto yieldOp = cast<scf::YieldOp>(loopOp.getBody()->getTerminator());
  SmallVector<Value> firstIterResults;
  for (Value operand : yieldOp.getOperands()) {
    firstIterResults.push_back(mapper.lookupOrDefault(operand));
  }

  Value newLb = rewriter.create<arith::AddIOp>(loc, lb, step);
  loopOp.setLowerBound(newLb);
  loopOp.getInitArgsMutable().assign(firstIterResults);

  return success();
}

struct NpuConvInnerTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    
    // 1. 基础检查
    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (!libCall || libCall.getValue() != "npu_conv") return failure();
    if (!op->hasAttr("npu.tiled")) return failure();
    if (op->hasAttr("npu.trivial_tiling")) return failure();
    if (op->hasAttr("npu.inner_tiled")) return failure();

    // 2. 获取外层传下来的模式
    StringRef parentMode = "init"; // 默认假设是第一块
    if (auto attr = op->getAttrOfType<StringAttr>("npu.accumulate_mode")) {
        parentMode = attr.getValue();
    }

    // 3. Tiling 参数设置
    SmallVector<int64_t> tileSizes(9, 0); 
    tileSizes[1] = 1; // OC
    tileSizes[4] = 1; // IC

    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizesOpFold =
        getAsOpFoldResult(rewriter.getI64ArrayAttr(tileSizes));

    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizesOpFold);
    // 强制循环交换: IC 在外, OC 在内 -> Input Stationary
    options.setInterchange(SmallVector<int64_t>{1, 0});

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult)) return failure();

    auto loops = tilingResult->loops;
    if (loops.empty()) return failure();

    scf::ForOp innerIcLoop = cast<scf::ForOp>(loops[0].getOperation());
    
    // 标记 Loop 类型
    innerIcLoop->setAttr("npu.loop_type", rewriter.getStringAttr("inner_ic"));
    if (loops.size() > 1) {
        if (auto loop = dyn_cast<scf::ForOp>(loops[1].getOperation())) {
            loop->setAttr("npu.loop_type", rewriter.getStringAttr("inner_oc"));
        }
    }

    // ==========================================================
    // [优化] 条件剥离 (Conditional Peeling)
    // ==========================================================
    
    if (parentMode == "init") {
        // Case A: 全局 Init 阶段
        
        SmallVector<Operation*> peeledOps;
        if (failed(peelFirstIteration(rewriter, innerIcLoop, peeledOps))) {
            return failure();
        }

        // 1. 标记 Peeled Ops (IC=0) -> INIT
        // 【修正点】使用 walk 深入查找嵌套在 OC Loop 里的 GenericOp
        for (auto *peeledOp : peeledOps) {
            peeledOp->walk([&](linalg::GenericOp tiledOp) {
                tiledOp->setAttr("npu.inner_tiled", rewriter.getUnitAttr());
                tiledOp->setAttr("npu.accumulate_mode", rewriter.getStringAttr("init"));
                // 继承 name...
                if (op->hasAttr("npu.layer_name")) 
                    tiledOp->setAttr("npu.layer_name", op->getAttr("npu.layer_name"));
            });
        }

        // 2. 标记 Loop Body (IC>0) -> ACCUMULATE
        innerIcLoop.walk([&](linalg::GenericOp tiledOp) {
            tiledOp->setAttr("npu.inner_tiled", rewriter.getUnitAttr());
            tiledOp->setAttr("npu.accumulate_mode", rewriter.getStringAttr("accumulate"));
            // 继承 name...
            if (op->hasAttr("npu.layer_name")) 
                tiledOp->setAttr("npu.layer_name", op->getAttr("npu.layer_name"));
        });

    } else {
        // Case B: 全局 Accumulate 阶段
        
        innerIcLoop.walk([&](linalg::GenericOp tiledOp) {
            tiledOp->setAttr("npu.inner_tiled", rewriter.getUnitAttr());
            tiledOp->setAttr("npu.accumulate_mode", rewriter.getStringAttr("accumulate"));
            
            if (op->hasAttr("npu.layer_name")) 
                tiledOp->setAttr("npu.layer_name", op->getAttr("npu.layer_name"));
        });
    }

    // 4. 替换原 Op
    rewriter.replaceOp(op, tilingResult->replacements);

    return success();
  }
};

} // namespace

void npux::populateConvInnerTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuConvInnerTilingPattern>(context);
}