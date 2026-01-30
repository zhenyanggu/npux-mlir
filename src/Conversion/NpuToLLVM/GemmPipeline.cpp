//=============================================================================
// src/Conversion/NpuOptimization/GemmPipeline.cpp
//
// Refactored using robust analysis logic from ConvSplit.cpp.
// Applies pipelining (Head/Body/Tail) configuration without tiling.
//=============================================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialect/Npux/NpuxOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace npux;

namespace {

// ============================================================================
// 1. Analysis Helpers (Adapted from ConvSplit.cpp)
// ============================================================================

struct GemmLoopAnalysis {
    ComputeRunOp computeOp = nullptr;
    DmaMvinOp biasMvinOp = nullptr;   // 只有在 Loop 内部定义的 Bias Mvin 才会被记录
    MvAccToSpmOp accToSpmOp = nullptr;
    DmaMvoutOp mvoutOp = nullptr;

    bool isValid() const { return computeOp != nullptr; }
};

// 类似于 ConvSplit 中的 traceMvinChain，但我们只关心是否在当前 Loop 作用域内
DmaMvinOp findInternalBiasMvin(Value biasValue, Operation *loopScope) {
    if (!biasValue) return nullptr;

    // 向上查找定义
    if (auto mvin = biasValue.getDefiningOp<DmaMvinOp>()) {
        // 检查这个 mvin 是否是当前 loop 的直接子节点（防止误删外层 Op）
        if (mvin->getParentOp() == loopScope) {
            return mvin;
        }
    }
    return nullptr;
}

GemmLoopAnalysis analyzeLoop(scf::ForOp op) {
    GemmLoopAnalysis analysis;
    Operation *loopScope = op.getOperation();

    // 1. 核心：使用 walk 查找 ComputeOp，这是最稳健的方法
    op.getBody()->walk([&](ComputeRunOp comp) { 
        analysis.computeOp = comp; 
    });

    if (!analysis.isValid()) return analysis;

    // 2. 查找 Bias Mvin (限定在 Loop 内部)
    if (Value bias = analysis.computeOp.getBiaspsumMemref()) {
        analysis.biasMvinOp = findInternalBiasMvin(bias, loopScope);
    }

    // 3. 查找 Output 链条 (AccToSpm -> Mvout)
    // 直接 walk 查找，不依赖脆弱的 Def-Use 链，确保找到 Loop 内的所有相关 Op
    op.getBody()->walk([&](MvAccToSpmOp acc) { 
        analysis.accToSpmOp = acc; 
    });

    op.getBody()->walk([&](DmaMvoutOp mvout) { 
        analysis.mvoutOp = mvout; 
    });

    return analysis;
}

// ============================================================================
// 2. Pattern Logic
// ============================================================================

struct GemmLoopPipelinePattern : public OpRewritePattern<scf::ForOp> {
    using OpRewritePattern<scf::ForOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::ForOp loopOp, PatternRewriter &rewriter) const override {
        // 1. 检查标记 (防止重复处理)
        if (loopOp->hasAttr("npu.pipeline_done")) return failure();

        // 2. 检查 Loop Part 标记
        auto partAttr = loopOp->getAttrOfType<StringAttr>("npu.loop_part");
        if (!partAttr) return failure();

        // 3. 必须是 Gemm 计算
        auto computeAttr = loopOp->getAttrOfType<StringAttr>("npu.computeop");
        if (!computeAttr || computeAttr.getValue() != "gemm") return failure();

        // 4. 执行分析 (使用稳健的 Walk 方式)
        GemmLoopAnalysis analysis = analyzeLoop(loopOp);
        if (!analysis.isValid()) return failure();

        StringRef part = partAttr.getValue();
        bool isHead = (part == "head" || part == "single");
        bool isTail = (part == "tail" || part == "single");
        bool hasBias = (analysis.computeOp.getBiaspsumMemref() != nullptr);

        ImplicitLocOpBuilder b(loopOp.getLoc(), rewriter);
        Value cTrue = b.create<arith::ConstantIntOp>(1, 1);
        Value cFalse = b.create<arith::ConstantIntOp>(0, 1);

        // =========================================================
        // Step A: Configure Compute Op
        // =========================================================
        rewriter.modifyOpInPlace(analysis.computeOp, [&] {
            if (isHead) {
                // Head: Init (Accum=0), Load Bias (Bias=1 if exists)
                analysis.computeOp.getIsAccumulateMutable().assign(cFalse);
                analysis.computeOp.getAccBiasMutable().assign(hasBias ? cTrue : cFalse);
            } else {
                // Body/Tail: Accumulate (Accum=1), No Bias Load (Bias=0)
                analysis.computeOp.getIsAccumulateMutable().assign(cTrue);
                analysis.computeOp.getAccBiasMutable().assign(cFalse);
                // Body/Tail 不读取 Bias Buffer，断开连接
                analysis.computeOp.getBiaspsumMemrefMutable().clear();
            }
        });

        // =========================================================
        // Step B: Robust Erasure Logic (清理冗余 Op)
        // =========================================================
        
        // 1. 如果不是 Head，且 loop 内存在 Bias Mvin，删除之
        if (!isHead && analysis.biasMvinOp) {
            rewriter.eraseOp(analysis.biasMvinOp);
        }

        // 2. 如果不是 Tail，删除 Output 相关的 Op
        // 注意删除顺序：先删 Consumer (Mvout)，再删 Producer (AccToSpm)
        if (!isTail) {
            if (analysis.mvoutOp) {
                rewriter.eraseOp(analysis.mvoutOp);
            }
            if (analysis.accToSpmOp) {
                rewriter.eraseOp(analysis.accToSpmOp);
            }
        }

        // 3. 标记完成，防止死循环
        rewriter.modifyOpInPlace(loopOp, [&] {
            loopOp->setAttr("npu.pipeline_done", rewriter.getUnitAttr());
        });

        return success();
    }
};

// ============================================================================
// Pattern 2: Singleton Pattern (不在循环内的 Gemm)
// ============================================================================
struct GemmSingletonPattern : public OpRewritePattern<ComputeRunOp> {
    using OpRewritePattern<ComputeRunOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ComputeRunOp op, PatternRewriter &rewriter) const override {
        if (op->hasAttr("npu.pipeline_done")) return failure();
        if (op.getOpType() != ::npux::ComputeOpType::gemm) return failure();

        // 如果在 Loop 内，交给 LoopPattern 处理
        if (auto parentLoop = op->getParentOfType<scf::ForOp>()) {
            if (parentLoop->hasAttr("npu.loop_part")) return failure();
        }

        bool hasBias = (op.getBiaspsumMemref() != nullptr);
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);
        Value cTrue = b.create<arith::ConstantIntOp>(1, 1);
        Value cFalse = b.create<arith::ConstantIntOp>(0, 1);

        rewriter.modifyOpInPlace(op, [&] {
            op.getIsAccumulateMutable().assign(cFalse);
            op.getAccBiasMutable().assign(hasBias ? cTrue : cFalse);
            op->setAttr("npu.pipeline_done", rewriter.getUnitAttr());
        });

        return success();
    }
};

struct GemmPipelinePass : public PassWrapper<GemmPipelinePass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GemmPipelinePass)
    StringRef getArgument() const override { return "npu-gemm-pipeline"; }
    
    void runOnOperation() override {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<GemmLoopPipelinePattern>(context);
        patterns.add<GemmSingletonPattern>(context);
        
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> npux::createGemmPipelinePass() {
    return std::make_unique<GemmPipelinePass>();
}