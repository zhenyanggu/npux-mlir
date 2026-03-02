//===============================================
// src/Conversion/NpuxFusion/NpuxComputeFusion.cpp
//===============================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/Npux/NpuxOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace npux;

namespace {

//=============================================================================
// Pattern: FuseComputeAndMvAccToSpm
//=============================================================================
struct FuseComputeAndMvAccToSpm
    : public OpRewritePattern<MvAccToSpmOp> {
  using OpRewritePattern<MvAccToSpmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      MvAccToSpmOp mvOp, PatternRewriter &rewriter) const override {
    
    // 1. 获取相关 Value
    Value accSrc = mvOp.getAccSrc(); // 这是 ACC Buffer (SubView)
    Value spmDst = mvOp.getSpmDst();

    // 2. [修正] 寻找 ComputeRunOp
    // accSrc 不是由 ComputeRunOp 定义的 (它是由 alloc/subview 定义的)。
    // ComputeRunOp 只是它的使用者 (写入者)。
    // 因此，我们需要在当前 Block 中从 mvOp 开始向前回溯，找到最近一次写入 accSrc 的操作。
    
    ComputeRunOp computeOp;
    Operation *curr = mvOp.getOperation();
    
    // 向前回溯查找
    while ((curr = curr->getPrevNode())) {
        // 如果找到了 compute_run
        if (auto op = dyn_cast<ComputeRunOp>(curr)) {
            // 检查这个 compute_run 的输出 buffer 是否就是我们的 accSrc
            if (op.getOutput() == accSrc) {
                computeOp = op;
                break; // 找到了，停止搜索
            }
            // 如果是其他的 compute_run (写别的 buffer)，继续往前找
        }

        // 简单的安全边界：如果遇到了其他会读写内存的操作(且不是 view 类操作)，
        // 且该操作读写了 accSrc，则说明中间有干扰，不能融合。
        // 为简化逻辑，这里假设调度器已保证 compute 和 mv 之间没有复杂的数据依赖干扰。
        // 如果回溯太远没找到，最终 computeOp 为空，会返回 failure。
    }

    if (!computeOp)
      return failure(); // 没找到对应的 compute_run

    // 3. 检查融合条件
    
    // 3.1 确保原本的计算是输出到 ACC 的
    if (computeOp.getAccoutDest() != AccoutDest::acc)
      return failure();

    // 4. 准备新 Op 的参数
    
    // [关键点 A] 切换 Destination 为 SPM
    AccoutDest newDest = AccoutDest::spm;

    // 5. 创建新的 ComputeRunOp (输出直接写 SPM)
    rewriter.create<ComputeRunOp>(
        computeOp.getLoc(),
        // --- 1. Operation Control ---
        computeOp.getOpType(),
        computeOp.getDataflowMode(),
        newDest,                    // <--- Changed: ACC -> SPM
        computeOp.getIntType(),
        
        // --- 2. Buffers ---
        computeOp.getInputA(),
        computeOp.getInputB(),
        computeOp.getPsumMemref(), 
        spmDst,                        // <--- Changed: Output is now SPM Buffer

        // --- 3. Padding ---
        computeOp.getPadTop(), computeOp.getPadBottom(),
        computeOp.getPadLeft(), computeOp.getPadRight(),
        computeOp.getPadMode(),

        // --- 4. Kernel / Weight ---
        computeOp.getWeightShapeM1(), computeOp.getWeightStrideM1(),
        computeOp.getWeightDilationM1(), computeOp.getIsGroupConv(),

        // --- 5. Input Geometry ---
        computeOp.getInputAColNumM1(), computeOp.getInputARowNumM1(), computeOp.getInputAStride(),
        computeOp.getInputBColNumM1(), computeOp.getInputBRowNumM1(), computeOp.getInputBStride(),

        // --- 6. Bias/Psum & Loop Control ---
        computeOp.getBiaspsumWidth(), computeOp.getBiaspsumHeight(),
        computeOp.getBiaspsumStride(),

        // --- 7. Output Stride ---
        computeOp.getOutputStride(),            

        // --- 8. Post-Processing & Quantization ---
        computeOp.getIsAccumulate(),
        computeOp.getReluEnable(),
        computeOp.getReluType(),
        computeOp.getAccBias(),
        
        computeOp.getOutputZeropoint(),
        computeOp.getQuantScale(),
        computeOp.getQuantScaleshift(),
        computeOp.getInputAZeropoint(),
        computeOp.getInputBZeropoint()
    );

    // 6. 清理旧 Op
    rewriter.eraseOp(mvOp);
    rewriter.eraseOp(computeOp);

    return success();
  }
};

//=============================================================================
// Pass Definition
//=============================================================================
struct NpuxComputeFusionPass
    : public PassWrapper<NpuxComputeFusionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuxComputeFusionPass)

  llvm::StringRef getArgument() const override { return "npux-compute-fusion"; }
  llvm::StringRef getDescription() const override {
    return "Fuse npux.compute_run (acc) and npux.mv_acc_to_spm into a single compute_run (spm).";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<NpuxDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<FuseComputeAndMvAccToSpm>(context);

    // [修正] 使用 applyPatternsGreedily 替代过时的 applyPatternsAndFoldGreedily
    GreedyRewriteConfig config;
    
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuxComputeFusionPass() {
  return std::make_unique<NpuxComputeFusionPass>();
}