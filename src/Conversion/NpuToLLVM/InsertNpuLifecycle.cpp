//=====================================================
//src/Conversion/NpuToLLVM/InsertNpuLifecycle.cpp
//this file implements npu lifecycle insertion pattern
//=====================================================


#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "src/Dialect/Npux/NpuxOps.hpp"
#include "src/Conversion/NpuToLLVM/NpuxConversionHelper.hpp"

using namespace mlir;
using namespace npux;

namespace {

struct InsertNpuLifecyclePattern : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op, PatternRewriter &rewriter) const override {
    // 1. 过滤：只处理入口函数
    // 你可以根据名字判断，比如 "main_graph" 或 "main"
    // 或者判断是否有 "llvm.emit_c_interface" 属性
    StringRef funcName = op.getName();
    if (funcName != "main_graph" && funcName != "main") {
      return failure();
    }

    // 防止重复插入 (比如 Pass 跑了两次)
    if (op.getBody().front().getOps<npux::InitOp>().begin() != 
        op.getBody().front().getOps<npux::InitOp>().end()) {
      return failure();
    }

    // 我们需要修改函数体，而不是替换整个函数 Op，所以使用 updateRootInPlace
    rewriter.modifyOpInPlace(op, [&] {
      Block &entryBlock = op.getBody().front();
      Location loc = op.getLoc();

      // 2. 在入口插入 Init
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&entryBlock);
        // InitOp 有一个返回值 status (I32)，虽然我们在 IR 里可能不检查它
        rewriter.create<npux::InitOp>(loc); 
      }

      // 3. 在所有 Return 处插入 Destroy
      op.walk([&](func::ReturnOp retOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(retOp);
        rewriter.create<npux::DestroyOp>(retOp.getLoc());
      });
    });

    return success();
  }
};

} // namespace

void npux::populateNpuLifecyclePatterns(RewritePatternSet &patterns) {
  patterns.add<InsertNpuLifecyclePattern>(patterns.getContext());
}