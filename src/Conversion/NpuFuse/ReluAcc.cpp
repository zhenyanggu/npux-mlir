//=======================================
// src/Conversion/NpuFuse/ReluAcc.cpp
// this file implements the fusion of relu
// operation and accumulator-invoved operations
//=======================================

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Conversion/NpuFuse/FuseHelper.hpp"

using namespace mlir;

namespace {

// 辅助函数：安全地获取 library_call 属性
StringRef getLibraryCall(linalg::GenericOp op) {
  if (!op) return "";
  auto libCall = op->getAttrOfType<StringAttr>("library_call");
  return libCall ? libCall.getValue() : "";
}

struct NpuReluAccFusion : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp reluOp, PatternRewriter &rewriter) const override {
    
    // 1. 检查 Consumer (ReLU/LeakyReLU)
    StringRef reluName = getLibraryCall(reluOp);
    int reluTypeVal = -1;
    if (reluName == "npu_relu") {
      reluTypeVal = 0;
    } else if (reluName == "npu_leaky_relu") {
      if (auto alpha = reluOp->getAttrOfType<FloatAttr>("alpha")) {
        if (alpha.getValueAsDouble() == 0.1) {
          reluTypeVal = 2; // LeakyReLU with alpha=0 is standard ReLU
        } else if (alpha.getValueAsDouble() == 0.2) {
          reluTypeVal = 3; // LeakyReLU with alpha=0.01
        } else if (alpha.getValueAsDouble() == 0.01) {
          reluTypeVal = 4; // Unsupported alpha value
        } else {
          llvm::errs() << "Unsupported alpha value for npu_leaky_relu: "
                       << alpha.getValueAsDouble() << "\n";
          return failure();
        }
      } else {
        llvm::errs() << "Missing alpha attribute for npu_leaky_relu\n";
        return failure();
      }
    } else {
      return failure();
    }

    // 2. 向上追溯，匹配特定的数据流模式
    Value currentInput = reluOp.getInputs()[0];
    auto prevOp = currentInput.getDefiningOp<linalg::GenericOp>();
    
    // 确保依赖链存在且唯一
    if (!prevOp || !prevOp->getResult(0).hasOneUse()) 
      return failure();

    StringRef prevName = getLibraryCall(prevOp);
    bool hasLayout = false;

    // a. 检查是否存在 Layout 转换 (针对 Conv -> mv -> layout -> relu 模式)
    // 根据 IR，这里可能是 npu_layout_nchwc32_to_nchw 等，所以用 contains 进行泛化匹配
    if (prevName.contains("layout")) {
      currentInput = prevOp.getInputs()[0];
      prevOp = currentInput.getDefiningOp<linalg::GenericOp>();
      if (!prevOp || !prevOp->getResult(0).hasOneUse()) 
        return failure();
      prevName = getLibraryCall(prevOp);
      hasLayout = true;
    }

    // b. 此时的 prevOp 必须是 mv_acc_to_spm
    if (prevName != "mv_acc_to_spm") 
      return failure();

    // c. 获取底层的 Producer (Compute Op)
    currentInput = prevOp.getInputs()[0];
    auto producerOp = currentInput.getDefiningOp<linalg::GenericOp>();
    if (!producerOp || !producerOp->getResult(0).hasOneUse()) 
      return failure();

    StringRef producerName = getLibraryCall(producerOp);

    // 3. 严格验证 Producer 是否与模式匹配
    if (hasLayout) {
      if (producerName != "npu_conv") 
        return failure();
    } else {
      if (producerName != "npu_gemm" && producerName != "npu_matmul") 
        return failure();
    }

    // 4. 准备新 Compute Op 的属性 (将 relu 的属性迁移过去)
    SmallVector<NamedAttribute> newAttrs;
    for (auto attr : producerOp->getAttrs()) {
      StringRef attrName = attr.getName().strref();
      if (attrName == "out_scale" || attrName == "out_zp")
        continue;
      newAttrs.push_back(attr);
    }

    newAttrs.push_back(rewriter.getNamedAttr("do_relu", rewriter.getI32IntegerAttr(1)));
    newAttrs.push_back(rewriter.getNamedAttr("relu_type", rewriter.getI32IntegerAttr(reluTypeVal)));

    if (auto outScale = reluOp->getAttr("out_scale")) {
      newAttrs.push_back(rewriter.getNamedAttr("out_scale", outScale));
    }
    if (auto outZp = reluOp->getAttr("out_zp")) {
      newAttrs.push_back(rewriter.getNamedAttr("out_zp", outZp));
    }

    // 5. 【关键修复】直接原地更新 ProducerOp 的属性！
    // 这样不会改变 Op 的位置，也不会影响它和 mv_acc_to_spm 之间的连线
    rewriter.modifyOpInPlace(producerOp, [&]() {
      producerOp->setAttrs(rewriter.getDictionaryAttr(newAttrs));
    });

    // 6. “短接” Relu：跳过 Relu，让下游直接使用 Relu 的输入
    rewriter.replaceOp(reluOp, reluOp.getInputs());

    return success();
  }
};

} // end anonymous namespace

void npux::populateReluAccFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<NpuReluAccFusion>(patterns.getContext());
}