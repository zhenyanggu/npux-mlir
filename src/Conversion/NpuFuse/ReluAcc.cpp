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
bool isFusibleOp(linalg::GenericOp op) {
  if (!op)
    return false;
  auto libCall = op->getAttrOfType<StringAttr>("library_call");
  if (!libCall)
    return false;
  StringRef name = libCall.getValue();
  return name == "npu_gemm" || name == "npu_matmul" || name == "npu_conv";
}

struct NpuReluAccFusion : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp reluOp, PatternRewriter &rewriter) const override {
    // 1. 检查 Consumer (ReLU/LeakyReLU)
    auto reluCall = reluOp->getAttrOfType<StringAttr>("library_call");
    if (!reluCall)
      return failure();
    StringRef reluName = reluCall.getValue();

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

    // 2. 获取 Producer (Compute Op)
    Value input = reluOp.getInputs()[0];
    auto producerOp = input.getDefiningOp<linalg::GenericOp>();

    // 3. 验证 Producer 是否合法
    if (!isFusibleOp(producerOp))
      return failure();

    // 4. 验证依赖关系 (Single User check)
    // 只有当 Compute 的结果只被这个 ReLU 使用时，才能安全融合
    if (!producerOp->getResult(0).hasOneUse())
      return failure();

    // 5. 准备新 Op 的属性
    SmallVector<NamedAttribute> newAttrs;

    for (auto attr : producerOp->getAttrs()) {
      StringRef attrName = attr.getName().strref();
      if (attrName == "out_scale" || attrName == "out_zp")
        continue;
      newAttrs.push_back(attr);
    }

    newAttrs.push_back(
        rewriter.getNamedAttr("do_relu", rewriter.getI32IntegerAttr(1)));
    newAttrs.push_back(rewriter.getNamedAttr(
        "relu_type", rewriter.getI32IntegerAttr(reluTypeVal)));

    if (auto outScale = reluOp->getAttr("out_scale")) {
      newAttrs.push_back(rewriter.getNamedAttr("out_scale", outScale));
    }
    if (auto outZp = reluOp->getAttr("out_zp")) {
      newAttrs.push_back(rewriter.getNamedAttr("out_zp", outZp));
    }

    auto newGenericOp = rewriter.create<linalg::GenericOp>(reluOp.getLoc(),
        reluOp.getResultTypes(),           // resultTensorTypes
        producerOp.getInputs(),            // inputs
        reluOp.getOutputs(),               // outputs
        producerOp.getIndexingMapsAttr(),  // indexingMaps
        producerOp.getIteratorTypesAttr(), // iteratorTypes
        StringAttr(),                      // doc (留空)
        StringAttr(),                      // library_call (留空)
        nullptr, // body builder (留空，因为后面用了 takeBody)
        newAttrs // attributes (直接传 ArrayRef<NamedAttribute>)
    );

    // 3. 移动 Region (Body)
    newGenericOp.getRegion().takeBody(producerOp.getRegion());

    // 4. 替换并擦除
    rewriter.replaceOp(reluOp, newGenericOp->getResults());
    rewriter.eraseOp(producerOp);

    return success();
  }
};

} // end anonymous namespace

void npux::populateReluAccFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<NpuReluAccFusion>(patterns.getContext());
}