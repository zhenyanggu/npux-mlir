//===================================================
// src/Conversion/NpuTiling/NpuTiling.cpp
// this file implements the all the tiling patterns for NPU
//===================================================

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Pass/Passes.hpp"

#include "src/Conversion/NpuTiling/ElemWise.hpp"

using namespace mlir;

namespace npux {
void populateNpuTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  populateElemWiseTilingPatterns(patterns, context);
};
} // namespace npux

namespace {
struct NpuTilingPass
    : public PassWrapper<NpuTilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuTilingPass)

  StringRef getArgument() const override { return "npu-tiling"; }
  StringRef getDescription() const override {
    return "Tile operations for NPU execution.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto targetAttr = func->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu") {
      return;
    }

    MLIRContext *context = &getContext();

    ConversionTarget target(*context);

    target.addLegalDialect<tensor::TensorDialect, scf::SCFDialect,
        arith::ArithDialect, bufferization::BufferizationDialect,
        linalg::LinalgDialect>();
    target.addDynamicallyLegalOp<linalg::GenericOp>(
        [](linalg::GenericOp op) { return op->hasAttr("npu.tiled"); });

    RewritePatternSet patterns(context);

    npux::populateNpuTilingPatterns(patterns, context);

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace

std::unique_ptr<Pass> npux::createNpuTilingPass() {
  return std::make_unique<NpuTilingPass>();
}