//=============================================================
// src/Conversion/NpuTiling/ElemWise.cpp
// this file is for elemwise op tiling pattern
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h" // 核心 Tiling 工具
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"


#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

// === 1. 定义分块策略逻辑 ===
// 这里是策略的核心：根据 Op 的特点决定由谁来处理，切多大
SmallVector<int64_t> getNpuTileSizes(linalg::GenericOp op) {
  // 获取 library_call 属性
  auto libCall = op->getAttrOfType<StringAttr>("library_call");
  if (!libCall)
    return {};

  StringRef opName = libCall.getValue();

  // 获取 Op 的 Rank (维度)
  // 假设输入都是 RankedTensorType
  auto inputType = mlir::cast<RankedTensorType>(op.getInputs()[0].getType());
  unsigned rank = inputType.getRank();

  // === 策略表 ===
  // 假设 NPU 的 Gelu 单元一次只能算 32x32
  // 如果是 4 维 [N, C, H, W]，我们通常希望在最后两个维度切分
  SmallVector<int64_t> sizes(rank, 0); // 0 代表不切分 (保持原样)

  if (opName == "npu_gelu") {
    // 简单的启发式策略：最后两个维度切成 32
    if (rank >= 2) {
      sizes[rank - 1] = 32; // W
      sizes[rank - 2] = 32; // H
    }
    // 如果还有更多维度，比如 N 和 C，可以设为 1 (完全展开) 或者 0 (不切)
    // 这里假设我们只对空间维度分块
  } else if (opName == "npu_add") {
    // 假设 Add 单元大一点，支持 64
    if (rank >= 1)
      sizes[rank - 1] = 64;
  }

  // 如果你在 ONNXConversion 阶段已经把 tile size 挂在 Attribute 上了
  // 也可以直接读 Attribute，那就更通用了
  // if (auto attr = op->getAttrOfType<ArrayAttr>("npux.tile_sizes")) ...

  return sizes;
}

// === 2. Tiling Pattern ===
struct NpuElemWiseTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    // A. 卫语句：防止死循环
    if (op->hasAttr("npu.tiled")) {
      return failure();
    }

    // B. 检查是否是 NPU 算子 (通过 library_call 判断)
    auto targetAttr = op->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu") {
      return failure();
    }

    // C. 获取切分策略
    SmallVector<int64_t> rawTileSizes = getNpuTileSizes(op);
    // 如果策略返回空或者全是0，说明不需要分块
    if (rawTileSizes.empty() ||
        llvm::all_of(rawTileSizes, [](int64_t s) { return s == 0; })) {
      return failure();
    }

    // D. 配置 Tiling 选项

    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());

    SmallVector<OpFoldResult> tileSizes =
        getAsOpFoldResult(rewriter.getI64ArrayAttr(rawTileSizes));
    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    // E. 执行 Tiling
    // tileUsingSCF 会做以下事情：
    // 1. 生成 scf.for 循环嵌套
    // 2. 在循环内部生成 tensor.extract_slice
    // 3. 复制原来的 op 到循环内部，并连接 slice
    // 4. 生成 tensor.insert_slice
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult)) {
      return failure();
    }

    // F. 【关键】标记循环内部的新 Op 已处理
    // tilingResult->tiledOps包含了循环内部新生成的 linalg.generic
    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());

      // 可选：把 library_call 也传下去 (通常 scf::tileUsingSCF 会自动克隆
      // Attribute) 但为了保险可以检查一下
    }

    // G. 替换原 Op
    // replacements 是循环整体的返回值 (scf.for 的 result)
    rewriter.replaceOp(op, tilingResult->replacements);

    return success();
  }
};

// === 3. 定义 Pass ===
struct NpuElemWiseTilingPass
    : public PassWrapper<NpuElemWiseTilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuElemWiseTilingPass)

  StringRef getArgument() const override { return "npu-tiling-elemwise"; }
  StringRef getDescription() const override {
    return "Tile element-wise ops for NPU";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<NpuElemWiseTilingPattern>(context);

    // 使用贪婪重写驱动应用 Pattern
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    if (failed(applyPatternsGreedily(func, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

// 暴露创建函数
std::unique_ptr<Pass> npux::createNpuElemWiseTilingPass() {
  return std::make_unique<NpuElemWiseTilingPass>();
}

