//=============================================================
// src/Conversion/NpuTiling/ElemWise.cpp
// this file is for elemwise op tiling pattern
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h" // 核心 Tiling 工具
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"


#include "src/Pass/Passes.hpp"
#include "src/Conversion/NpuTiling/ElemWise.hpp"
#include "src/Compiler/NpuConfig.hpp"

#define DEBUG_TYPE "npu-tiling"
using namespace mlir;

namespace {

  static void applyTileConfig(SmallVectorImpl<int64_t> &sizes, 
                            const std::vector<int64_t> &configSizes) {
  if (configSizes.empty()) return;

  int opRank = sizes.size();
  int configRank = configSizes.size();

  int opIdx = opRank - 1;
  int cfgIdx = configRank - 1;

  // 从最后一个维度向前遍历
  while (opIdx >= 0 && cfgIdx >= 0) {
    sizes[opIdx] = configSizes[cfgIdx];
    opIdx--;
    cfgIdx--;
  }
}


// === 1. 定义分块策略逻辑 ===
// 这里是策略的核心：根据 Op 的特点决定由谁来处理，切多大
SmallVector<int64_t> getNpuTileSizes(linalg::GenericOp op) {
  // 1. 获取 library_call 属性
  auto libCall = op->getAttrOfType<StringAttr>("library_call");
  if (!libCall) return {}; // 不是 NPU Op，不切分

  StringRef opName = libCall.getValue();

  // 2. 获取 Op 的 Rank
  auto inputType = mlir::dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
  if (!inputType) return {}; // 无法处理非 Ranked 类型
  unsigned rank = inputType.getRank();

  // 初始化为 0 (0 表示该维度不切分)
  SmallVector<int64_t> sizes(rank, 0);

  // 3. 获取全局配置单例
  auto &config = npux::NPUConfig::getInstance();

  // 4. 根据 Op 名字查找配置
  if (opName == "npu_gelu") {
    // 从 Config 读取 (CLI > JSON > Default)
    std::vector<int64_t> configSizes = config.getGeluTileSize();
    applyTileConfig(sizes, configSizes);
    
  } else if (opName == "npu_conv") {
    // 假设你有 getConvTileSize
    // std::vector<int64_t> configSizes = config.getConvTileSize();
    // applyTileConfig(sizes, configSizes);
  } 
  // ... 其他 Op 处理 ...

  // Debug 打印 (可选，只在 -debug 时显示)
  LLVM_DEBUG({
    llvm::dbgs() << "NPU Tiling " << opName << ": [";
    for (auto s : sizes) llvm::dbgs() << s << " ";
    llvm::dbgs() << "]\n";
  });

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

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult)) {
      return failure();
    }

    // F. 【关键】标记循环内部的新 Op 已处理
    // tilingResult->tiledOps包含了循环内部新生成的 linalg.generic
    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
    }

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


void npux::populateElemWiseTilingPatterns(RewritePatternSet &patterns,
                                    MLIRContext *context) {
  patterns.add<NpuElemWiseTilingPattern>(context);
}

std::unique_ptr<Pass> npux::createNpuElemWiseTilingPass() {
  return std::make_unique<NpuElemWiseTilingPass>();
}

