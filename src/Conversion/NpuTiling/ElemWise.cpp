//=============================================================
// src/Conversion/NpuTiling/ElemWise.cpp
// this file is for elemwise op tiling pattern
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h" // 核心 Tiling 工具
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Pass/Passes.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

#define DEBUG_TYPE "npu-tiling"
using namespace mlir;
using namespace npux; 

namespace {
SmallVector<int64_t> calculateAutoElemWiseTile(
    linalg::GenericOp op, int64_t maxElems) {
  
  auto loopRanges = op.getStaticLoopRanges();
  int64_t rank = loopRanges.size();
  
  // 默认全部切分为 1 (最保守情况)
  SmallVector<int64_t> tileSizes(rank, 1);
  if (rank == 0) return tileSizes; // 处理 Scalar

  if (maxElems <= 0)
    return tileSizes;

  int64_t remainingElems = maxElems;

  // 2. 贪心策略：从最内层 (rank-1) 向最外层 (0) 填充
  // 最内层通常在内存中是连续的 (Row-Major)，优先填满能最大化 DMA 效率
  for (int i = rank - 1; i >= 0; --i) {
      int64_t dimSize = loopRanges[i];
      
      // 容错处理：如果是动态维度 (<=0)，保守设为 1
      if (dimSize <= 0) dimSize = 1; 

      if (remainingElems >= dimSize) {
          // SPM 容量足够放下当前整个维度
          tileSizes[i] = dimSize;
          remainingElems /= dimSize; 
      } else {
          // SPM 容量放不下当前整个维度了，全部分配给当前维度
          // 硬件对齐优化：如果 NPU 的 DMA 对 16 或 32 字节对齐敏感，可以在这里对齐
          int64_t tile = (remainingElems / 16) * 16; 
          if (tile == 0) tile = remainingElems; // 如果连 16 都不到，能放多少放多少

          tileSizes[i] = tile;
          remainingElems = 1; // 空间耗尽
          break; // 外层维度保持默认值 1
      }
  }

  return tileSizes;
}

SmallVector<int64_t> getElemWiseTileSizes(
    linalg::GenericOp op, StringRef opName) {
  auto &config = npux::NPUConfig::getInstance();
  int64_t spmSize = config.getSpmSize();
  int64_t accSize = config.getAccSize();

  auto outputType = cast<RankedTensorType>(op.getOutputs()[0].getType());
  int64_t outputElemBits = outputType.getElementType().getIntOrFloatBitWidth();
  int64_t outputElemBytes = std::max<int64_t>(1, outputElemBits / 8);
  int64_t numOperands = op.getNumDpsInputs() + op.getNumDpsInits();
  int64_t bytesPerIteration = std::max<int64_t>(1, numOperands * outputElemBytes);
  int64_t maxElems = spmSize / bytesPerIteration;

  if (opName == "npu_matadd") {
    // MatAdd 输入在 ACC（int32）: A + B 两路，各 4B/elem；输出在 SPM（int8）。
    int64_t maxByAcc = accSize / (2 * 4);
    int64_t maxBySpm = spmSize / std::max<int64_t>(1, outputElemBytes);
    maxElems = std::max<int64_t>(1, std::min(maxByAcc, maxBySpm));
  }

  // 默认获取自动计算的分块大小
  SmallVector<int64_t> tileSizes = calculateAutoElemWiseTile(op, maxElems);

  if (opName == "npu_matadd") {
    int64_t rank = tileSizes.size();
    // MATADD 硬件字段限制：row/col 均为 8bit。
    if (rank >= 1)
      tileSizes[rank - 1] = std::max<int64_t>(1, std::min<int64_t>(255, tileSizes[rank - 1]));
    if (rank >= 2)
      tileSizes[rank - 2] = std::max<int64_t>(1, std::min<int64_t>(255, tileSizes[rank - 2]));
  }

  // 日志打印 (动态拼接维度信息)
  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "Tiling [" << opName << "] (Auto, Any-Rank): SPM=" << spmSize
     << ", ACC=" << accSize << " Problem=[";
  
  auto loopRanges = op.getStaticLoopRanges();
  for (size_t i = 0; i < loopRanges.size(); ++i) {
      os << loopRanges[i] << (i == loopRanges.size() - 1 ? "" : ", ");
  }
  os << "] -> Tile=[";
  for (size_t i = 0; i < tileSizes.size(); ++i) {
      os << tileSizes[i] << (i == tileSizes.size() - 1 ? "" : ", ");
  }
  os << "]\n";
  llvm::errs() << os.str();

  return tileSizes;
}

// === 2. Tiling Pattern ===
struct NpuElemWiseTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    
    if (op->hasAttr("npu.tiled")) return failure();

    auto libCall = op->getAttrOfType<StringAttr>("library_call");
    if (!libCall)
      return failure();
    if (libCall.getValue() != "npu_gelu" &&
        libCall.getValue() != "npu_matadd") {
        return failure(); // 把机会留给 NpuConvTilingPattern
    }

    StringRef opName = libCall.getValue();

    SmallVector<int64_t> rawTileSizes = getElemWiseTileSizes(op, opName);

    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizes = getAsOpFoldResult(rewriter.getI64ArrayAttr(rawTileSizes));
    
    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult)) return failure();

    for (Operation *tiledOp : tilingResult->tiledOps) {
      tiledOp->setAttr("npu.tiled", rewriter.getUnitAttr());
    }

    // 【修复 2】: 稳健的 Peeling 逻辑
    // 我们必须确保从内向外 Peel，并且正确处理 Loop 结构的更新
    auto loops = tilingResult->loops;
    
    SmallVector<Value> finalResults = tilingResult->replacements;

    // 倒序遍历处理 Peeling (从内向外)
    for (int i = loops.size() - 1; i >= 0; --i) {
      auto loopOp = dyn_cast<scf::ForOp>(loops[i].getOperation());
      if (!loopOp) continue;

      scf::ForOp partialIteration;
      LogicalResult status = scf::peelForLoopAndSimplifyBounds(rewriter, loopOp, partialIteration);

      if (succeeded(status)) {
        // 1. 标记 Tail (可选)
        partialIteration->setAttr("npu.peeled_tail", rewriter.getUnitAttr());
        
        // 2. 【关键修复】更新替换值
        // 如果当前处理的是最外层循环 (index 0)，或者该循环的结果直接对应 Op 的结果
        // 我们必须把 finalResults 更新为 Tail Loop 的结果
        // 因为 Tail Loop 串在 Main Loop 后面，它才持有最终完整的数据
        if (i == 0) {
            finalResults = partialIteration->getResults();
        }
      }
    }

    // 使用更新后的结果进行替换
    rewriter.replaceOp(op, finalResults);
    return success();
  }
};


} // namespace

void npux::populateElemWiseTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuElemWiseTilingPattern>(context);
}
