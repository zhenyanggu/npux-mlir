//=============================================================
// src/Conversion/NpuTiling/ElemWise.cpp
// this file is for elemwise op tiling pattern
//=============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h" // 核心 Tiling 工具
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include <algorithm>

#include "src/Pass/Passes.hpp"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

#define DEBUG_TYPE "npu-tiling"
using namespace mlir;
using namespace npux; 

namespace {

// ============================================================================
// 核心逻辑修改：区分普通的 SPM 算子 (如 GeLU) 和跨存储算子 (如 MatAdd)
// ============================================================================
SmallVector<int64_t> calculateAutoElemWiseTile(
    linalg::GenericOp op, StringRef opName, int64_t spmSize, int64_t accSize) {
  
  auto loopRanges = op.getStaticLoopRanges();
  int64_t rank = loopRanges.size();
  
  // 默认全部切分为 1 (最保守情况)
  SmallVector<int64_t> tileSizes(rank, 1);
  if (rank == 0) return tileSizes; // 处理 Scalar

  auto outputType = cast<RankedTensorType>(op.getOutputs()[0].getType());
  int64_t bitWidth = outputType.getElementType().getIntOrFloatBitWidth();
  int64_t bytesPerElem = std::max<int64_t>(1, bitWidth / 8);

  int64_t maxElems = 0;

  // 1. 根据算子类型计算最大可容纳的元素个数 (考虑到不同的内存映射)
  if (opName == "npu_matadd") {
    // MatAdd: 两个输入在 ACC (占据 2 份容量)，输出在 SPM (占据 1 份容量)
    // 需要同时满足 ACC 和 SPM 的限制，取短板
    int64_t maxElemsByAcc = accSize / (2 * 4 * bytesPerElem);//acc的精度是int32
    int64_t maxElemsBySpm = spmSize / (1 * bytesPerElem);
    maxElems = std::min(maxElemsByAcc, maxElemsBySpm);
  } else {
    // 默认情况 (如 GeLU): 全部放在 SPM 中
    int64_t numOperands = op.getNumDpsInputs() + op.getNumDpsInits(); 
    maxElems = spmSize / (numOperands * bytesPerElem);
  }

  if (maxElems <= 0) return tileSizes; // 极度受限时的保护

  // MatAdd 的 shape 参数按 "-1" 语义编码到 8-bit 寄存器：
  // 0 <= col_num_m1 <= 255, 0 <= row_num_m1 <= 255，
  // 即实际尺寸支持 1..256。
  // 其中 col_num 取最后一维，row_num 取其余维度乘积。
  if (opName == "npu_matadd") {
    constexpr int64_t kMataddActualMax = 256;
    SmallVector<int64_t> dims(rank, 1);
    for (int64_t i = 0; i < rank; ++i) {
      dims[i] = loopRanges[i] > 0 ? loopRanges[i] : 1;
    }

    int64_t colTile =
        std::min<int64_t>({dims.back(), maxElems, kMataddActualMax});
    colTile = std::max<int64_t>(1, colTile);
    tileSizes[rank - 1] = colTile;

    int64_t rowBudgetByMem = std::max<int64_t>(1, maxElems / colTile);
    int64_t rowBudget = std::min<int64_t>(kMataddActualMax, rowBudgetByMem);
    int64_t rowProduct = 1;

    for (int64_t i = rank - 2; i >= 0; --i) {
      int64_t dimSize = dims[i];
      int64_t maxForDim = std::max<int64_t>(1, rowBudget / rowProduct);
      int64_t tile = std::min<int64_t>(dimSize, maxForDim);
      tileSizes[i] = tile;
      rowProduct *= tile;
      if (rowProduct >= rowBudget)
        break;
    }
    return tileSizes;
  }

  int64_t remainingElems = maxElems;

  // 2. 贪心策略：从最内层 (rank-1) 向最外层 (0) 填充
  // 最内层通常在内存中是连续的 (Row-Major)，优先填满能最大化 DMA 效率
  for (int i = rank - 1; i >= 0; --i) {
      int64_t dimSize = loopRanges[i];
      
      // 容错处理：如果是动态维度 (<=0)，保守设为 1
      if (dimSize <= 0) dimSize = 1; 

      if (remainingElems >= dimSize) {
          // 容量足够放下当前整个维度
          tileSizes[i] = dimSize;
          remainingElems /= dimSize; 
      } else {
          // 容量放不下当前整个维度了，全部分配给当前维度
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

SmallVector<int64_t> getElemWiseTileSizes(linalg::GenericOp op, StringRef opName) {
  auto &config = npux::NPUConfig::getInstance();
  int64_t spmSize = config.getSpmSize();
  int64_t accSize = config.getAccSize();

  // 获取自动计算的分块大小，把 opName 和 accSize 传进去
  SmallVector<int64_t> tileSizes = calculateAutoElemWiseTile(op, opName, spmSize, accSize);

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
    if (!libCall) return failure();

    StringRef opName = libCall.getValue();
    
    // 【修改点】：允许 npu_gelu 和 npu_matadd 通过
    if (opName != "npu_gelu" && opName != "npu_matadd") {
        return failure(); // 把机会留给其他 Tiling Pattern (如 Conv)
    }

    SmallVector<int64_t> rawTileSizes = getElemWiseTileSizes(op, opName);
    auto loopRanges = op.getStaticLoopRanges();

    auto tilingInterfaceOp = llvm::cast<TilingInterface>(op.getOperation());
    SmallVector<OpFoldResult> tileSizes = getAsOpFoldResult(rewriter.getI64ArrayAttr(rawTileSizes));
    
    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(tilingResult)) return failure();

    for (auto loop : tilingResult->loops) {
      loop->setAttr("npu.target", rewriter.getStringAttr("npu"));
    }

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
