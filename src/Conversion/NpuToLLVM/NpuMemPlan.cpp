//================================================
// src/Conversion/NpuToLLVM/NpuMemPlan.cpp
// this file implements npu memory planning pass
//================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/Compiler/NpuConfig.hpp"
#include "src/Dialect/Npux/NpuxOps.hpp"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <vector>

using namespace mlir;
using namespace npux;

namespace {

// ==========================================
// 辅助结构：内存块
// ==========================================
struct MemBlock {
  int64_t start;
  int64_t end; // end is exclusive: [start, end)
  int64_t size;

  // 用于排序，按起始地址从小到大
  bool operator<(const MemBlock &other) const { return start < other.start; }
};

// ==========================================
// 辅助函数：计算 MemRef 字节大小
// ==========================================
int64_t getMemRefSize(MemRefType type) {
  if (!type.hasStaticShape()) {
    // 静态规划目前不支持动态 Shape，或者需要给一个预估的最大值
    // 这里简单处理，如果不是静态，暂定为0或报错
    return 0;
  }

  int64_t size = 1;
  for (int64_t dim : type.getShape()) {
    size *= dim;
  }

  int64_t bitWidth = type.getElementTypeBitWidth();
  // 向上取整到字节，例如 i1 -> 1 byte
  int64_t elemSize = (bitWidth + 7) / 8;
  return size * elemSize;
}

// 内存对齐 (例如 32 字节对齐)
int64_t alignUp(int64_t addr, int64_t alignment) {
  if (alignment == 0)
    return addr;
  return (addr + alignment - 1) & ~(alignment - 1);
}

class NpuMemPlanPass
    : public PassWrapper<NpuMemPlanPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuMemPlanPass)

  StringRef getArgument() const override { return "npu-memory-plan"; }
  StringRef getDescription() const override {
    return "Plan NPU SRAM memory allocation with reuse";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto targetAttr = func->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu") {
      return;
    }

    // 配置参数
    const int64_t ALIGNMENT = 32;
    const int64_t SRAM_LIMIT = npux::NPUConfig::getInstance().getSpmSize();

    // 状态管理
    // occupiedBlocks: 当前时刻被占用的内存块列表，保持按地址排序
    std::vector<MemBlock> occupiedBlocks;

    // allocMap: 记录某个 SSA Value (sram_alloc result) 分配到了哪个地址
    // 这样在遇到 free(value) 时知道要释放哪个区间
    DenseMap<Value, MemBlock> allocMap;

    int64_t maxUsage = 0; // 记录峰值内存使用量，用于 Debug 或 Profiling

    // 我们使用 walk 来线性模拟执行流
    // 注意：对于复杂的控制流（如并行 scf.parallel），这种简单的 walk 可能不准确
    // 但对于 NPU Kernel 常见的串行铺排或简单 scf.for，这通常是可行的
    auto result = func.walk([&](Operation *op) -> WalkResult {
      // -------------------------------------------------------
      // 处理 Alloc: 寻找空闲区间 (First-Fit 策略)
      // -------------------------------------------------------
      if (auto allocOp = dyn_cast<npux::SramAllocOp>(op)) {
        auto type = cast<MemRefType>(allocOp.getMemref().getType());
        int64_t size = getMemRefSize(type);

        if (size == 0) {
          // 处理 0 大小或者动态大小的情况
          // 这里我们简单赋 0 或者跳过
        }

        // 寻找 Gap
        int64_t candidateAddr = 0;
        bool placed = false;

        // 尝试插入到现有块的缝隙中
        // 缝隙定义：[0, block[0].start) 或者 [block[i].end, block[i+1].start)

        // 1. 检查第 0 块之前的缝隙
        // (不需要特殊写，循环逻辑能覆盖，如果 occupiedBlocks 为空也不影响)

        // 我们遍历 occupiedBlocks，试图找到 candidateAddr
        // 每次我们尝试紧挨着前一个块结束的位置（对齐后）放置

        int64_t tryAddr = 0;

        // 这种插入逻辑需要 occupiedBlocks 始终有序
        for (const auto &block : occupiedBlocks) {
          int64_t alignedTryAddr = alignUp(tryAddr, ALIGNMENT);

          // 如果从 alignedTryAddr 开始放 size 大小，是否会碰到当前 block.start?
          if (alignedTryAddr + size <= block.start) {
            // 找到了空隙！
            candidateAddr = alignedTryAddr;
            placed = true;
            break;
          }

          // 如果放不下，更新 tryAddr 为当前 block 的结束位置，继续往后找
          tryAddr = std::max(tryAddr, block.end);
        }

        if (!placed) {
          // 如果中间没空隙，放在最后
          candidateAddr = alignUp(tryAddr, ALIGNMENT);
        }

        int64_t endAddr = candidateAddr + size;

        // 检查溢出
        if (endAddr > SRAM_LIMIT) {
          allocOp.emitError()
              << "SRAM Allocation failed: Out of memory. "
              << "Requested " << size << " bytes, "
              << "Needs end addr " << endAddr << ", Limit " << SRAM_LIMIT;
          return WalkResult::interrupt();
        }

        // 记录分配信息
        MemBlock newBlock = {candidateAddr, endAddr, size};
        allocMap[allocOp.getMemref()] = newBlock;

        // 插入并保持有序
        occupiedBlocks.push_back(newBlock);
        std::sort(occupiedBlocks.begin(), occupiedBlocks.end());

        // 记录峰值
        if (occupiedBlocks.back().end > maxUsage) {
          maxUsage = occupiedBlocks.back().end;
        }

        // 设置 IR 属性 "npu.offset"
        allocOp->setAttr(
            "npu.offset", IntegerAttr::get(IntegerType::get(&getContext(), 32),
                              candidateAddr));

        // Debug Log (可选)
        // llvm::errs() << "[Alloc] Addr: " << candidateAddr << " Size: " <<
        // size << "\n";
      }

      // -------------------------------------------------------
      // 处理 Free: 回收内存
      // -------------------------------------------------------
      else if (auto freeOp = dyn_cast<npux::SramFreeOp>(op)) {
        Value memref = freeOp.getSramMemref();

        if (allocMap.find(memref) == allocMap.end()) {
          // 可能是没经过 sram_alloc 的 memref (比如函数参数?)
          // 如果是函数参数传递进来的 SRAM Buffer，这里不需要
          // Plan，因为它在外部已分配 但如果是内部 alloc 却没找到，那就是 Bug
          // 简单起见，这里忽略
          return WalkResult::advance();
        }

        MemBlock blockToFree = allocMap[memref];

        // 从 occupiedBlocks 中移除
        // 使用 erase-remove idiom
        auto it = std::remove_if(occupiedBlocks.begin(), occupiedBlocks.end(),
            [&](const MemBlock &b) { return b.start == blockToFree.start; });

        if (it != occupiedBlocks.end()) {
          occupiedBlocks.erase(it, occupiedBlocks.end());
          // Debug Log (可选)
          // llvm::errs() << "[Free]  Addr: " << blockToFree.start << "\n";
        }

        // 从 map 中移除
        allocMap.erase(memref);
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }

    llvm::errs() << "NPU MemPlan Completed. Peak SRAM Usage: " << maxUsage
                 << " / " << SRAM_LIMIT << "\n";
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuMemPlanPass() {
  return std::make_unique<NpuMemPlanPass>();
}