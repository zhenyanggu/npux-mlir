//================================================
// src/Conversion/NpuToLLVM/NpuMemPlan.cpp
// this file implements npu memory planning pass
//================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "src/Compiler/NpuConfig.hpp"
#include "src/Dialect/Npux/NpuxOps.hpp"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <vector>
#include <string>
#include <llvm/Support/raw_ostream.h>

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
    return 0; // 静态规划暂不支持动态 Shape
  }

  int64_t size = 1;
  for (int64_t dim : type.getShape()) {
    size *= dim;
  }

  int64_t bitWidth = type.getElementTypeBitWidth();
  int64_t elemSize = (bitWidth + 7) / 8; // 向上取整到字节
  return size * elemSize;
}

// 内存对齐
int64_t alignUp(int64_t addr, int64_t alignment) {
  if (alignment == 0) return addr;
  return (addr + alignment - 1) & ~(alignment - 1);
}

// ==========================================
// 核心类：通用内存分配器
// ==========================================
class MemoryAllocator {
public:
  MemoryAllocator(StringRef name, int64_t sizeLimit, int64_t alignment)
      : name(name), limit(sizeLimit), alignment(alignment) {}

  // 尝试分配内存，并设置 op 的 "npu.offset" 属性
  LogicalResult allocate(Operation *op, Value memref) {
    auto type = cast<MemRefType>(memref.getType());
    int64_t size = getMemRefSize(type);

    if (size == 0) {
      // 0 大小或动态 Shape，设为 0 偏移，不占用空间
      setOffsetAttr(op, 0);
      return success();
    }

    // First-Fit 策略寻找空隙
    int64_t candidateAddr = 0;
    bool placed = false;
    int64_t tryAddr = 0;

    for (const auto &block : occupiedBlocks) {
      int64_t alignedTryAddr = alignUp(tryAddr, alignment);
      
      // 检查缝隙 [alignedTryAddr, block.start) 是否够大
      if (alignedTryAddr + size <= block.start) {
        candidateAddr = alignedTryAddr;
        placed = true;
        break;
      }
      tryAddr = std::max(tryAddr, block.end);
    }

    if (!placed) {
      candidateAddr = alignUp(tryAddr, alignment);
    }

    int64_t endAddr = candidateAddr + size;

    // 检查溢出
    if (endAddr > limit) {
      op->emitError()
          << "[" << name << "] Allocation failed: Out of memory. "
          << "Requested " << size << " bytes, "
          << "Needs end addr " << endAddr << ", Limit " << limit;
      //return failure();
    }

    // 记录分配
    MemBlock newBlock = {candidateAddr, endAddr, size};
    allocMap[memref] = newBlock;
    occupiedBlocks.push_back(newBlock);
    
    // 保持有序，方便下次查找
    std::sort(occupiedBlocks.begin(), occupiedBlocks.end());

    // 更新统计
    if (occupiedBlocks.back().end > maxUsage) {
      maxUsage = occupiedBlocks.back().end;
    }

    // 设置 IR 属性
    setOffsetAttr(op, candidateAddr);
    return success();
  }

  // 释放内存
  void deallocate(Value memref) {
    if (allocMap.find(memref) == allocMap.end()) {
      return; // 忽略未被追踪的 memref
    }

    MemBlock blockToFree = allocMap[memref];

    // 从 occupiedBlocks 中移除
    auto it = std::remove_if(occupiedBlocks.begin(), occupiedBlocks.end(),
                             [&](const MemBlock &b) { return b.start == blockToFree.start; });
    
    if (it != occupiedBlocks.end()) {
      occupiedBlocks.erase(it, occupiedBlocks.end());
    }

    allocMap.erase(memref);
  }

  int64_t getPeakUsage() const { return maxUsage; }
  int64_t getLimit() const { return limit; }

private:
  std::string name;
  int64_t limit;
  int64_t alignment;
  int64_t maxUsage = 0;

  std::vector<MemBlock> occupiedBlocks;
  DenseMap<Value, MemBlock> allocMap;

  void setOffsetAttr(Operation *op, int64_t offset) {
    op->setAttr("npu.offset", 
                IntegerAttr::get(IntegerType::get(op->getContext(), 32), offset));
  }
};


class NpuMemPlanPass
    : public PassWrapper<NpuMemPlanPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuMemPlanPass)

  StringRef getArgument() const override { return "npu-memory-plan"; }
  StringRef getDescription() const override {
    return "Plan NPU SRAM & ACC memory allocation independently";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto targetAttr = func->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu") {
      return;
    }

    auto &config = npux::NPUConfig::getInstance();
    
    // 1. 初始化两个独立的分配器
    // SPM: 也就是 SRAM (Space 2)
    // ACC: 也就是 Accumulator (Space 3)
    // 注意：请确保你的 NPUConfig 中有 getAccSize()，否则请在此处硬编码
    int64_t spmSize = config.getSpmSize(); 
    int64_t accSize = config.getAccSize(); // Default 4KB if config missing, or config.getAccSize();

    // Alignment: SPM 通常 32/64 byte 对齐用于 DMA，ACC 通常 4 byte 对齐用于 int32
    MemoryAllocator spmAllocator("SPM", spmSize, 32);
    MemoryAllocator accAllocator("ACC", accSize, 4);

    auto result = func.walk([&](Operation *op) -> WalkResult {
      // -------------------------------------------------------
      // Alloc Ops`
      // -------------------------------------------------------
      if (auto allocOp = dyn_cast<npux::SramAllocOp>(op)) {
        if (failed(spmAllocator.allocate(op, allocOp.getMemref())))
          return WalkResult::interrupt();
      }
      else if (auto allocOp = dyn_cast<npux::AccAllocOp>(op)) {
        if (failed(accAllocator.allocate(op, allocOp.getMemref())))
          return WalkResult::interrupt();
      }

      // -------------------------------------------------------
      // Free Ops
      // -------------------------------------------------------
      else if (auto freeOp = dyn_cast<npux::SramFreeOp>(op)) {
        spmAllocator.deallocate(freeOp.getSramMemref());
      }
      else if (auto freeOp = dyn_cast<npux::AccFreeOp>(op)) {
        accAllocator.deallocate(freeOp.getAccMemref());
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    std::string msg;
    llvm::raw_string_ostream os(msg);

    os << "[NPU MemPlan] Function: @" << func.getName() << "\n"
       << "  -> SPM Usage: " << spmAllocator.getPeakUsage() << " / " << spmAllocator.getLimit() 
       << " bytes (" << (spmAllocator.getPeakUsage() * 100 / std::max((int64_t)1, spmAllocator.getLimit())) << "%)\n"
       << "  -> ACC Usage: " << accAllocator.getPeakUsage() << " / " << accAllocator.getLimit() 
       << " bytes (" << (accAllocator.getPeakUsage() * 100 / std::max((int64_t)1, accAllocator.getLimit())) << "%)\n";

    llvm::errs() << os.str();
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuMemPlanPass() {
  return std::make_unique<NpuMemPlanPass>();
}