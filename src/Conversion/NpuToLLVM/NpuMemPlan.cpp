//================================================
// src/Conversion/NpuToLLVM/NpuMemPlan.cpp
// this file implements npu memory planning pass at Global/Module level
//================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h" // 引入 ModuleOp
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
// 辅助结构与分配器类保持不变 (MemBlock, getMemRefSize, MemoryAllocator)
// 为了简洁，这里略过重复定义的实现，逻辑与你提供的版本一致
// ==========================================

struct MemBlock {
  int64_t start;
  int64_t end; 
  int64_t size;
  bool operator<(const MemBlock &other) const { return start < other.start; }
};

int64_t getMemRefSize(MemRefType type) {
  if (!type.hasStaticShape()) return 0;
  int64_t size = 1;
  for (int64_t dim : type.getShape()) size *= dim;
  int64_t bitWidth = type.getElementTypeBitWidth();
  int64_t elemSize = (bitWidth + 7) / 8;
  return size * elemSize;
}

int64_t alignUp(int64_t addr, int64_t alignment) {
  if (alignment == 0) return addr;
  return (addr + alignment - 1) & ~(alignment - 1);
}

class MemoryAllocator {
public:
  MemoryAllocator(StringRef name, int64_t sizeLimit, int64_t alignment)
      : name(name.str()), limit(sizeLimit), alignment(alignment) {}

  LogicalResult allocate(Operation *op, Value memref) {
    auto type = cast<MemRefType>(memref.getType());
    int64_t size = getMemRefSize(type);
    if (size == 0) {
      setOffsetAttr(op, 0);
      return success();
    }

    int64_t candidateAddr = 0;
    bool placed = false;
    int64_t tryAddr = 0;

    for (const auto &block : occupiedBlocks) {
      int64_t alignedTryAddr = alignUp(tryAddr, alignment);
      if (alignedTryAddr + size <= block.start) {
        candidateAddr = alignedTryAddr;
        placed = true;
        break;
      }
      tryAddr = std::max(tryAddr, block.end);
    }
    if (!placed) candidateAddr = alignUp(tryAddr, alignment);

    int64_t endAddr = candidateAddr + size;
    if (endAddr > limit) {
      op->emitWarning() << "[" << name << "] Potential OOM: Needs " << endAddr << " bytes, Limit " << limit;
    }

    MemBlock newBlock = {candidateAddr, endAddr, size};
    allocMap[memref] = newBlock;
    occupiedBlocks.push_back(newBlock);
    std::sort(occupiedBlocks.begin(), occupiedBlocks.end());
    if (occupiedBlocks.back().end > maxUsage) maxUsage = occupiedBlocks.back().end;

    setOffsetAttr(op, candidateAddr);
    return success();
  }

  void deallocate(Value memref) {
    if (allocMap.find(memref) == allocMap.end()) return;
    MemBlock blockToFree = allocMap[memref];
    auto it = std::remove_if(occupiedBlocks.begin(), occupiedBlocks.end(),
                             [&](const MemBlock &b) { return b.start == blockToFree.start; });
    if (it != occupiedBlocks.end()) occupiedBlocks.erase(it, occupiedBlocks.end());
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
    op->setAttr("npu.offset", IntegerAttr::get(IntegerType::get(op->getContext(), 32), offset));
  }
};

// ==========================================
// 修改后的 Pass 类：作用于 ModuleOp
// ==========================================
class NpuMemPlanPass
    : public PassWrapper<NpuMemPlanPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuMemPlanPass)

  StringRef getArgument() const override { return "npu-memory-plan"; }
  StringRef getDescription() const override {
    return "Plan NPU SRAM & ACC memory allocation at Module level";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto &config = npux::NPUConfig::getInstance();
    
    // 初始化分配器（全局生命周期）
    int64_t spmSize = config.getSpmSize(); 
    int64_t accSize = config.getAccSize();

    MemoryAllocator spmAllocator("SPM", spmSize, 32);
    MemoryAllocator accAllocator("ACC", accSize, 4);

    // 遍历 Module 内的所有算子进行规划
    auto result = module.walk([&](Operation *op) -> WalkResult {
      // -------------------------------------------------------
      // Alloc Ops
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

    // 打印统计信息
    llvm::errs() << "[NPU Global MemPlan Report]\n"
                 << "  -> SPM Usage: " << spmAllocator.getPeakUsage() << " / " << spmAllocator.getLimit() 
                 << " bytes (" << (spmAllocator.getPeakUsage() * 100 / std::max((int64_t)1, spmAllocator.getLimit())) << "%)\n"
                 << "  -> ACC Usage: " << accAllocator.getPeakUsage() << " / " << accAllocator.getLimit() 
                 << " bytes (" << (accAllocator.getPeakUsage() * 100 / std::max((int64_t)1, accAllocator.getLimit())) << "%)\n";
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuMemPlanPass() {
  return std::make_unique<NpuMemPlanPass>();
}