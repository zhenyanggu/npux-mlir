//================================================
//src/Conversion/NpuToLLVM/NpuMemPlan.cpp
//this file implements npu memory planning pass
//================================================

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "src/Dialect/Npux/NpuxOps.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "src/Pass/Passes.hpp"
#include "src/Compiler/NpuConfig.hpp"

using namespace mlir;

namespace {

// 获取 MemRef 的字节大小
int64_t getMemRefSize(MemRefType type) {
  int64_t size = 1;
  for (int64_t dim : type.getShape()) {
    size *= dim;
  }
  // 简单起见，假设 element type 也就是 1 byte (i8) 或 4 bytes (f32)
  // 严谨写法需用 DataLayout
  int64_t elemSize = type.getElementTypeBitWidth() / 8;
  if (elemSize == 0) elemSize = 1; // i1 or similar
  return size * elemSize;
}

// 内存对齐 (例如 32 字节对齐)
int64_t align(int64_t addr, int64_t alignment) {
  return (addr + alignment - 1) & ~(alignment - 1);
}

struct NpuMemPlanPass : public PassWrapper<NpuMemPlanPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuMemPlanPass)

  StringRef getArgument() const override {
    return "npu-memory-plan";
  }
  StringRef getDescription() const override {
    return "Plan NPU SRAM memory allocation for memref alloc ops";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    
    // 简单的线性分配器：current_offset 永远只增不减
    // 对于简单的 Tile 循环，这足够了，因为 Alloc Op 在循环体里只出现有限次。
    int64_t currentOffset = 0;
    const int64_t ALIGNMENT = 32; 

    func.walk([&](memref::AllocOp allocOp) {
      auto type = cast<MemRefType>(allocOp.getType());
      
      // 只处理 Space 2 (NPU SRAM)
      if (type.getMemorySpaceAsInt() != 2) return;

      // 1. 计算大小
      int64_t size = getMemRefSize(type);

      // 2. 分配当前 Offset
      int64_t allocatedAddr = align(currentOffset, ALIGNMENT);
      
      // 3. 将 Offset 附加为属性 "npu.offset"
      // 这样下一步 Lowering 到 LLVM 时，直接读这个属性就知道地址了
      allocOp->setAttr("npu.offset", 
                       IntegerAttr::get(IntegerType::get(&getContext(), 32), allocatedAddr));

      // 4. 更新 Offset 指针
      currentOffset = allocatedAddr + size;
      
      // 打印调试信息，让你看到分配结果
      // llvm::errs() << "Allocated SRAM: Offset " << allocatedAddr << " Size " << size << "\n";
    });
    
    // 可选：检查 total size 是否超过硬件 SRAM 上限 (例如 256KB)
    int64_t sramSize = npux::NPUConfig::getInstance().getSramSize();

    if (currentOffset > sramSize) {
      func.emitError("SRAM allocation exceeded limit!");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuMemPlanPass() {
  return std::make_unique<NpuMemPlanPass>();
}