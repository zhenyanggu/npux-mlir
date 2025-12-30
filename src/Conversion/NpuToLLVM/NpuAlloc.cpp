//==============================================================
// src/Conversion/NpuToLLVM/NpuAlloc.cpp
// Unified Memory Allocation Pass for Host (DRAM) and NPU (SRAM)
//==============================================================

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "src/Pass/Passes.hpp"


using namespace mlir;

namespace {

struct NpuMemoryAllocationPass : public PassWrapper<NpuMemoryAllocationPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuMemoryAllocationPass)
    StringRef getArgument() const override { return "npu-memory-alloc"; }
    
    // SRAM 总大小 (例如 256KB)
    const int64_t SRAM_SIZE_LIMIT = 256 * 1024; 
    const int NPU_SRAM_SPACE = 1;

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        
        // 简单的线性分配器指针
        int64_t current_offset = 0;

        // 遍历函数内的每一个操作
        func.walk([&](memref::AllocOp allocOp) {
            MemRefType type = allocOp.getType();

            // 只处理 NPU SRAM (Space 1)
            if (type.getMemorySpaceAsInt() != NPU_SRAM_SPACE) return;

            // 1. 计算 buffer 大小
            // 注意：这里需要考虑 padding 和 alignment (通常 NPU 要求 32/64 字节对齐)
            int64_t sizeInBytes = type.getElementTypeBitWidth() / 8;
            for (auto dim : type.getShape()) {
                sizeInBytes *= dim;
            }
            
            // 64字节对齐
            int64_t alignment = 64;
            int64_t alignedSize = (sizeInBytes + alignment - 1) & ~(alignment - 1);

            // 2. 检查是否越界
            if (current_offset + alignedSize > SRAM_SIZE_LIMIT) {
                allocOp.emitError() << "NPU SRAM OOM! Needed " << alignedSize 
                                    << " bytes, but only " << (SRAM_SIZE_LIMIT - current_offset) 
                                    << " remaining.";
                return signalPassFailure();
            }

            // 3. 【核心】将计算出的 Offset 写入 Attribute
            // 以后 Lowering Pass 只要读这个属性就知道地址是多少了
            allocOp->setAttr("npu.sram_offset", 
                             IntegerAttr::get(IntegerType::get(&getContext(), 32), current_offset));

            // 4. 更新指针 (极简版：不回收，一直往后堆)
            // 进阶版在这里要结合 Liveness Analysis 做内存回收
            current_offset += alignedSize;
        });
        
        // 可选：打印一下总共用了多少 SRAM
        // llvm::outs() << "Total SRAM used: " << current_offset << " bytes\n";
    }
};


} // namespace
std::unique_ptr<Pass> npux::createNpuMemoryAllocationPass() {
    return std::make_unique<NpuMemoryAllocationPass>();
}

