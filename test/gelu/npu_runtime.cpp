#include "npu_runtime.h"
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <sys/types.h> // 确保 off_t 定义

// 【修复 1】与驱动保持一致，使用 0x10000000
#define REG_MAP_OFFSET      0x10000000
#define DDR_MAP_SIZE        0x10000000 // 256MB

// IOCTL 定义
#define IOCTL_WAIT_IRQ      _IOR('N', 1, uint32_t)
#define IOCTL_GET_DDR_PADDR _IOR('N', 2, uint32_t)

// Global Instance for C Interface
static NpuRuntime* g_npu_runtime = nullptr;

NpuRuntime::NpuRuntime() : fd(-1), regs_virt_base(nullptr), data_virt_base(nullptr), free_list_head(nullptr) {}

NpuRuntime::~NpuRuntime() {
    if (regs_virt_base) munmap(regs_virt_base, 0x10000);
    if (data_virt_base) munmap(data_virt_base, DDR_MAP_SIZE);
    if (fd >= 0) close(fd);
}

bool NpuRuntime::init() {
    fd = open("/dev/npu_v2", O_RDWR);
    if (fd < 0) {
        perror("Failed to open /dev/npu_v2");
        return false;
    }

    // 1. 获取 DDR 物理基地址
    if (ioctl(fd, IOCTL_GET_DDR_PADDR, &data_phy_base) < 0) {
        data_phy_base = 0x30000000;
        std::cerr << "Warning: IOCTL_GET_DDR_PADDR failed, using hardcoded 0x30000000" << std::endl;
    }

    // 2. 映射寄存器 (Offset = 0x10000000)
    // 注意：explicitly cast offset to off_t ensures correct sizing
    regs_virt_base = mmap(NULL, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)REG_MAP_OFFSET);
    if (regs_virt_base == MAP_FAILED) {
        perror("MMAP Registers failed");
        close(fd); fd = -1; // Cleanup
        return false;
    }

    // 3. 映射 DDR 数据区 (Offset = 0)
    data_virt_base = mmap(NULL, DDR_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data_virt_base == MAP_FAILED) {
        perror("MMAP Data failed");
        // Cleanup previous map
        munmap(regs_virt_base, 0x10000); regs_virt_base = nullptr;
        close(fd); fd = -1;
        return false;
    }

    printf("NPU Runtime Initialized.\n");
    printf("  - Regs Virt: %p (Mapped at offset 0x%X)\n", regs_virt_base, REG_MAP_OFFSET);
    printf("  - Data Virt: %p (Mapped at offset 0x0)\n", data_virt_base);
    printf("  - Phys Base: 0x%08X\n", data_phy_base);
    
    // Initialize Memory Allocator
    init_allocator();

    return true;
}
void* NpuRuntime::get_memory_base() {
    return data_virt_base;
}

// 【核心】虚拟地址转物理地址
// 公式：物理地址 = 物理基地址 + (当前虚拟地址 - 虚拟基地址)
uint32_t NpuRuntime::virt_to_phys(void* ptr) {
    uintptr_t virt_addr = (uintptr_t)ptr;
    uintptr_t base_virt = (uintptr_t)data_virt_base;
    
    if (virt_addr < base_virt || virt_addr >= base_virt + DDR_MAP_SIZE) {
        throw std::runtime_error("Pointer is out of NPU managed memory range!");
    }
    
    return data_phy_base + (uint32_t)(virt_addr - base_virt);
}

void NpuRuntime::reg_write(uint32_t offset, uint32_t val) {
    volatile uint32_t* reg_ptr = (volatile uint32_t*)((char*)regs_virt_base + offset);
    printf("[REG] Write Offset 0x%02X <= 0x%08X\n", offset, val);
    *reg_ptr = val;
}

// 处理 64 位寄存器写 (低 32 位在 offset，高 32 位在 offset+4)
void NpuRuntime::reg_write64(uint32_t offset, uint64_t val) {
    reg_write(offset, (uint32_t)(val & 0xFFFFFFFF));
    reg_write(offset + 4, (uint32_t)((val >> 32) & 0xFFFFFFFF));
}

void NpuRuntime::wait_irq() {
    uint32_t status = 0;
    // 调用驱动的 Wait，挂起当前线程直到中断发生
    int ret = ioctl(fd, IOCTL_WAIT_IRQ, &status);
    if (ret < 0) {
        perror("Wait IRQ failed");
    }
    printf("GET IRQ\n");
}

// ==========================================
// 指令实现：将 HAL 逻辑翻译为 C++
// ==========================================

void NpuRuntime::run_mvin(const MvinConfig& cfg) {
    // 1. 地址转换
    uint32_t phys_dram = virt_to_phys(cfg.host_ptr);
    
    // 2. 填写寄存器 (完全对应你的 npu_hal.c)
    uint64_t val_dram = (uint64_t)phys_dram | ((uint64_t)cfg.row_num << 32);
    reg_write64(RegOffset::MVIN_DRAM_ADDR, val_dram);
    
    uint64_t val_sram = (uint64_t)cfg.sram_addr | ((uint64_t)cfg.col_num << 32);
    reg_write64(RegOffset::MVIN_SRAM_ADDR, val_sram);
    
    uint64_t val_cfg = ((uint64_t)cfg.input_type & 0x3) |
                       (((uint64_t)cfg.precision & 0x3) << 2) |
                       (((uint64_t)cfg.is_quant & 0x1) << 4) |
                       (((uint64_t)cfg.dest & 0x1) << 5) |
                       ((uint64_t)cfg.sram_stride << 32) |
                       ((uint64_t)cfg.dram_stride << 48);
    reg_write64(RegOffset::MVIN_CFG, val_cfg);
    
    if (cfg.is_quant) {
        uint64_t val_quant = (uint64_t)cfg.quant_zero |
                             ((uint64_t)cfg.quant_scale << 32) |
                             ((uint64_t)cfg.quant_shift << 48);
        reg_write64(RegOffset::MVIN_QUANT, val_quant);
    }
    
    // 3. 发射指令
    reg_write(RegOffset::START, BIT_START_DMA_MVIN);
    
    // 4. 等待完成
    wait_irq();
}

void NpuRuntime::run_mvout(const MvoutConfig& cfg) {
    uint32_t phys_dram = virt_to_phys(cfg.host_ptr);
    
    uint64_t val_dram = (uint64_t)phys_dram | ((uint64_t)cfg.row_num << 32);
    reg_write64(RegOffset::MVOUT_DRAM_ADDR, val_dram);
    
    uint64_t val_sram = (uint64_t)cfg.sram_addr | ((uint64_t)cfg.col_num << 32);
    reg_write64(RegOffset::MVOUT_SRAM_ADDR, val_sram);
    
    uint64_t val_cfg = ((uint64_t)cfg.output_type & 0x3) |
                       (((uint64_t)cfg.precision & 0x3) << 2) |
                       (((uint64_t)cfg.is_quant & 0x1) << 4) |
                       (((uint64_t)cfg.source & 0x1) << 5) |
                       ((uint64_t)cfg.sram_stride << 32) |
                       ((uint64_t)cfg.dram_stride << 48);
    reg_write64(RegOffset::MVOUT_CFG, val_cfg);
    
    if (cfg.is_quant) {
        uint64_t val_quant = (uint64_t)cfg.quant_zero |
                             ((uint64_t)cfg.quant_scale << 32) |
                             ((uint64_t)cfg.quant_shift << 48);
        reg_write64(RegOffset::MVOUT_QUANT, val_quant);
    }
    
    reg_write(RegOffset::START, BIT_START_DMA_MVOUT);
    wait_irq();
}

void NpuRuntime::run_sfu(const SfuConfig& cfg) {
    // 1. Configure SFU Config 1 (Reg 10)
    uint64_t val_cfg1 = ((uint64_t)cfg.op_type & 0x3F) |
                   (((uint64_t)cfg.int_type & 0x3) << 6) |
                   (((uint64_t)cfg.is_quant & 0x1) << 8) |
                   ((uint64_t)cfg.output_zeropoint << 16) |
                   ((uint64_t)cfg.input_zeropoint << 32);
    reg_write64(RegOffset::CFG_SFU_1, val_cfg1);

    // 2. Configure SFU Config 2 (Reg 11)
    uint64_t val_cfg2 = ((uint64_t)cfg.input_scale & 0xFFFF) |
                   ((uint64_t)cfg.input_scale_shift << 16) |
                   ((uint64_t)cfg.output_scale << 32) |
                   ((uint64_t)cfg.output_scale_shift << 48);
    reg_write64(RegOffset::CFG_SFU_2, val_cfg2);

    // 3. Configure SFU Input (Reg 20)
    uint64_t val_input = (cfg.input_sram_addr & 0xFFFFFFFF) |
                    ((uint64_t)cfg.input_col_num << 32) |
                    ((uint64_t)cfg.input_row_num << 48);
    reg_write64(RegOffset::SFU_INPUT, val_input);

    // 4. Configure SFU Output (Reg 21)
    uint64_t val_output = (cfg.output_sram_addr & 0xFFFFFFFF);
    reg_write64(RegOffset::SFU_OUTPUT, val_output);

    // 5. Start SFU
    reg_write(RegOffset::START, BIT_START_SFU);

    // 6. Wait for completion
    wait_irq();
}

// ==========================================
// Memory Allocator Implementation
// ==========================================

void NpuRuntime::init_allocator() {
    if (!data_virt_base) return;

    // Initialize the first block covering the entire 256MB
    // Total size = Header + Data.
    // Available Data Size = DDR_MAP_SIZE - sizeof(BlockHeader).
    
    free_list_head = (BlockHeader*)data_virt_base;
    free_list_head->size = DDR_MAP_SIZE - sizeof(BlockHeader);
    free_list_head->is_free = true;
    free_list_head->next = nullptr;
    free_list_head->prev = nullptr;
    
    printf("Allocator Initialized. Total Heap: %zu bytes\n", free_list_head->size);
}

void* NpuRuntime::alloc(size_t size) {
    // 1. Align the requested size to ALIGNMENT (64)
    size_t aligned_size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
    // 2. Find a suitable block (First Fit)
    BlockHeader* curr = free_list_head;
    while (curr) {
        if (curr->is_free && curr->size >= aligned_size) {
            // Found a block
            // Check if we can split it
            // We need enough space for the requested data + a new header + at least some data
            // Minimal block size = sizeof(BlockHeader) + ALIGNMENT
            
            if (curr->size >= aligned_size + sizeof(BlockHeader) + ALIGNMENT) {
                // Split
                BlockHeader* new_block = (BlockHeader*)((uint8_t*)curr + sizeof(BlockHeader) + aligned_size);
                
                new_block->size = curr->size - aligned_size - sizeof(BlockHeader);
                new_block->is_free = true;
                new_block->next = curr->next;
                new_block->prev = curr;
                
                if (curr->next) {
                    curr->next->prev = new_block;
                }
                
                curr->next = new_block;
                curr->size = aligned_size;
            }
            
            curr->is_free = false;
            
            // Return pointer to data (after header)
            return (void*)((uint8_t*)curr + sizeof(BlockHeader));
        }
        curr = curr->next;
    }
    
    std::cerr << "NPU Allocator: Out of Memory! Requested: " << size << std::endl;
    return nullptr;
}

void NpuRuntime::free(void* ptr) {
    if (!ptr) return;
    
    // Get header
    BlockHeader* block = (BlockHeader*)((uint8_t*)ptr - sizeof(BlockHeader));
    
    // Sanity check
    if (block->is_free) {
        std::cerr << "Double free detected!" << std::endl;
        return;
    }
    
    block->is_free = true;
    
    // Coalesce
    coalesce(block);
}

void NpuRuntime::coalesce(BlockHeader* block) {
    // Merge with next if free
    if (block->next && block->next->is_free) {
        block->size += sizeof(BlockHeader) + block->next->size;
        block->next = block->next->next;
        if (block->next) {
            block->next->prev = block;
        }
    }
    
    // Merge with prev if free
    if (block->prev && block->prev->is_free) {
        block->prev->size += sizeof(BlockHeader) + block->size;
        block->prev->next = block->next;
        if (block->next) {
            block->next->prev = block->prev;
        }
    }
}

// ==========================================
// C Interface for MLIR
// ==========================================

extern "C" {

int _mlir_ciface_npu_init() {
    if (g_npu_runtime) return 0; // Already initialized
    
    g_npu_runtime = new NpuRuntime();
    if (!g_npu_runtime->init()) {
        delete g_npu_runtime;
        g_npu_runtime = nullptr;
        return -1;
    }
    return 0;
}

void _mlir_ciface_npu_destroy() {
    if (g_npu_runtime) {
        delete g_npu_runtime;
        g_npu_runtime = nullptr;
    }
}

void* _mlir_ciface_npu_mem_alloc(size_t size) {
    if (!g_npu_runtime) {
        if (_mlir_ciface_npu_init() < 0) return nullptr;
    }
    return g_npu_runtime->alloc(size);
}

void _mlir_ciface_npu_mem_free(void* ptr) {
    if (g_npu_runtime) {
        g_npu_runtime->free(ptr);
    }
}

void _mlir_ciface_npu_dma_mvin(
    void* host_ptr,
    uint32_t sram_addr,
    uint16_t col_num,
    uint16_t row_num,
    uint16_t sram_stride,
    uint16_t dram_stride,
    uint8_t  precision,
    uint8_t  input_type,
    uint8_t  dest,
    bool     is_quant,
    uint32_t quant_zero,
    uint16_t quant_scale,
    uint16_t quant_shift
) {
    if (g_npu_runtime) {
        MvinConfig cfg;
        cfg.host_ptr = host_ptr;
        cfg.sram_addr = sram_addr;
        cfg.col_num = col_num;
        cfg.row_num = row_num;
        cfg.sram_stride = sram_stride;
        cfg.dram_stride = dram_stride;
        cfg.precision = precision;
        cfg.input_type = input_type;
        cfg.dest = dest;
        cfg.is_quant = is_quant;
        cfg.quant_zero = quant_zero;
        cfg.quant_scale = quant_scale;
        cfg.quant_shift = quant_shift;
        g_npu_runtime->run_mvin(cfg);
    }
}

void _mlir_ciface_npu_dma_mvout(
    void* host_ptr,
    uint32_t sram_addr,
    uint16_t col_num,
    uint16_t row_num,
    uint16_t sram_stride,
    uint16_t dram_stride,
    uint8_t  precision,
    uint8_t  output_type,
    uint8_t  source,
    bool     is_quant,
    uint32_t quant_zero,
    uint16_t quant_scale,
    uint16_t quant_shift
) {
    if (g_npu_runtime) {
        MvoutConfig cfg;
        cfg.host_ptr = host_ptr;
        cfg.sram_addr = sram_addr;
        cfg.col_num = col_num;
        cfg.row_num = row_num;
        cfg.sram_stride = sram_stride;
        cfg.dram_stride = dram_stride;
        cfg.precision = precision;
        cfg.output_type = output_type;
        cfg.source = source;
        cfg.is_quant = is_quant;
        cfg.quant_zero = quant_zero;
        cfg.quant_scale = quant_scale;
        cfg.quant_shift = quant_shift;
        g_npu_runtime->run_mvout(cfg);
    }
}

void _mlir_ciface_npu_sfu_run(
    uint8_t  op_type,
    uint8_t  int_type,
    bool     is_quant,
    uint32_t input_sram_addr,
    uint16_t input_col_num,
    uint16_t input_row_num,
    uint32_t output_sram_addr,
    uint32_t input_zeropoint,
    uint16_t output_zeropoint,
    uint16_t input_scale,
    uint16_t input_scale_shift,
    uint16_t output_scale,
    uint16_t output_scale_shift
) {
    if (g_npu_runtime) {
        SfuConfig cfg;
        cfg.op_type = op_type;
        cfg.int_type = int_type;
        cfg.is_quant = is_quant;
        cfg.input_sram_addr = input_sram_addr;
        cfg.input_col_num = input_col_num;
        cfg.input_row_num = input_row_num;
        cfg.output_sram_addr = output_sram_addr;
        cfg.input_zeropoint = input_zeropoint;
        cfg.output_zeropoint = output_zeropoint;
        cfg.input_scale = input_scale;
        cfg.input_scale_shift = input_scale_shift;
        cfg.output_scale = output_scale;
        cfg.output_scale_shift = output_scale_shift;
        g_npu_runtime->run_sfu(cfg);
    }
}

void _mlir_ciface_npu_dma_mvin_test(
    void* host_ptr,
    uint32_t sram_addr,
    uint16_t col_num,
    uint16_t row_num,
    uint16_t sram_stride,
    uint16_t dram_stride,
    uint8_t  precision,
    uint8_t  input_type,
    uint8_t  dest,
    bool     is_quant,
    uint32_t quant_zero,
    uint16_t quant_scale,
    uint16_t quant_shift
) {
    printf("[TEST] npu_dma_mvin called:\n");
    printf("  host_ptr: %p\n", host_ptr);
    printf("  sram_addr: 0x%x\n", sram_addr);
    printf("  col_num: %d\n", col_num);
    printf("  row_num: %d\n", row_num);
    printf("  sram_stride: %d\n", sram_stride);
    printf("  dram_stride: %d\n", dram_stride);
    printf("  precision: %d\n", precision);
    printf("  input_type: %d\n", input_type);
    printf("  dest: %d\n", dest);
    printf("  is_quant: %d\n", is_quant);
    if (is_quant) {
        printf("  quant_zero: %d\n", quant_zero);
        printf("  quant_scale: %d\n", quant_scale);
        printf("  quant_shift: %d\n", quant_shift);
    }
}

void _mlir_ciface_npu_dma_mvout_test(
    void* host_ptr,
    uint32_t sram_addr,
    uint16_t col_num,
    uint16_t row_num,
    uint16_t sram_stride,
    uint16_t dram_stride,
    uint8_t  precision,
    uint8_t  output_type,
    uint8_t  source,
    bool     is_quant,
    uint32_t quant_zero,
    uint16_t quant_scale,
    uint16_t quant_shift
) {
    printf("[TEST] npu_dma_mvout called:\n");
    printf("  host_ptr: %p\n", host_ptr);
    printf("  sram_addr: 0x%x\n", sram_addr);
    printf("  col_num: %d\n", col_num);
    printf("  row_num: %d\n", row_num);
    printf("  sram_stride: %d\n", sram_stride);
    printf("  dram_stride: %d\n", dram_stride);
    printf("  precision: %d\n", precision);
    printf("  output_type: %d\n", output_type);
    printf("  source: %d\n", source);
    printf("  is_quant: %d\n", is_quant);
    if (is_quant) {
        printf("  quant_zero: %d\n", quant_zero);
        printf("  quant_scale: %d\n", quant_scale);
        printf("  quant_shift: %d\n", quant_shift);
    }
}

} // extern "C"