#include "npu_runtime.h"
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <sys/types.h>

// 常量定义
#define REG_MAP_OFFSET      0x10000000
#define DDR_MAP_SIZE        0x10000000 // 256MB
#define REG_MAP_SIZE        0x10000    // 64KB

// IOCTL 定义
#define IOCTL_WAIT_IRQ      _IOR('N', 1, uint32_t)
#define IOCTL_GET_DDR_PADDR _IOR('N', 2, uint32_t)

// Global Instance
static NpuRuntime* g_npu_runtime = nullptr;

// ==========================================
// Lifecycle
// ==========================================

NpuRuntime::NpuRuntime() 
    : fd(-1), regs_virt_base(nullptr), data_virt_base(nullptr), free_list_head(nullptr) {
    reset_shadows();
}

NpuRuntime::~NpuRuntime() {
    if (regs_virt_base) munmap(regs_virt_base, REG_MAP_SIZE);
    if (data_virt_base) munmap(data_virt_base, DDR_MAP_SIZE);
    if (fd >= 0) close(fd);
}

bool NpuRuntime::init() {
    fd = open("/dev/npu_v2", O_RDWR);
    if (fd < 0) {
        perror("Failed to open /dev/npu_v2");
        return false;
    }

    // 1. 获取物理地址
    if (ioctl(fd, IOCTL_GET_DDR_PADDR, &data_phy_base) < 0) {
        data_phy_base = 0x30000000; 
        std::cerr << "Warning: IOCTL_GET_DDR_PADDR failed, using hardcoded 0x30000000" << std::endl;
    }

    // 2. 映射寄存器
    regs_virt_base = mmap(NULL, REG_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)REG_MAP_OFFSET);
    if (regs_virt_base == MAP_FAILED) {
        perror("MMAP Registers failed");
        close(fd); fd = -1;
        return false;
    }

    // 3. 映射数据区
    data_virt_base = mmap(NULL, DDR_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data_virt_base == MAP_FAILED) {
        perror("MMAP Data failed");
        munmap(regs_virt_base, REG_MAP_SIZE); regs_virt_base = nullptr;
        close(fd); fd = -1;
        return false;
    }

    printf("NPU Runtime Initialized (Shadow Regs: Enabled, Wait: IRQ Only).\n");
    init_allocator();
    reset_shadows();
    
    return true;
}

void* NpuRuntime::get_memory_base() {
    return data_virt_base;
}

uint32_t NpuRuntime::get_memory_size() {
    return DDR_MAP_SIZE;
}

// ==========================================
// Register Operations & Shadow Logic
// ==========================================

void NpuRuntime::reg_write(uint32_t offset, uint32_t val) {
    volatile uint32_t* reg_ptr = (volatile uint32_t*)((char*)regs_virt_base + offset);
    *reg_ptr = val;
}

void NpuRuntime::reg_write64(uint32_t offset, uint64_t val) {
    reg_write(offset, (uint32_t)(val & 0xFFFFFFFF));
    reg_write(offset + 4, (uint32_t)((val >> 32) & 0xFFFFFFFF));
}

uint32_t NpuRuntime::reg_read(uint32_t offset) {
    volatile uint32_t* reg_ptr = (volatile uint32_t*)((char*)regs_virt_base + offset);
    return *reg_ptr;
}

void NpuRuntime::reset_shadows() {
    // 将 shadow 值设为全F，确保首次执行时必定写入硬件
    memset(&shadow, 0xFF, sizeof(ShadowRegs));
}

// 【核心优化】带缓存的寄存器写
// 只有当新值与 shadow 中的旧值不同时，才真正发起 MMIO 写
void NpuRuntime::reg_write64_cached(uint32_t offset, uint64_t val, uint64_t* cache_ptr) {
    if (*cache_ptr != val) {
        reg_write64(offset, val);
        *cache_ptr = val; // 更新缓存
    }
}

void NpuRuntime::wait_irq() {
    uint32_t status = 0;
    // 阻塞直到中断发生
    int ret = ioctl(fd, IOCTL_WAIT_IRQ, &status);
    if (ret < 0) {
        perror("Wait IRQ failed");
    }
}

uint32_t NpuRuntime::virt_to_phys(void* ptr) {
    uintptr_t virt_addr = (uintptr_t)ptr;
    uintptr_t base_virt = (uintptr_t)data_virt_base;
    if (virt_addr < base_virt || virt_addr >= base_virt + DDR_MAP_SIZE) {
        throw std::runtime_error("Pointer out of NPU memory range");
    }
    return data_phy_base + (uint32_t)(virt_addr - base_virt);
}

// ==========================================
// Instruction Implementation
// ==========================================

void NpuRuntime::run_mvin(const MvinConfig& cfg) {
    uint32_t phys_dram = virt_to_phys(cfg.host_ptr);

    // 1. 地址类寄存器：每次操作地址通常都会变，不走 Shadow 缓存，直接写
    uint64_t val_dram = (uint64_t)phys_dram | (((uint64_t)cfg.row_num-1) << 32);
    reg_write64(RegOffset::MVIN_DRAM_ADDR, val_dram);
    
    uint64_t val_sram = (uint64_t)cfg.sram_addr | (((uint64_t)cfg.col_num-1) << 32);
    reg_write64(RegOffset::MVIN_SRAM_ADDR, val_sram);
    
    // 2. 配置类寄存器：使用 Shadow 缓存加速
    uint64_t val_cfg = ((uint64_t)cfg.input_type & 0x3) |
                       (((uint64_t)cfg.precision & 0x3) << 2) |
                       (((uint64_t)cfg.is_quant & 0x1) << 4) |
                       (((uint64_t)cfg.dest & 0x1) << 5) |
                       ((uint64_t)cfg.sram_stride << 32) |
                       ((uint64_t)cfg.dram_stride << 48);
    reg_write64_cached(RegOffset::MVIN_CFG, val_cfg, &shadow.mvin_cfg);
    
    // 3. 量化参数：使用 Shadow 缓存加速
    if (cfg.is_quant) {
        uint64_t val_quant = (uint64_t)cfg.quant_zero |
                             ((uint64_t)cfg.quant_scale << 32) |
                             ((uint64_t)cfg.quant_shift << 48);
        reg_write64_cached(RegOffset::MVIN_QUANT, val_quant, &shadow.mvin_quant);
    }
    
    // 4. 启动与等待
    reg_write(RegOffset::START, BIT_START_DMA_MVIN);
    wait_irq(); // 直接进入中断等待
}

void NpuRuntime::run_mvout(const MvoutConfig& cfg) {
    uint32_t phys_dram = virt_to_phys(cfg.host_ptr);
    
    // 1. 地址：直接写
    uint64_t val_dram = (uint64_t)phys_dram | (((uint64_t)cfg.row_num-1) << 32);
    reg_write64(RegOffset::MVOUT_DRAM_ADDR, val_dram);
    
    uint64_t val_sram = (uint64_t)cfg.sram_addr | (((uint64_t)cfg.col_num-1) << 32);
    reg_write64(RegOffset::MVOUT_SRAM_ADDR, val_sram);
    
    // 2. CFG：缓存写
    uint64_t val_cfg = ((uint64_t)cfg.output_type & 0x3) |
                       (((uint64_t)cfg.precision & 0x3) << 2) |
                       (((uint64_t)cfg.is_quant & 0x1) << 4) |
                       (((uint64_t)cfg.source & 0x1) << 5) |
                       ((uint64_t)cfg.sram_stride << 32) |
                       ((uint64_t)cfg.dram_stride << 48);
    reg_write64_cached(RegOffset::MVOUT_CFG, val_cfg, &shadow.mvout_cfg);
    
    // 3. Quant：缓存写
    if (cfg.is_quant) {
        uint64_t val_quant = (uint64_t)cfg.quant_zero |
                             ((uint64_t)cfg.quant_scale << 32) |
                             ((uint64_t)cfg.quant_shift << 48);
        reg_write64_cached(RegOffset::MVOUT_QUANT, val_quant, &shadow.mvout_quant);
    }
    
    reg_write(RegOffset::START, BIT_START_DMA_MVOUT);
    wait_irq();
}

void NpuRuntime::run_sfu(const SfuConfig& cfg) {
    // 1. SFU CFG 1：缓存写
    uint64_t val_cfg1 = ((uint64_t)cfg.op_type & 0x3F) |
                   (((uint64_t)cfg.int_type & 0x3) << 6) |
                   (((uint64_t)cfg.is_quant & 0x1) << 8) |
                   ((uint64_t)cfg.output_zeropoint << 16) |
                   ((uint64_t)cfg.input_zeropoint << 32);
    reg_write64_cached(RegOffset::CFG_SFU_1, val_cfg1, &shadow.sfu_cfg1);

    // 2. SFU CFG 2：缓存写
    uint64_t val_cfg2 = ((uint64_t)cfg.input_scale & 0xFFFF) |
                   ((uint64_t)cfg.input_scale_shift << 16) |
                   ((uint64_t)cfg.output_scale << 32) |
                   ((uint64_t)cfg.output_scale_shift << 48);
    reg_write64_cached(RegOffset::CFG_SFU_2, val_cfg2, &shadow.sfu_cfg2);

    // 3. Input/Output 地址：直接写
    uint64_t val_input = (cfg.input_sram_addr & 0xFFFFFFFF) |
                        (((uint64_t)cfg.input_col_num-1) << 32) |
                        (((uint64_t)cfg.input_row_num-1) << 48);
    reg_write64(RegOffset::SFU_INPUT, val_input);

    uint64_t val_output = (cfg.output_sram_addr & 0xFFFFFFFF);
    reg_write64(RegOffset::SFU_OUTPUT, val_output);

    reg_write(RegOffset::START, BIT_START_SFU);
    wait_irq();
}

void NpuRuntime::run_conv(const ConvConfig& cfg) {
    // 1. Compute Config 1
    uint64_t val_cfg1 = 0;
    val_cfg1 |= ((uint64_t)cfg.dataflow_mode & 0x1) << 0;        // [0]
    val_cfg1 |= ((uint64_t)cfg.pad_left & 0x3) << 1;             // [2:1]
    val_cfg1 |= ((uint64_t)cfg.pad_right & 0x3) << 3;            // [4:3]
    val_cfg1 |= ((uint64_t)cfg.pad_top & 0x3) << 5;              // [6:5]
    val_cfg1 |= ((uint64_t)cfg.pad_bottom & 0x3) << 7;           // [8:7]
    val_cfg1 |= ((uint64_t)cfg.pad_mode & 0x3) << 9;             // [10:9]
    val_cfg1 |= ((uint64_t)cfg.weight_shape_m1 & 0xF) << 11;     // [14:11]
    val_cfg1 |= ((uint64_t)cfg.weight_stride_m1 & 0x3) << 15;    // [16:15]
    val_cfg1 |= ((uint64_t)cfg.weight_dilation_m1 & 0x1F) << 17; // [21:17]
    val_cfg1 |= ((uint64_t)cfg.is_group_conv & 0x1) << 22;       // [22]
    val_cfg1 |= ((uint64_t)cfg.int_type & 0x3) << 23;            // [24:23]
    val_cfg1 |= ((uint64_t)cfg.op_type & 0x3) << 25;             // [26:25]
    val_cfg1 |= ((uint64_t)cfg.accout_dest & 0x1) << 27;         // [27]
    val_cfg1 |= ((uint64_t)cfg.input_a_zeropoint & 0xFFFF) << 32; // [47:32]
    val_cfg1 |= ((uint64_t)cfg.input_b_zeropoint & 0xFFFF) << 48; // [63:48]
    reg_write64_cached(RegOffset::CFG_COMPUTE_1, val_cfg1, &shadow.compute_cfg1);

    // 2. Compute Config 2
    uint64_t val_cfg2 = 0;
    val_cfg2 |= ((uint64_t)cfg.output_zeropoint & 0xFFFFFFFF) << 0;  // [31:0]
    val_cfg2 |= ((uint64_t)cfg.quant_scale & 0xFFFF) << 32;          // [47:32]
    val_cfg2 |= ((uint64_t)cfg.quant_scaleshift & 0xFFFF) << 48;     // [63:48]
    reg_write64_cached(RegOffset::CFG_COMPUTE_2, val_cfg2, &shadow.compute_cfg2);

    // 3. SA Input A
    uint64_t val_input_a = 0;
    val_input_a |= ((uint64_t)cfg.input_a_addr & 0xFFFFFFFF) << 0;  // [31:0]
    val_input_a |= ((uint64_t)cfg.input_a_col_num_m1 & 0xFF) << 32; // [39:32]
    val_input_a |= ((uint64_t)cfg.input_a_row_num_m1 & 0xFF) << 40; // [47:40]
    val_input_a |= ((uint64_t)cfg.input_a_stride & 0xFFFF) << 48;   // [63:48]
    reg_write64(RegOffset::SA_INPUT_A, val_input_a);

    // 4. SA Input B
    uint64_t val_input_b = 0;
    val_input_b |= ((uint64_t)cfg.input_b_addr & 0xFFFFFFFF) << 0;  // [31:0]
    val_input_b |= ((uint64_t)cfg.input_b_col_num_m1 & 0xFF) << 32; // [39:32]
    val_input_b |= ((uint64_t)cfg.input_b_row_num_m1 & 0xFF) << 40; // [47:40]
    val_input_b |= ((uint64_t)cfg.input_b_stride & 0xFFFF) << 48;   // [63:48]
    reg_write64(RegOffset::SA_INPUT_B, val_input_b);

    // 5. Accumulator 1
    uint64_t val_acc1 = 0;
    val_acc1 |= ((uint64_t)cfg.biaspsum_addr & 0xFFFFFFFF) << 0;  // [31:0]
    val_acc1 |= ((uint64_t)cfg.biaspsum_stride & 0xFFFF) << 32;   // [47:32]
    val_acc1 |= ((uint64_t)cfg.biaspsum_width & 0xFF) << 48;      // [55:48]
    val_acc1 |= ((uint64_t)cfg.biaspsum_height & 0xFF) << 56;     // [63:56]
    reg_write64_cached(RegOffset::CFG_ACCU_1, val_acc1, &shadow.accu_cfg1);

    // 6. Accumulator 2
    uint64_t val_acc2 = 0;
    val_acc2 |= ((uint64_t)cfg.output_addr & 0xFFFFFFFF) << 0;    // [31:0]
    val_acc2 |= ((uint64_t)cfg.output_stride & 0xFFFF) << 32;     // [47:32]
    val_acc2 |= ((uint64_t)cfg.is_accumulate & 0x1) << 48;        // [48]
    val_acc2 |= ((uint64_t)cfg.relu_enable & 0x1) << 49;          // [49]
    val_acc2 |= ((uint64_t)cfg.relu_type & 0x7) << 50;            // [52:50]
    reg_write64_cached(RegOffset::CFG_ACCU_2, val_acc2, &shadow.accu_cfg2);

    // 7. Start SA
    reg_write(RegOffset::START, BIT_START_SA);
    wait_irq();
}

// ==========================================
// Allocator Implementation
// ==========================================

void NpuRuntime::init_allocator() {
    if (!data_virt_base) return;
    free_list_head = (BlockHeader*)data_virt_base;
    free_list_head->size = DDR_MAP_SIZE - sizeof(BlockHeader);
    free_list_head->is_free = true;
    free_list_head->next = nullptr;
    free_list_head->prev = nullptr;
}

void* NpuRuntime::alloc(size_t size) {
    size_t aligned_size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    BlockHeader* curr = free_list_head;
    
    while (curr) {
        if (curr->is_free && curr->size >= aligned_size) {
            if (curr->size >= aligned_size + sizeof(BlockHeader) + ALIGNMENT) {
                // Split
                BlockHeader* new_block = (BlockHeader*)((uint8_t*)curr + sizeof(BlockHeader) + aligned_size);
                new_block->size = curr->size - aligned_size - sizeof(BlockHeader);
                new_block->is_free = true;
                new_block->next = curr->next;
                new_block->prev = curr;
                if (curr->next) curr->next->prev = new_block;
                
                curr->next = new_block;
                curr->size = aligned_size;
            }
            curr->is_free = false;
            return (void*)((uint8_t*)curr + sizeof(BlockHeader));
        }
        curr = curr->next;
    }
    std::cerr << "NPU OOM: Requested " << size << std::endl;
    return nullptr;
}

void NpuRuntime::free(void* ptr) {
    if (!ptr) return;
    BlockHeader* block = (BlockHeader*)((uint8_t*)ptr - sizeof(BlockHeader));
    if (block->is_free) return;
    block->is_free = true;
    coalesce(block);
}

void NpuRuntime::coalesce(BlockHeader* block) {
    if (block->next && block->next->is_free) {
        block->size += sizeof(BlockHeader) + block->next->size;
        block->next = block->next->next;
        if (block->next) block->next->prev = block;
    }
    if (block->prev && block->prev->is_free) {
        block->prev->size += sizeof(BlockHeader) + block->size;
        block->prev->next = block->next;
        if (block->next) block->next->prev = block->prev;
    }
}

// ==========================================
// C Interface Implementation
// ==========================================

extern "C" {

int npu_init() {
    if (g_npu_runtime) return 0;
    g_npu_runtime = new NpuRuntime();
    if (!g_npu_runtime->init()) {
        delete g_npu_runtime; g_npu_runtime = nullptr;
        return -1;
    }
    return 0;
}

void npu_destroy() {
    if (g_npu_runtime) { delete g_npu_runtime; g_npu_runtime = nullptr; }
}

void* npu_mem_alloc(size_t size) {
    if (!g_npu_runtime && npu_init() < 0) return nullptr;
    return g_npu_runtime->alloc(size);
}

void npu_mem_free(void* ptr) {
    if (g_npu_runtime) g_npu_runtime->free(ptr);
}

void npu_dma_mvin(
    void* host_ptr, uint32_t sram_addr, uint16_t col_num, uint16_t row_num,
    uint16_t sram_stride, uint16_t dram_stride, uint8_t precision, uint8_t input_type,
    uint8_t dest, bool is_quant, uint32_t quant_zero, uint16_t quant_scale, uint16_t quant_shift
) {
    if (g_npu_runtime) {
        MvinConfig cfg = {host_ptr, sram_addr, col_num, row_num, sram_stride, dram_stride,
                          precision, input_type, dest, is_quant, quant_zero, quant_scale, quant_shift};
        g_npu_runtime->run_mvin(cfg);
    }
}

void npu_dma_mvout(
    void* host_ptr, uint32_t sram_addr, uint16_t col_num, uint16_t row_num,
    uint16_t sram_stride, uint16_t dram_stride, uint8_t precision, uint8_t output_type,
    uint8_t source, bool is_quant, uint32_t quant_zero, uint16_t quant_scale, uint16_t quant_shift
) {
    if (g_npu_runtime) {
        MvoutConfig cfg = {host_ptr, sram_addr, col_num, row_num, sram_stride, dram_stride,
                           precision, output_type, source, is_quant, quant_zero, quant_scale, quant_shift};
        g_npu_runtime->run_mvout(cfg);
    }
}

void npu_sfu_run(
    uint8_t op_type, uint8_t int_type, bool is_quant, uint32_t input_sram_addr,
    uint16_t input_col_num, uint16_t input_row_num, uint32_t output_sram_addr,
    uint32_t input_zeropoint, uint16_t output_zeropoint, uint16_t input_scale,
    uint16_t input_scale_shift, uint16_t output_scale, uint16_t output_scale_shift
) {
    if (g_npu_runtime) {
        SfuConfig cfg = {op_type, int_type, is_quant, input_sram_addr, input_col_num, input_row_num,
                         output_sram_addr, input_zeropoint, output_zeropoint, input_scale,
                         input_scale_shift, output_scale, output_scale_shift};
        g_npu_runtime->run_sfu(cfg);
    }
}

void npu_conv_run(
    uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left, uint8_t pad_right, uint8_t pad_mode,
    uint8_t weight_shape_m1, uint8_t weight_stride_m1, uint8_t weight_dilation_m1, uint8_t is_group_conv,
    uint8_t int_type, uint8_t op_type, uint8_t dataflow_mode, uint8_t accout_dest,
    uint16_t input_a_zeropoint, uint16_t input_b_zeropoint,
    uint32_t input_a_addr, uint8_t input_a_col_num_m1, uint8_t input_a_row_num_m1, uint16_t input_a_stride,
    uint32_t input_b_addr, uint8_t input_b_col_num_m1, uint8_t input_b_row_num_m1, uint16_t input_b_stride,
    uint8_t biaspsum_width, uint8_t biaspsum_height, uint32_t biaspsum_addr, uint16_t biaspsum_stride,
    uint32_t output_addr, uint16_t output_stride, uint8_t is_accumulate, uint8_t relu_enable, uint8_t relu_type,
    uint32_t output_zeropoint, uint16_t quant_scale, uint16_t quant_scaleshift
) {
    if (g_npu_runtime) {
        ConvConfig cfg = {
            pad_top, pad_bottom, pad_left, pad_right, pad_mode,
            weight_shape_m1, weight_stride_m1, weight_dilation_m1, is_group_conv,
            int_type, op_type, dataflow_mode, accout_dest,
            input_a_zeropoint, input_b_zeropoint,
            input_a_addr, input_a_col_num_m1, input_a_row_num_m1, input_a_stride,
            input_b_addr, input_b_col_num_m1, input_b_row_num_m1, input_b_stride,
            biaspsum_width, biaspsum_height, biaspsum_addr, biaspsum_stride,
            output_addr, output_stride, is_accumulate, relu_enable, relu_type,
            output_zeropoint, quant_scale, quant_scaleshift
        };
        g_npu_runtime->run_conv(cfg);
    }
}

// Test wrappers...
void npu_dma_mvin_test(void* host_ptr, uint32_t sram_addr, uint16_t col_num, uint16_t row_num,
                       uint16_t sram_stride, uint16_t dram_stride, uint8_t precision, uint8_t input_type,
                       uint8_t dest, bool is_quant, uint32_t quant_zero, uint16_t quant_scale, uint16_t quant_shift) {
    printf("[TEST] MVIN: Host=%p SRAM=0x%x Size=%dx%d\n", host_ptr, sram_addr, row_num, col_num);
}

void npu_dma_mvout_test(void* host_ptr, uint32_t sram_addr, uint16_t col_num, uint16_t row_num,
                        uint16_t sram_stride, uint16_t dram_stride, uint8_t precision, uint8_t output_type,
                        uint8_t source, bool is_quant, uint32_t quant_zero, uint16_t quant_scale, uint16_t quant_shift) {
    printf("[TEST] MVOUT: Host=%p SRAM=0x%x Size=%dx%d\n", host_ptr, sram_addr, row_num, col_num);
}

} // extern "C"