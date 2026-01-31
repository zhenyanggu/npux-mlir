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
#include <chrono>

// ==========================================
// Profiling (Enable with -DNPU_PROFILE)
// ==========================================
#ifdef NPU_PROFILE
    #define NPU_PROFILE_LOG(fmt, ...) \
        fprintf(stdout, "[NPU_PROFILE] " fmt "\n", ##__VA_ARGS__)

    class ScopedTimer {
    public:
        explicit ScopedTimer(const char* name)
            : name_(name), start_(std::chrono::steady_clock::now()) {}
        ~ScopedTimer() {
            auto end = std::chrono::steady_clock::now();
            auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
            NPU_PROFILE_LOG("%s: %lld ns", name_, (long long)ns);
        }
    private:
        const char* name_;
        std::chrono::steady_clock::time_point start_;
    };
    
    // Profile 模式下的计时宏
    #define NPU_TIMER_TOTAL(func_name) ScopedTimer _total_timer(func_name "(total)")
    #define NPU_TIMER_SECTION_BEGIN(name) { ScopedTimer _sec_timer(name);
    #define NPU_TIMER_SECTION_END() }
#else
    #define NPU_PROFILE_LOG(fmt, ...) do {} while(0)
    
    // Release 模式下完全消除计时开销
    #define NPU_TIMER_TOTAL(func_name) ((void)0)
    #define NPU_TIMER_SECTION_BEGIN(name) 
    #define NPU_TIMER_SECTION_END()
#endif

// ==========================================
// Debug/Release Mode Configuration
// ==========================================

// #define NPU_DEBUG 

#ifndef NPU_CPU_WIDTH
#define NPU_CPU_WIDTH 64
#endif

#ifdef NPU_DEBUG
    #define NPU_LOG(fmt, ...) \
        fprintf(stdout, "[NPU_DEBUG] " fmt "\n", ##__VA_ARGS__)
    
    #define NPU_REG_LOG(offset, val, width) \
        fprintf(stdout, "[NPU_REG] WR%d Offset:0x%04X Val:0x%0llX\n", width, offset, (unsigned long long)val)
#else
    #define NPU_LOG(fmt, ...) do {} while(0)
    #define NPU_REG_LOG(offset, val, width) do {} while(0)
#endif

#define NPU_ERR(fmt, ...) fprintf(stderr, "[NPU_ERROR] " fmt "\n", ##__VA_ARGS__)

// ==========================================
// Constants & Definitions
// ==========================================

#define REG_MAP_OFFSET      0x10000000
#define DDR_MAP_SIZE        0x20000000 // 512MB
#define REG_MAP_SIZE        0x1000     // 4KB

// IOCTL Commands (Must match driver)
#define IOCTL_WAIT_IRQ      _IOR('N', 1, uint32_t)
#define IOCTL_GET_DDR_PADDR _IOR('N', 2, uint32_t)
#define IOCTL_RESET_DEV     _IO('N', 3)
#define IOCTL_SET_IRQ_MODE  _IOW('N', 4, uint32_t)

// IRQ Mode Constants
#define IRQ_MODE_KERNEL     0   // 内核ISR处理中断
#define IRQ_MODE_USERSPACE  1   // 用户态轮询处理

// ==========================================
// Hybrid Polling Configuration
// ==========================================
// 先轮询后中断策略：先快速轮询一定次数，若未完成则切换到中断等待
// 可根据实际硬件延迟调整以下参数

#ifndef NPU_POLL_SPIN_COUNT
#define NPU_POLL_SPIN_COUNT     1000    // 首轮自旋轮询次数（无延迟）
#endif

#ifndef NPU_POLL_YIELD_COUNT  
#define NPU_POLL_YIELD_COUNT    100     // 让出CPU的轮询次数
#endif

#ifndef NPU_POLL_YIELD_US
#define NPU_POLL_YIELD_US       1       // 每次让出的微秒数
#endif

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
    NPU_LOG("NpuRuntime Destroyed.");
}

bool NpuRuntime::init() {
    fd = open("/dev/npu_driver", O_RDWR);
    if (fd < 0) {
        perror("Failed to open /dev/npu_driver");
        return false;
    }

    // 1. 获取物理地址
    if (ioctl(fd, IOCTL_GET_DDR_PADDR, &data_phy_base) < 0) {
        data_phy_base = 0x60000000; 
        NPU_ERR("Warning: IOCTL_GET_DDR_PADDR failed, using hardcoded 0x60000000");
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

    NPU_LOG("NPU Runtime Initialized (Shadow Regs: Enabled).");
    init_allocator();
    
    // 4. [新增] 设置为用户态轮询模式（内核ISR不会ACK中断）
    ioctl(fd, IOCTL_SET_IRQ_MODE, IRQ_MODE_USERSPACE);
    
    // 5. [新增] 启动时执行一次复位，确保硬件和软件状态同步
    reset(); 
    
    return true;
}

// ----------------------------------------------------
// [新增] 核心复位实现
// ----------------------------------------------------
void NpuRuntime::reset() {
    NPU_LOG("Requesting NPU Hardware Reset...");
    
    // 1. 调用驱动接口执行硬件脉冲复位
    if (ioctl(fd, IOCTL_RESET_DEV) < 0) {
        // 在 Release 模式下可能需要根据 errno 判断是否要报警
        // 比如如果驱动太旧不支持该 IOCTL，返回 ENOTTY
        NPU_ERR("Failed to reset NPU hardware via IOCTL (errno=%d)", errno);
    } else {
        NPU_LOG("NPU Hardware Reset Pulse OK.");
    }

    // 2. [关键] 重置影子寄存器
    // 硬件复位后，内部寄存器归零，必须让 Runtime 知道这一点
    // 否则 Runtime 以为缓存值仍有效，导致配置不生效
    reset_shadows();
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
    NPU_REG_LOG(offset, val, 32);
    volatile uint32_t* reg_ptr = (volatile uint32_t*)((char*)regs_virt_base + offset);
    *reg_ptr = val;
}

void NpuRuntime::reg_write64(uint32_t offset, uint64_t val) {
    NPU_REG_LOG(offset, val, 64);

#if NPU_CPU_WIDTH == 64
    volatile uint64_t* reg_ptr = (volatile uint64_t*)((char*)regs_virt_base + offset);
    *reg_ptr = val;
#elif NPU_CPU_WIDTH == 32
    reg_write(offset, (uint32_t)(val & 0xFFFFFFFF));
    reg_write(offset + 4, (uint32_t)((val >> 32) & 0xFFFFFFFF));
#else
#error "NPU_CPU_WIDTH must be 32 or 64"
#endif
}

uint32_t NpuRuntime::reg_read(uint32_t offset) {
    volatile uint32_t* reg_ptr = (volatile uint32_t*)((char*)regs_virt_base + offset);
    uint32_t val = *reg_ptr;
    #ifdef NPU_DEBUG
    fprintf(stdout, "[NPU_REG] READ Offset:0x%04X Val:0x%08X\n", offset, val);
    #endif
    return val;
}

void NpuRuntime::reset_shadows() {
    // 0xFF 意味着所有缓存失效（假设没有合法配置全是 0xFF）
    // 或者，因为硬件复位后通常是0，我们也可以设为 0xFFFFFFFF 来强制第一次写入
    memset(&shadow, 0xFF, sizeof(ShadowRegs));
    NPU_LOG("Shadow registers cache cleared.");
}

void NpuRuntime::reg_write64_cached(uint32_t offset, uint64_t val, uint64_t* cache_ptr) {
    if (*cache_ptr != val) {
        #ifdef NPU_DEBUG
        fprintf(stdout, "[NPU_REG] CACHE_MISS Write Offset:0x%04X Val:0x%016llX\n", offset, (unsigned long long)val);
        #endif
        
        reg_write64(offset, val);
        *cache_ptr = val;
    } else {
        // NPU_LOG("Cache Hit Offset:0x%04X", offset);
    }
}

// 检查中断是否挂起（读取 IPR 寄存器）- 强制内联
inline __attribute__((always_inline)) bool NpuRuntime::check_irq_pending() {
    volatile uint32_t* reg_ptr = (volatile uint32_t*)((char*)regs_virt_base + RegOffset::IPR);
    return (*reg_ptr != 0);
}

// 清除中断（写 IAR 寄存器）- 强制内联
inline __attribute__((always_inline)) void NpuRuntime::ack_irq() {
    volatile uint32_t* reg_ptr = (volatile uint32_t*)((char*)regs_virt_base + RegOffset::IAR);
    *reg_ptr = 0xFFFFFFFF;
}

void NpuRuntime::wait_irq() {
    NPU_TIMER_SECTION_BEGIN("wait_irq")
    NPU_LOG("Waiting for IRQ (hybrid polling)...");
    
    // ========== Phase 1: 自旋轮询（无延迟，最低延迟路径）==========
    // 在用户态轮询期间，中断由用户态处理，内核ISR不ACK
    // 直接获取寄存器指针，避免每次循环重复计算
    volatile uint32_t* ipr_ptr = (volatile uint32_t*)((char*)regs_virt_base + RegOffset::IPR);
    
    // 循环展开：每次迭代检测4次，减少循环开销
    int i = 0;
    for (; i < NPU_POLL_SPIN_COUNT - 3; i += 4) {
        if (*ipr_ptr) { ack_irq(); NPU_LOG("IRQ received via spin polling (iter=%d)", i); NPU_TIMER_SECTION_END() return; }
        if (*ipr_ptr) { ack_irq(); NPU_LOG("IRQ received via spin polling (iter=%d)", i+1); NPU_TIMER_SECTION_END() return; }
        if (*ipr_ptr) { ack_irq(); NPU_LOG("IRQ received via spin polling (iter=%d)", i+2); NPU_TIMER_SECTION_END() return; }
        if (*ipr_ptr) { ack_irq(); NPU_LOG("IRQ received via spin polling (iter=%d)", i+3); NPU_TIMER_SECTION_END() return; }
    }
    // 处理剩余迭代
    for (; i < NPU_POLL_SPIN_COUNT; ++i) {
        if (*ipr_ptr) { ack_irq(); NPU_LOG("IRQ received via spin polling (iter=%d)", i); NPU_TIMER_SECTION_END() return; }
    }
    
    // ========== Phase 2: 让出式轮询（短暂sleep，减少CPU占用）==========
    for (int i = 0; i < NPU_POLL_YIELD_COUNT; ++i) {
        if (check_irq_pending()) {
            ack_irq();
            NPU_LOG("IRQ received via yield polling (iter=%d)", i);
            NPU_TIMER_SECTION_END()
            return;
        }
        usleep(NPU_POLL_YIELD_US);
    }
    
    // ========== Phase 3: 中断等待（回退到阻塞模式）==========
    // 【关键修复】在进入内核等待前，先清除可能已经挂起的中断，
    // 然后切换到内核模式，让内核ISR负责ACK后续中断
    ack_irq();
    
    // 切换到内核中断模式：从此刻起，内核ISR会ACK中断
    ioctl(fd, IOCTL_SET_IRQ_MODE, IRQ_MODE_KERNEL);
    
    NPU_LOG("Polling timeout, falling back to kernel IRQ wait...");
    uint32_t status = 0;
    int ret = ioctl(fd, IOCTL_WAIT_IRQ, &status);
    
    // 返回用户态轮询模式：为下一次 wait_irq 做准备
    ioctl(fd, IOCTL_SET_IRQ_MODE, IRQ_MODE_USERSPACE);
    
    if (ret < 0) {
        perror("Wait IRQ failed");
    }
    NPU_LOG("IRQ received via interrupt (status=0x%X).", status);
    NPU_TIMER_SECTION_END()
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
    NPU_TIMER_TOTAL("run_mvin");
    
    uint32_t phys_dram = virt_to_phys(cfg.host_ptr);
    NPU_TIMER_SECTION_BEGIN("run_mvin(pre_reg)")
    NPU_LOG("Running MVIN (HostPtr=%p, phy_dram=0x%x, SRAM=0x%x)", cfg.host_ptr, phys_dram, cfg.sram_addr);
    NPU_TIMER_SECTION_END()

    NPU_TIMER_SECTION_BEGIN("run_mvin(reg_write)")
    uint64_t val_dram = REG_FIELD(MVIN_CTRL0, DRAM_ADDR, phys_dram) |
                        REG_FIELD(MVIN_CTRL0, ROW_NUM, cfg.row_num - 1);
    reg_write64(RegOffset::MVIN_DRAM_ADDR, val_dram);
    
    uint64_t val_sram = REG_FIELD(MVIN_CTRL1, SRAM_ADDR, cfg.sram_addr) |
                        REG_FIELD(MVIN_CTRL1, COL_NUM, cfg.col_num - 1);
    reg_write64(RegOffset::MVIN_SRAM_ADDR, val_sram);
    
    uint64_t val_cfg = REG_FIELD(CFG_MVIN0, INPUT_TYPE, cfg.input_type) |
                       REG_FIELD(CFG_MVIN0, INPUT_PRECISION, cfg.precision) |
                       REG_FIELD(CFG_MVIN0, IS_QUANT, cfg.is_quant) |
                       REG_FIELD(CFG_MVIN0, DEST, cfg.dest) |
                       REG_FIELD(CFG_MVIN0, IS_BIAS, cfg.is_bias) |
                       REG_FIELD(CFG_MVIN0, SRAM_STRIDE, cfg.sram_stride) |
                       REG_FIELD(CFG_MVIN0, DRAM_STRIDE, cfg.dram_stride);
    reg_write64_cached(RegOffset::MVIN_CFG, val_cfg, &shadow.mvin_cfg);
    
    if (cfg.is_quant) {
        uint64_t val_quant = REG_FIELD(CFG_MVIN1, ZEROPOINT, cfg.quant_zero) |
                             REG_FIELD(CFG_MVIN1, SCALE, cfg.quant_scale) |
                             REG_FIELD(CFG_MVIN1, SCALE_SHIFT, cfg.quant_shift);
        reg_write64_cached(RegOffset::MVIN_QUANT, val_quant, &shadow.mvin_quant);
    }
    reg_write(RegOffset::START, BIT_START_DMA_MVIN);
    NPU_TIMER_SECTION_END()

    NPU_TIMER_SECTION_BEGIN("run_mvin(wait_irq)")
    wait_irq();
    NPU_TIMER_SECTION_END()
}

void NpuRuntime::run_mvout(const MvoutConfig& cfg) {
    NPU_TIMER_TOTAL("run_mvout");
    
    uint32_t phys_dram = virt_to_phys(cfg.host_ptr);
    NPU_TIMER_SECTION_BEGIN("run_mvout(pre_reg)")
    NPU_LOG("Running MVOUT (HostPtr=%p, phy_dram=0x%x, SRAM=0x%x)", cfg.host_ptr, phys_dram, cfg.sram_addr);
    NPU_TIMER_SECTION_END()
    
    NPU_TIMER_SECTION_BEGIN("run_mvout(reg_write)")
    uint64_t val_dram = REG_FIELD(MVOUT_CTRL0, DRAM_ADDR, phys_dram) |
                        REG_FIELD(MVOUT_CTRL0, ROW_NUM, cfg.row_num - 1);
    reg_write64(RegOffset::MVOUT_DRAM_ADDR, val_dram);
    
    uint64_t val_sram = REG_FIELD(MVOUT_CTRL1, SRAM_ADDR, cfg.sram_addr) |
                        REG_FIELD(MVOUT_CTRL1, COL_NUM, cfg.col_num - 1);
    reg_write64(RegOffset::MVOUT_SRAM_ADDR, val_sram);
    
    uint64_t val_cfg = REG_FIELD(CFG_MVOUT0, OUTPUT_TYPE, cfg.output_type) |
                       REG_FIELD(CFG_MVOUT0, OUTPUT_PRECISION, cfg.precision) |
                       REG_FIELD(CFG_MVOUT0, IS_QUANT, cfg.is_quant) |
                       REG_FIELD(CFG_MVOUT0, SOURCE, cfg.source) |
                       REG_FIELD(CFG_MVOUT0, SRAM_STRIDE, cfg.sram_stride) |
                       REG_FIELD(CFG_MVOUT0, DRAM_STRIDE, cfg.dram_stride);
    reg_write64_cached(RegOffset::MVOUT_CFG, val_cfg, &shadow.mvout_cfg);
    
    if (cfg.is_quant) {
        uint64_t val_quant = REG_FIELD(CFG_MVOUT1, ZEROPOINT, cfg.quant_zero) |
                             REG_FIELD(CFG_MVOUT1, SCALE, cfg.quant_scale) |
                             REG_FIELD(CFG_MVOUT1, SCALE_SHIFT, cfg.quant_shift);
        reg_write64_cached(RegOffset::MVOUT_QUANT, val_quant, &shadow.mvout_quant);
    }
    reg_write(RegOffset::START, BIT_START_DMA_MVOUT);
    NPU_TIMER_SECTION_END()

    NPU_TIMER_SECTION_BEGIN("run_mvout(wait_irq)")
    wait_irq();
    NPU_TIMER_SECTION_END()
}

void NpuRuntime::run_sfu(const SfuConfig& cfg) {
    NPU_TIMER_TOTAL("run_sfu");
    
    NPU_TIMER_SECTION_BEGIN("run_sfu(pre_reg)")
    NPU_LOG("Running SFU (Op=%d)", cfg.op_type);
    NPU_TIMER_SECTION_END()
    
    NPU_TIMER_SECTION_BEGIN("run_sfu(reg_write)")
    uint64_t val_cfg1 = REG_FIELD(CFG_SFU0, OP, cfg.op_type) |
                        REG_FIELD(CFG_SFU0, INT_TYPE, cfg.int_type) |
                        REG_FIELD(CFG_SFU0, IS_QUANT, cfg.is_quant) |
                        REG_FIELD(CFG_SFU0, OUT_ZP, cfg.output_zeropoint) |
                        REG_FIELD(CFG_SFU0, IN_ZP, cfg.input_zeropoint);
    reg_write64_cached(RegOffset::CFG_SFU_1, val_cfg1, &shadow.sfu_cfg1);

    uint64_t val_cfg2 = REG_FIELD(CFG_SFU1, IN_SCALE, cfg.input_scale) |
                        REG_FIELD(CFG_SFU1, IN_SHIFT, cfg.input_scale_shift) |
                        REG_FIELD(CFG_SFU1, OUT_SCALE, cfg.output_scale) |
                        REG_FIELD(CFG_SFU1, OUT_SHIFT, cfg.output_scale_shift);
    reg_write64_cached(RegOffset::CFG_SFU_2, val_cfg2, &shadow.sfu_cfg2);

    uint64_t val_input = REG_FIELD(SFU_EXE0, IN_ADDR, cfg.input_sram_addr) |
                         REG_FIELD(SFU_EXE0, COL, cfg.input_col_num - 1) |
                         REG_FIELD(SFU_EXE0, ROW, cfg.input_row_num - 1);
    reg_write64(RegOffset::SFU_INPUT, val_input);

    uint64_t val_output = REG_FIELD(SFU_EXE1, OUT_ADDR, cfg.output_sram_addr);
    reg_write64(RegOffset::SFU_OUTPUT, val_output);

    reg_write(RegOffset::START, BIT_START_SFU);
    NPU_TIMER_SECTION_END()

    NPU_TIMER_SECTION_BEGIN("run_sfu(wait_irq)")
    wait_irq();
    NPU_TIMER_SECTION_END()
}

void NpuRuntime::run_conv(const ConvConfig& cfg) {
    NPU_TIMER_TOTAL("run_conv");
    
    NPU_TIMER_SECTION_BEGIN("run_conv(pre_reg)")
    NPU_LOG("Running CONV (DataFlow=%d)", cfg.dataflow_mode);
    NPU_TIMER_SECTION_END()
    
    NPU_TIMER_SECTION_BEGIN("run_conv(reg_write)")
    // 1. Compute Config 1
    uint64_t val_cfg1 = REG_FIELD(CFG_COMPUTE0, DATAFLOW, cfg.dataflow_mode) |
                        REG_FIELD(CFG_COMPUTE0, PAD_L, cfg.pad_left) |
                        REG_FIELD(CFG_COMPUTE0, PAD_R, cfg.pad_right) |
                        REG_FIELD(CFG_COMPUTE0, PAD_T, cfg.pad_top) |
                        REG_FIELD(CFG_COMPUTE0, PAD_B, cfg.pad_bottom) |
                        REG_FIELD(CFG_COMPUTE0, PAD_MODE, cfg.pad_mode) |
                        REG_FIELD(CFG_COMPUTE0, WEIGHT_SHAPE, cfg.weight_shape_m1) |
                        REG_FIELD(CFG_COMPUTE0, WEIGHT_STRIDE, cfg.weight_stride_m1) |
                        REG_FIELD(CFG_COMPUTE0, WEIGHT_DILATION, cfg.weight_dilation_m1) |
                        REG_FIELD(CFG_COMPUTE0, IS_GROUP, cfg.is_group_conv) |
                        REG_FIELD(CFG_COMPUTE0, INT_TYPE, cfg.int_type) |
                        REG_FIELD(CFG_COMPUTE0, OPTYPE, cfg.op_type) |
                        REG_FIELD(CFG_COMPUTE0, ACCOUT_DEST, cfg.accout_dest) |
                        REG_FIELD(CFG_COMPUTE0, INPUTA_ZP, cfg.input_a_zeropoint) |
                        REG_FIELD(CFG_COMPUTE0, INPUTB_ZP, cfg.input_b_zeropoint);
    reg_write64_cached(RegOffset::CFG_COMPUTE_1, val_cfg1, &shadow.compute_cfg1);

    // 2. Compute Config 2
    uint64_t val_cfg2 = REG_FIELD(CFG_COMPUTE1, OUT_ZP, cfg.output_zeropoint) |
                        REG_FIELD(CFG_COMPUTE1, OUT_SCALE, cfg.quant_scale) |
                        REG_FIELD(CFG_COMPUTE1, OUT_SHIFT, cfg.quant_scaleshift);
    reg_write64_cached(RegOffset::CFG_COMPUTE_2, val_cfg2, &shadow.compute_cfg2);

    // 3. SA Input A
    uint64_t val_input_a = REG_FIELD(SA_IN_A, ADDR, cfg.input_a_addr) |
                           REG_FIELD(SA_IN_A, COL, cfg.input_a_col_num_m1) |
                           REG_FIELD(SA_IN_A, ROW, cfg.input_a_row_num_m1) |
                           REG_FIELD(SA_IN_A, STRIDE, cfg.input_a_stride);
    reg_write64(RegOffset::SA_INPUT_A, val_input_a);

    // 4. SA Input B
    uint64_t val_input_b = REG_FIELD(SA_IN_B, ADDR, cfg.input_b_addr) |
                           REG_FIELD(SA_IN_B, COL, cfg.input_b_col_num_m1) |
                           REG_FIELD(SA_IN_B, ROW, cfg.input_b_row_num_m1) |
                           REG_FIELD(SA_IN_B, STRIDE, cfg.input_b_stride);
    reg_write64(RegOffset::SA_INPUT_B, val_input_b);

    // 5. Accumulator 1
    uint64_t val_acc1 = REG_FIELD(CFG_ACCU0, BIASPSUM_ADDR, cfg.biaspsum_addr) |
                        REG_FIELD(CFG_ACCU0, STRIDE, cfg.biaspsum_stride) |
                        REG_FIELD(CFG_ACCU0, WIDTH, cfg.biaspsum_width) |
                        REG_FIELD(CFG_ACCU0, HEIGHT, cfg.biaspsum_height);
    reg_write64_cached(RegOffset::CFG_ACCU_1, val_acc1, &shadow.accu_cfg1);

    // 6. Accumulator 2
    uint64_t val_acc2 = REG_FIELD(CFG_ACCU1, OUT_ADDR, cfg.output_addr) |
                        REG_FIELD(CFG_ACCU1, OUT_STRIDE, cfg.output_stride) |
                        REG_FIELD(CFG_ACCU1, IS_ACCU, cfg.is_accumulate) |
                        REG_FIELD(CFG_ACCU1, RELU, cfg.relu_enable) |
                        REG_FIELD(CFG_ACCU1, RELU_TYPE, cfg.relu_type) |
                        REG_FIELD(CFG_ACCU1, IS_BIAS, cfg.is_bias);
    reg_write64_cached(RegOffset::CFG_ACCU_2, val_acc2, &shadow.accu_cfg2);

    // 7. Start SA
    reg_write(RegOffset::START, BIT_START_SA);
    NPU_TIMER_SECTION_END()

    NPU_TIMER_SECTION_BEGIN("run_conv(wait_irq)")
    wait_irq();
    NPU_TIMER_SECTION_END()
}

void NpuRuntime::run_gemm(const GemmConfig& cfg) {
    NPU_TIMER_TOTAL("run_gemm");
    
    NPU_TIMER_SECTION_BEGIN("run_gemm(pre_reg)")
    NPU_LOG("Running GEMM (DataFlow=%d)", cfg.dataflow);
    NPU_TIMER_SECTION_END()

    NPU_TIMER_SECTION_BEGIN("run_gemm(reg_write)")
    // 1. Compute Config 1
    uint64_t val_cfg1 = REG_FIELD(CFG_COMPUTE0, DATAFLOW, cfg.dataflow) |
                        REG_FIELD(CFG_COMPUTE0, INT_TYPE, cfg.int_type) |
                        REG_FIELD(CFG_COMPUTE0, OPTYPE, cfg.optype) |
                        REG_FIELD(CFG_COMPUTE0, ACCOUT_DEST, cfg.accout_dest) |
                        REG_FIELD(CFG_COMPUTE0, INPUTA_ZP, cfg.input_a_zeropoint) |
                        REG_FIELD(CFG_COMPUTE0, INPUTB_ZP, cfg.input_b_zeropoint);
    reg_write64_cached(RegOffset::CFG_COMPUTE_1, val_cfg1, &shadow.compute_cfg1);

    // 2. Compute Config 2
    uint64_t val_cfg2 = REG_FIELD(CFG_COMPUTE1, OUT_ZP, cfg.output_zeropoint) |
                        REG_FIELD(CFG_COMPUTE1, OUT_SCALE, cfg.output_scale) |
                        REG_FIELD(CFG_COMPUTE1, OUT_SHIFT, cfg.output_scaleshift);
    reg_write64_cached(RegOffset::CFG_COMPUTE_2, val_cfg2, &shadow.compute_cfg2);

    // 3. Accumulator 1
    uint64_t val_acc1 = REG_FIELD(CFG_ACCU0, BIASPSUM_ADDR, cfg.biaspsum_addr) |
                        REG_FIELD(CFG_ACCU0, STRIDE, cfg.biaspsum_stride) |
                        REG_FIELD(CFG_ACCU0, WIDTH, cfg.biaspsum_width) |
                        REG_FIELD(CFG_ACCU0, HEIGHT, cfg.biaspsum_height);
    reg_write64_cached(RegOffset::CFG_ACCU_1, val_acc1, &shadow.accu_cfg1);

    // 4. Accumulator 2
    uint64_t val_acc2 = REG_FIELD(CFG_ACCU1, OUT_ADDR, cfg.output_addr) |
                        REG_FIELD(CFG_ACCU1, OUT_STRIDE, cfg.output_stride) |
                        REG_FIELD(CFG_ACCU1, IS_ACCU, cfg.isaccu) |
                        REG_FIELD(CFG_ACCU1, RELU, cfg.relu) |
                        REG_FIELD(CFG_ACCU1, RELU_TYPE, cfg.relu_type) |
                        REG_FIELD(CFG_ACCU1, IS_BIAS, cfg.is_bias);
    reg_write64_cached(RegOffset::CFG_ACCU_2, val_acc2, &shadow.accu_cfg2);

    // 5. SA Input A
    uint64_t val_input_a = REG_FIELD(SA_IN_A, ADDR, cfg.input_a_addr) |
                           REG_FIELD(SA_IN_A, COL, cfg.input_a_col_num) |
                           REG_FIELD(SA_IN_A, ROW, cfg.input_a_row_num) |
                           REG_FIELD(SA_IN_A, STRIDE, cfg.input_a_stride);
    reg_write64(RegOffset::SA_INPUT_A, val_input_a);

    // 6. SA Input B
    uint64_t val_input_b = REG_FIELD(SA_IN_B, ADDR, cfg.input_b_addr) |
                           REG_FIELD(SA_IN_B, COL, cfg.input_b_col_num) |
                           REG_FIELD(SA_IN_B, ROW, cfg.input_b_row_num) |
                           REG_FIELD(SA_IN_B, STRIDE, cfg.input_b_stride);
    reg_write64(RegOffset::SA_INPUT_B, val_input_b);

    // 7. Start SA
    reg_write(RegOffset::START, BIT_START_SA);
    NPU_TIMER_SECTION_END()

    NPU_TIMER_SECTION_BEGIN("run_gemm(wait_irq)")
    wait_irq();
    NPU_TIMER_SECTION_END()
}

void NpuRuntime::run_matadd(const MataddConfig& cfg) {
    NPU_TIMER_TOTAL("run_matadd");
    
    NPU_TIMER_SECTION_BEGIN("run_matadd(pre_reg)")
    NPU_LOG("Running MATADD (A=0x%X, B=0x%X, Out=0x%X, Col=%d, Row=%d)", 
            cfg.input_a_addr, cfg.input_b_addr, cfg.output_addr,
            cfg.col_num, cfg.row_num);
    NPU_TIMER_SECTION_END()
    
    NPU_TIMER_SECTION_BEGIN("run_matadd(reg_write)")
    // 1. Compute Config 2 - Quantization output parameters
    uint64_t val_cfg2 = REG_FIELD(CFG_COMPUTE1, OUT_ZP, cfg.output_zeropoint) |
                        REG_FIELD(CFG_COMPUTE1, OUT_SCALE, cfg.output_scale) |
                        REG_FIELD(CFG_COMPUTE1, OUT_SHIFT, cfg.output_scaleshift);
    reg_write64_cached(RegOffset::CFG_COMPUTE_2, val_cfg2, &shadow.compute_cfg2);
    
    // 2. MATADD CTRL0 - Input A and B addresses
    uint64_t val_ctrl0 = REG_FIELD(MATADD_CTRL0, A_ADDR, cfg.input_a_addr) |
                         REG_FIELD(MATADD_CTRL0, B_ADDR, cfg.input_b_addr);
    reg_write64(RegOffset::MATADD_CTRL_0, val_ctrl0);
    
    // 3. MATADD CTRL1 - Output address and dimensions
    uint64_t val_ctrl1 = REG_FIELD(MATADD_CTRL1, OUT_ADDR, cfg.output_addr) |
                         REG_FIELD(MATADD_CTRL1, COL, cfg.col_num) |
                         REG_FIELD(MATADD_CTRL1, ROW, cfg.row_num);
    reg_write64(RegOffset::MATADD_CTRL_1, val_ctrl1);
    
    // 4. Start MATADD
    reg_write(RegOffset::START, BIT_START_MATADD);
    NPU_TIMER_SECTION_END()
    
    NPU_TIMER_SECTION_BEGIN("run_matadd(wait_irq)")
    wait_irq();
    NPU_TIMER_SECTION_END()
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
    NPU_LOG("Allocator initialized. Total size: 0x%X", DDR_MAP_SIZE);
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
            NPU_LOG("Allocated %zu bytes at offset 0x%lX", size, (uint8_t*)curr - (uint8_t*)data_virt_base);
            return (void*)((uint8_t*)curr + sizeof(BlockHeader));
        }
        curr = curr->next;
    }
    NPU_ERR("NPU OOM: Requested %zu", size);
    return nullptr;
}

void NpuRuntime::free(void* ptr) {
    if (!ptr) return;
    BlockHeader* block = (BlockHeader*)((uint8_t*)ptr - sizeof(BlockHeader));
    if (block->is_free) return;
    
    NPU_LOG("Freeing block at offset 0x%lX", (uint8_t*)ptr - (uint8_t*)data_virt_base);
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

void npu_reset() { // <--- [新增]
    if (g_npu_runtime) {
        g_npu_runtime->reset();
    }
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
    uint8_t dest, bool is_bias, bool is_quant, uint32_t quant_zero, uint16_t quant_scale, uint16_t quant_shift
) {
    if (g_npu_runtime) {
        MvinConfig cfg = {host_ptr, sram_addr, col_num, row_num, sram_stride, dram_stride,
                          precision, input_type, dest, is_bias, is_quant, quant_zero, quant_scale, quant_shift};
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
    uint8_t is_bias, uint32_t output_zeropoint, uint16_t quant_scale, uint16_t quant_scaleshift
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
            output_addr, output_stride, is_accumulate, relu_enable, relu_type, is_bias,
            output_zeropoint, quant_scale, quant_scaleshift
        };
        g_npu_runtime->run_conv(cfg);
    }
}

void npu_gemm_run(
    uint8_t dataflow, uint8_t int_type, uint8_t optype, uint8_t accout_dest,
    uint16_t input_a_zeropoint, uint16_t input_b_zeropoint,
    uint32_t output_zeropoint, uint16_t output_scale, uint16_t output_scaleshift,
    uint32_t biaspsum_addr, uint16_t biaspsum_stride, uint8_t biaspsum_width, uint8_t biaspsum_height,
    uint32_t output_addr, uint16_t output_stride, uint8_t isaccu, uint8_t relu, uint8_t relu_type,
    uint8_t is_bias,
    uint32_t input_a_addr, uint8_t input_a_col_num, uint8_t input_a_row_num, uint16_t input_a_stride,
    uint32_t input_b_addr, uint8_t input_b_col_num, uint8_t input_b_row_num, uint16_t input_b_stride
) {
    if (g_npu_runtime) {
        GemmConfig cfg = {
            dataflow, int_type, optype, accout_dest,
            input_a_zeropoint, input_b_zeropoint,
            output_zeropoint, output_scale, output_scaleshift,
            biaspsum_addr, biaspsum_stride, biaspsum_width, biaspsum_height,
            output_addr, output_stride, isaccu, relu, relu_type, is_bias,
            input_a_addr, input_a_col_num, input_a_row_num, input_a_stride,
            input_b_addr, input_b_col_num, input_b_row_num, input_b_stride
        };
        g_npu_runtime->run_gemm(cfg);
    }
}

void npu_matadd_run(
    uint32_t input_a_addr,
    uint32_t input_b_addr,
    uint32_t output_addr,
    uint8_t  col_num,
    uint8_t  row_num,
    uint32_t output_zeropoint,
    uint16_t output_scale,
    uint16_t output_scaleshift
) {
    if (g_npu_runtime) {
        MataddConfig cfg = {
            input_a_addr,
            input_b_addr,
            output_addr,
            col_num,
            row_num,
            output_zeropoint,
            output_scale,
            output_scaleshift
        };
        g_npu_runtime->run_matadd(cfg);
    }
}

void npu_dma_mvin_test(void* host_ptr, uint32_t sram_addr, uint16_t col_num, uint16_t row_num,
                       uint16_t sram_stride, uint16_t dram_stride, uint8_t precision, uint8_t input_type,
                       uint8_t dest, bool is_bias, bool is_quant, uint32_t quant_zero, uint16_t quant_scale, uint16_t quant_shift) {
    #ifdef NPU_DEBUG
    printf("[TEST] MVIN: Host=%p SRAM=0x%x Size=%dx%d\n", host_ptr, sram_addr, row_num, col_num);
    #endif
}

void npu_dma_mvout_test(void* host_ptr, uint32_t sram_addr, uint16_t col_num, uint16_t row_num,
                        uint16_t sram_stride, uint16_t dram_stride, uint8_t precision, uint8_t output_type,
                        uint8_t source, bool is_quant, uint32_t quant_zero, uint16_t quant_scale, uint16_t quant_shift) {
    #ifdef NPU_DEBUG
    printf("[TEST] MVOUT: Host=%p SRAM=0x%x Size=%dx%d\n", host_ptr, sram_addr, row_num, col_num);
    #endif
}

} // extern "C"