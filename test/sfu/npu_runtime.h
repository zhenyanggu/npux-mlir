#ifndef NPU_RUNTIME_H
#define NPU_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

// ==========================================
// 寄存器与硬件定义
// ==========================================

namespace RegOffset {
    const uint32_t START          = 0xB0;
    const uint32_t IAR            = 0xB8; // Interrupt Ack Register
    const uint32_t MER            = 0xC0; // Message Enable Register
    const uint32_t IER            = 0xC8; // Interrupt Enable Register
    const uint32_t IPR            = 0xD8; // Interrupt Pending Register
    
    // MVIN Registers
    const uint32_t MVIN_DRAM_ADDR = 0x00; 
    const uint32_t MVIN_SRAM_ADDR = 0x08;
    const uint32_t MVIN_CFG       = 0x20;
    const uint32_t MVIN_QUANT     = 0x28;
    
    // MVOUT Registers
    const uint32_t MVOUT_DRAM_ADDR = 0x10;
    const uint32_t MVOUT_SRAM_ADDR = 0x18;
    const uint32_t MVOUT_CFG       = 0x30;
    const uint32_t MVOUT_QUANT     = 0x38;

    // SFU Registers
    const uint32_t CFG_SFU_1      = 0x50;
    const uint32_t CFG_SFU_2      = 0x58;
    const uint32_t SFU_INPUT      = 0xA0;
    const uint32_t SFU_OUTPUT     = 0xA8;

    // Compute (SA / Conv) Registers
    const uint32_t CFG_COMPUTE_1  = 0x40;
    const uint32_t CFG_COMPUTE_2  = 0x48;
    const uint32_t CFG_ACCU_1     = 0x60;
    const uint32_t CFG_ACCU_2     = 0x68;
    const uint32_t SA_INPUT_A     = 0x70;
    const uint32_t SA_INPUT_B     = 0x78;
}

// 启动位定义
#define BIT_START_DMA_MVIN  (1 << 0)
#define BIT_START_DMA_MVOUT (1 << 1)
#define BIT_START_SA        (1 << 2)
#define BIT_START_SFU       (1 << 5)

// SFU Opcodes
#define SFU_OP_SOFTMAX          0
#define SFU_OP_GELU             1
#define SFU_OP_LAYERNORM        2
#define SFU_OP_TRANSPOSE        8

// ==========================================
// 配置结构体定义
// ==========================================

struct MvinConfig {
    void* host_ptr;        // 虚拟地址指针
    uint32_t sram_addr;    
    uint16_t col_num;
    uint16_t row_num;
    uint16_t sram_stride;
    uint16_t dram_stride;
    uint8_t  precision;
    uint8_t  input_type;
    uint8_t  dest;
    bool     is_quant;
    uint32_t quant_zero;
    uint16_t quant_scale;
    uint16_t quant_shift;
};

struct MvoutConfig {
    void* host_ptr;      // 虚拟地址指针
    uint32_t sram_addr;
    uint16_t col_num;
    uint16_t row_num;
    uint16_t sram_stride;
    uint16_t dram_stride;
    uint8_t  precision;
    uint8_t  output_type;
    uint8_t  source;
    bool     is_quant;
    uint32_t quant_zero;
    uint16_t quant_scale;
    uint16_t quant_shift;
};

struct SfuConfig {
    uint8_t  op_type;            // SFU Opcode
    uint8_t  int_type;           // Data Type (0=int8, 1=int16, etc.)
    bool     is_quant;           // Is Quantized

    // Input/Output Config
    uint32_t input_sram_addr;
    uint16_t input_col_num;      // Width - 1
    uint16_t input_row_num;      // Height - 1
    uint32_t output_sram_addr;

    // Quant Parameters
    uint32_t input_zeropoint;
    uint16_t output_zeropoint;
    uint16_t input_scale;
    uint16_t input_scale_shift;
    uint16_t output_scale;
    uint16_t output_scale_shift;
};

struct ConvConfig {
    // Padding
    uint8_t pad_top;
    uint8_t pad_bottom;
    uint8_t pad_left;
    uint8_t pad_right;
    uint8_t pad_mode; // 0=Zero

    // Weight
    uint8_t weight_shape_m1; // 2 for 3x3
    uint8_t weight_stride_m1;
    uint8_t weight_dilation_m1;
    uint8_t is_group_conv;

    // Compute
    uint8_t int_type; // 0=int8
    uint8_t op_type; // 1=Conv (0=GEMM, 2=GEMV)
    uint8_t dataflow_mode; // 0=im2col & OS, 1=OS only
    uint8_t accout_dest; // 0=SPM, 1=ACC

    // Zeropoints
    uint16_t input_a_zeropoint;
    uint16_t input_b_zeropoint;

    // Input A (IFM)
    uint32_t input_a_addr;
    uint8_t input_a_col_num_m1;
    uint8_t input_a_row_num_m1;
    uint16_t input_a_stride;

    // Input B (Weights)
    uint32_t input_b_addr;
    uint8_t input_b_col_num_m1;
    uint8_t input_b_row_num_m1;
    uint16_t input_b_stride;

    // Accumulator
    uint8_t biaspsum_width;
    uint8_t biaspsum_height;
    uint32_t biaspsum_addr;
    uint16_t biaspsum_stride;

    // Output
    uint32_t output_addr;
    uint16_t output_stride;
    uint8_t is_accumulate;      // 0=No accumulate, 1=Accumulate with previous psum
    uint8_t relu_enable;        // 0=Disabled, 1=Enabled
    uint8_t relu_type;          // 0=relu, 1=relu6, 2=leaky(0.1), 3=leaky(0.2), 4=leaky(0.01)

    // Quantization
    uint32_t output_zeropoint;
    uint16_t quant_scale;
    uint16_t quant_scaleshift;
};

// ==========================================
// NpuRuntime 类定义
// ==========================================

class NpuRuntime {
public:
    NpuRuntime();
    ~NpuRuntime();

    // 初始化：打开驱动，进行 mmap
    bool init();

    // 内存管理基础接口
    void* get_memory_base(); 
    uint32_t get_memory_size();

    // 执行指令接口
    void run_mvin(const MvinConfig& cfg);
    void run_mvout(const MvoutConfig& cfg);
    void run_sfu(const SfuConfig& cfg);
    void run_conv(const ConvConfig& cfg);

    // --- Memory Allocator (Heap) ---
    void* alloc(size_t size);
    void free(void* ptr);

private:
    int fd;
    void* regs_virt_base;   // 寄存器空间的虚拟基地址
    void* data_virt_base;   // DDR 数据空间的虚拟基地址
    uint32_t data_phy_base; // DDR 数据空间的物理基地址

    // Allocator State
    static const size_t ALIGNMENT = 64; 
    
    struct alignas(64) BlockHeader {
        size_t size;       
        bool is_free;
        BlockHeader* next;
        BlockHeader* prev;
    };

    BlockHeader* free_list_head;
    void init_allocator();
    void coalesce(BlockHeader* block);

    // 辅助函数：虚拟地址转物理地址
    uint32_t virt_to_phys(void* ptr);
    
    // 基础寄存器读写
    void reg_write(uint32_t offset, uint32_t val);
    void reg_write64(uint32_t offset, uint64_t val);
    uint32_t reg_read(uint32_t offset);
    
    // ----------------------------------------------------
    // 【优化】影子寄存器逻辑
    // ----------------------------------------------------
    
    // 仅缓存配置类寄存器 (CFG, QUANT)
    struct ShadowRegs {
        uint64_t mvin_cfg;
        uint64_t mvin_quant;
        uint64_t mvout_cfg;
        uint64_t mvout_quant;
        uint64_t sfu_cfg1;
        uint64_t sfu_cfg2;
        uint64_t compute_cfg1;
        uint64_t compute_cfg2;
        uint64_t accu_cfg1;
        uint64_t accu_cfg2;
    } shadow;

    // 初始化/重置影子寄存器
    void reset_shadows();
    
    // 带缓存检查的写操作：只有值变化时才写硬件
    void reg_write64_cached(uint32_t offset, uint64_t val, uint64_t* cache_ptr);

    // 简单的中断等待 (Block until IRQ)
    void wait_irq(); 
};

// ==========================================
// C Interface (API)
// ==========================================

extern "C" {
    // Lifecycle
    int npu_init();
    void npu_destroy();
    
    // Memory
    void* npu_mem_alloc(size_t size);
    void npu_mem_free(void* ptr);

    // DMA Operations
    void npu_dma_mvin(
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
    );

    void npu_dma_mvout(
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
    );

    // SFU Operations
    void npu_sfu_run(
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
    );

    // Convolution (SA)
    void npu_conv_run(
        uint8_t pad_top,
        uint8_t pad_bottom,
        uint8_t pad_left,
        uint8_t pad_right,
        uint8_t pad_mode,
        uint8_t weight_shape_m1,
        uint8_t weight_stride_m1,
        uint8_t weight_dilation_m1,
        uint8_t is_group_conv,
        uint8_t int_type,
        uint8_t op_type,
        uint8_t dataflow_mode,
        uint8_t accout_dest,
        uint16_t input_a_zeropoint,
        uint16_t input_b_zeropoint,
        uint32_t input_a_addr,
        uint8_t input_a_col_num_m1,
        uint8_t input_a_row_num_m1,
        uint16_t input_a_stride,
        uint32_t input_b_addr,
        uint8_t input_b_col_num_m1,
        uint8_t input_b_row_num_m1,
        uint16_t input_b_stride,
        uint8_t biaspsum_width,
        uint8_t biaspsum_height,
        uint32_t biaspsum_addr,
        uint16_t biaspsum_stride,
        uint32_t output_addr,
        uint16_t output_stride,
        uint8_t is_accumulate,
        uint8_t relu_enable,
        uint8_t relu_type,
        uint32_t output_zeropoint,
        uint16_t quant_scale,
        uint16_t quant_scaleshift
    );

    // Test Interface
    void npu_dma_mvin_test(
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
    );

    void npu_dma_mvout_test(
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
    );
}

#endif // NPU_RUNTIME_H