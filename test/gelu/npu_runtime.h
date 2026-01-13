#ifndef NPU_RUNTIME_H
#define NPU_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

// 保持和你硬件定义一致的偏移量
// 注意：请根据你的 Verilog 代码核对这些 OFFSET
namespace RegOffset {
    const uint32_t START      = 0xB0;
    const uint32_t IAR        = 0xB8;
    const uint32_t MER        = 0xC0;
    const uint32_t IER        = 0xC8;
    const uint32_t IPR        = 0xD8;
    
    // MVIN Registers (假设偏移，请替换为真实值)
    const uint32_t MVIN_DRAM_ADDR = 0x00; 
    const uint32_t MVIN_SRAM_ADDR = 0x08;
    const uint32_t MVIN_CFG       = 0x20; // 你的代码注释提到是 0x20
    const uint32_t MVIN_QUANT     = 0x28;
    
    // MVOUT Registers
    const uint32_t MVOUT_DRAM_ADDR = 0x10; // 假设偏移
    const uint32_t MVOUT_SRAM_ADDR = 0x18;
    const uint32_t MVOUT_CFG       = 0x30; // 你的代码注释提到是 0x30
    const uint32_t MVOUT_QUANT     = 0x38;

    // SFU Registers
    const uint32_t CFG_SFU_1      = 0x50;
    const uint32_t CFG_SFU_2      = 0x58;
    const uint32_t SFU_INPUT      = 0xA0;
    const uint32_t SFU_OUTPUT     = 0xA8;
}

// 启动位定义
#define BIT_START_DMA_MVIN  (1 << 0)
#define BIT_START_DMA_MVOUT (1 << 1)
#define BIT_START_SFU       (1 << 5)

// SFU Opcodes
#define SFU_OP_SOFTMAX          0
#define SFU_OP_GELU             1
#define SFU_OP_LAYERNORM        2
#define SFU_OP_TRANSPOSE        8

// 结构体定义 (直接复用你的 HAL 定义，做 C++ 适配)
struct MvinConfig {
    void* host_ptr;        // [变化] 用户传入的是虚拟地址指针
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
    void* host_ptr;      // [变化] 虚拟地址指针
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

class NpuRuntime {
public:
    NpuRuntime();
    ~NpuRuntime();

    // 初始化：打开驱动，进行 mmap
    bool init();

    // 内存管理：简单封装
    // 在真实场景中，这里应该管理那 256MB 的保留内存
    void* get_memory_base(); 
    uint32_t get_memory_size();

    // 执行指令
    void run_mvin(const MvinConfig& cfg);
    void run_mvout(const MvoutConfig& cfg);
    void run_sfu(const SfuConfig& cfg);

    // --- Memory Allocator ---
    void* alloc(size_t size);
    void free(void* ptr);

private:
    int fd;
    void* regs_virt_base;  // 寄存器空间的虚拟基地址
    void* data_virt_base;  // DDR 数据空间的虚拟基地址
    uint32_t data_phy_base; // DDR 数据空间的物理基地址 (0x30000000)

    // Allocator State
    static const size_t ALIGNMENT = 64; // Align to 64 bytes (Cache line / AXI burst friendly)
    
    struct alignas(64) BlockHeader {
        size_t size;       // Size of the data block (aligned)
        bool is_free;
        BlockHeader* next;
        BlockHeader* prev;
    };

    BlockHeader* free_list_head;
    void init_allocator();
    void coalesce(BlockHeader* block);

    // 辅助函数：虚拟地址转物理地址
    uint32_t virt_to_phys(void* ptr);
    
    // 寄存器读写 helper
    void reg_write(uint32_t offset, uint32_t val);
    void reg_write64(uint32_t offset, uint64_t val);
    uint32_t reg_read(uint32_t offset);
    
    // 等待中断
    void wait_irq(); 
};

extern "C" {
    // C Interface for MLIR / Compiler
    int npu_init();
    void npu_destroy();
    void* npu_mem_alloc(size_t size);
    void npu_mem_free(void* ptr);

    // DMA Interface
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

    
    // Test Interface (Dummy Print)
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

#endif