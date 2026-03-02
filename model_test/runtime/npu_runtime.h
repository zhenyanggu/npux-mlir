#ifndef NPU_RUNTIME_H
#define NPU_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

// ==========================================
// 寄存器与硬件定义 (来自 RDL 自动生成)
// ==========================================
#include "npu_regs_compat.h"

// ==========================================
// 配置结构体定义
// ==========================================

struct MvinConfig {
    void* host_ptr;        // 虚拟地址指针
    uint32_t sram_addr;    
    uint16_t col_num;
    uint16_t row_num;
    uint16_t sram_stride;//row num=0是随便配置
    uint32_t dram_stride;//同上
    uint8_t  precision;    // 2-bit: 数据精度，01是int8，int32也是int8，所以全部配1
    uint8_t  input_type;   // 2-bit: 输入类型，00=IFM, 01=WEIGHT（IFM还是weight不区分）都是int8输入 | 10=BIAS int32 到acc
    bool     dest;         // 1-bit: 0=SPM, 1=ACC
    bool     is_bias;      // 1-bit: 0=No, 1=Bias（mvin到acc的bias寄存器时为1，一次32个数据）
    bool     is_quant;     // 1-bit: 是否量化，是则输入int8后量化到int32存到acc
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
    uint32_t dram_stride;
    uint8_t  precision;    // 2-bit: 数据精度：仍然全1
    uint8_t  output_type;  // 2-bit: 输出类型：00代表int8（从spm），01代表int32（从acc）
    bool     source;       // 1-bit: 0=SPM, 1=ACC
    bool     is_quant;     // 1-bit: 是否量化（没有作用，全0）
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
    uint8_t weight_shape_m1;    // 4-bit: kernel size - 1 (e.g., 2 for 3x3)
    uint8_t weight_stride_m1;   // 2-bit: stride - 1
    uint8_t weight_dilation_m1; // 5-bit: dilation - 1
    bool    is_group_conv;      // 1-bit: 是否分组卷积

    // Compute
    uint8_t int_type;           // 2-bit: 0=int8, 1=int16, ...
    uint8_t op_type;            // 2-bit: 0=GEMM, 1=Conv, 2=GEMV
    bool    dataflow_mode;      // 1-bit: 0=im2col & OS, 1=OS only
    bool    accout_dest;        // 1-bit: 0=SPM, 1=ACC

    // Zeropoints
    uint16_t input_a_zeropoint;
    uint16_t input_b_zeropoint;

    // Input A (IFM)
    uint32_t input_a_addr;
    uint16_t input_a_col_num_m1;
    uint8_t input_a_row_num_m1;
    uint16_t input_a_stride;

    // Input B (Weights)
    uint32_t input_b_addr;
    uint8_t input_b_col_num_m1;
    uint16_t input_b_row_num_m1;
    uint16_t input_b_stride;

    // Accumulator
    uint8_t biaspsum_width;
    uint8_t biaspsum_height;
    uint32_t biaspsum_addr;
    uint16_t biaspsum_stride;

    // Output
    uint32_t output_addr;
    uint16_t output_stride;
    bool    is_accumulate;      // 1-bit: 0=No accumulate, 1=Accumulate with previous psum
    bool    relu_enable;        // 1-bit: 0=Disabled, 1=Enabled
    uint8_t relu_type;          // 3-bit: 0=relu, 1=relu6, 2=leaky(0.1), 3=leaky(0.2), 4=leaky(0.01)
    bool    is_bias;            // 1-bit: 0=accumulate psum, 1=accumulate bias

    // Quantization
    uint32_t output_zeropoint;
    uint16_t quant_scale;
    uint16_t quant_scaleshift;
};

struct GemmConfig {
    // config_compute
    bool     dataflow;            // 1-bit: 0=im2col & OS, 1=OS only，只支持os，填1（卷积时填0）
    uint8_t  int_type;            // 2-bit: 0=int8, 1=int16, ...全部填00，代表int8
    uint8_t  optype;              // 2-bit: 0=GEMM, 1=Conv, （2=GEMV）
    bool     accout_dest;         // 1-bit: 0=SPM（量化到spm 变成int8）, 1=ACC（int32存回acc）
    uint16_t input_a_zeropoint;//对称量化全0
    uint16_t input_b_zeropoint;

    uint32_t output_zeropoint;    // ACC to SPM zeropoint
    uint16_t output_scale;        // ACC to SPM scale，只需要配置out的scale，=scale_a*scaleb/scale_out
    uint16_t output_scaleshift;   // ACC to SPM scaleshift
    
    // config_accumulate
    uint32_t biaspsum_addr;         //acc中的部分和地址，如果是加bias请使用bias寄存器
    uint16_t biaspsum_stride;       
    uint8_t  biaspsum_width;        //不累加也要配置，也就是输出矩阵的大小，不需要减1
    uint8_t  biaspsum_height;       //不累加也要配置

    uint32_t output_addr;         // Output matrix address (To ACC/SPM)，先不要原位写
    uint16_t output_stride;       // Output matrix stride  (To ACC/SPM)
    bool     isaccu;              // 1-bit: 0=No accumulate, 1=Accumulate with previous psum，
    bool     relu;                // 1-bit: 0=Disabled, 1=Enabled
    uint8_t  relu_type;           // 3-bit: 0=relu, 1=relu6, 2=leaky(0.1), 3=leaky(0.2), 4=leaky(0.01)
    bool     is_bias;             // 1-bit: 0=accumulate psum, 1=accumulate bias，bias现在都在acc中的bias寄存器里，如果这位是1则加寄存器中的值

    // compute_sa
    uint32_t input_a_addr;
    uint16_t input_a_col_num;
    uint8_t  input_a_row_num;
    uint16_t input_a_stride;

    uint32_t input_b_addr;
    uint8_t  input_b_col_num;
    uint16_t input_b_row_num;
    uint16_t input_b_stride;
};

struct MataddConfig {
    // Input/Output addresses in ACC/SPM
    uint32_t input_a_addr;     // Matrix A address in ACC
    uint32_t input_b_addr;     // Matrix B address in ACC
    uint32_t output_addr;      // Output address in SPM
    
    // Dimensions
    uint8_t  col_num;          // Column number (actual, not -1)
    uint8_t  row_num;          // Row number (actual, not -1)
    
    // Output Quantization (int32 -> int8)
    uint32_t output_zeropoint;
    uint16_t output_scale;
    uint16_t output_scaleshift;
};

struct TransposeConfig {
    // Transpose: 通过 SFU 实现矩阵转置
    uint32_t input_sram_addr;   // 输入矩阵在 SPM 中的地址
    uint32_t output_sram_addr;  // 输出矩阵在 SPM 中的地址
    uint16_t col_num;           // 列数 (Width - 1)
    uint16_t row_num;           // 行数 (Height - 1)
};

struct ResampleConfig {
    // Resample: 2x 下采样/上采样操作 (通过 SFU 实现)
    uint8_t  resample_type;     // 0=downsample, 1=upsample, 2=pooling
    uint8_t  resample_op;       // 0=max/nearest, 1=avg/bilinear
    uint32_t input_sram_addr;   // 输入数据在 SPM 中的地址
    uint32_t output_sram_addr;  // 输出数据在 SPM 中的地址
    uint16_t input_col_num;     // 输入列数 (Width - 1)
    uint16_t input_row_num;     // 输入行数 (Height - 1)
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

    // 复位：软件控制硬件复位 + 清除影子寄存器
    void reset(); // <--- [新增]

    // 内存管理基础接口
    void* get_memory_base(); 
    uint32_t get_memory_size();

    // 执行指令接口
    void run_mvin(const MvinConfig& cfg);
    void run_mvout(const MvoutConfig& cfg);
    void run_sfu(const SfuConfig& cfg);
    void run_conv(const ConvConfig& cfg);
    void run_gemm(const GemmConfig& cfg);
    void run_matadd(const MataddConfig& cfg);
    void run_transpose(const TransposeConfig& cfg);
    void run_resample(const ResampleConfig& cfg);

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

    // 混合轮询+中断等待
    void wait_irq();
    bool check_irq_pending();  // 检查中断是否挂起，返回挂起的中断位
    void ack_irq();  // 清除/应答全部中断位
    void dump_irq_regs(); // 轮询超时后打印中断相关寄存器
};

// ==========================================
// C Interface (API)
// ==========================================

extern "C" {
    // Lifecycle
    int npu_init();
    void npu_destroy();
    void npu_reset(); // <--- [新增]
    
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
        uint32_t dram_stride,
        uint8_t  precision,    // 2-bit
        uint8_t  input_type,   // 2-bit
        bool     dest,         // 1-bit: 0=SPM, 1=ACC
        bool     is_bias,      // 1-bit
        bool     is_quant,     // 1-bit
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
        uint32_t dram_stride,
        uint8_t  precision,    // 2-bit
        uint8_t  output_type,  // 2-bit
        bool     source,       // 1-bit: 0=SPM, 1=ACC
        bool     is_quant,     // 1-bit
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
        uint8_t pad_top,           // 2-bit
        uint8_t pad_bottom,        // 2-bit
        uint8_t pad_left,          // 2-bit
        uint8_t pad_right,         // 2-bit
        uint8_t pad_mode,          // 2-bit
        uint8_t weight_shape_m1,   // 4-bit
        uint8_t weight_stride_m1,  // 2-bit
        uint8_t weight_dilation_m1,// 5-bit
        bool    is_group_conv,     // 1-bit
        uint8_t int_type,          // 2-bit
        uint8_t op_type,           // 2-bit
        bool    dataflow_mode,     // 1-bit
        bool    accout_dest,       // 1-bit
        uint16_t input_a_zeropoint,
        uint16_t input_b_zeropoint,
        uint32_t input_a_addr,
        uint16_t input_a_col_num_m1,
        uint8_t input_a_row_num_m1,
        uint16_t input_a_stride,
        uint32_t input_b_addr,
        uint8_t input_b_col_num_m1,
        uint16_t input_b_row_num_m1,
        uint16_t input_b_stride,
        uint8_t biaspsum_width,
        uint8_t biaspsum_height,
        uint32_t biaspsum_addr,
        uint16_t biaspsum_stride,
        uint32_t output_addr,
        uint16_t output_stride,
        bool    is_accumulate,     // 1-bit
        bool    relu_enable,       // 1-bit
        uint8_t relu_type,         // 3-bit
        bool    is_bias,           // 1-bit
        uint32_t output_zeropoint,
        uint16_t quant_scale,
        uint16_t quant_scaleshift
    );

    void npu_gemm_run(
        bool     dataflow,         // 1-bit: 0=im2col & OS, 1=OS only
        uint8_t  int_type,         // 2-bit
        uint8_t  optype,           // 2-bit
        bool     accout_dest,      // 1-bit: 0=SPM, 1=ACC
        uint16_t input_a_zeropoint,
        uint16_t input_b_zeropoint,
        uint32_t output_zeropoint,
        uint16_t output_scale,
        uint16_t output_scaleshift,
        uint32_t biaspsum_addr,
        uint16_t biaspsum_stride,
        uint8_t  biaspsum_width,
        uint8_t  biaspsum_height,
        uint32_t output_addr,
        uint16_t output_stride,
        bool     isaccu,           // 1-bit
        bool     relu,             // 1-bit
        uint8_t  relu_type,        // 3-bit
        bool     is_bias,          // 1-bit
        uint32_t input_a_addr,
        uint16_t input_a_col_num,
        uint8_t  input_a_row_num,
        uint16_t input_a_stride,
        uint32_t input_b_addr,
        uint8_t  input_b_col_num,
        uint16_t input_b_row_num,
        uint16_t input_b_stride
    );

    // MATADD Operations
    void npu_matadd_run(
        uint32_t input_a_addr,
        uint32_t input_b_addr,
        uint32_t output_addr,
        uint8_t  col_num,
        uint8_t  row_num,
        uint32_t output_zeropoint,
        uint16_t output_scale,
        uint16_t output_scaleshift
    );

    // Transpose Operation (via SFU)
    void npu_transpose_run(
        uint32_t input_sram_addr,   // 输入矩阵在 SPM 中的地址
        uint32_t output_sram_addr,  // 输出矩阵在 SPM 中的地址
        uint16_t col_num,           // 列数 (Width - 1)
        uint16_t row_num            // 行数 (Height - 1)
    );

    // Resample Operation (via SFU)
    // resample_type: 0=downsample, 1=upsample, 2=pooling
    // resample_op: 0=max/nearest, 1=avg/bilinear
    void npu_resample_run(
        uint8_t  resample_type,     // 采样类型
        uint8_t  resample_op,       // 采样操作
        uint32_t input_sram_addr,   // 输入数据在 SPM 中的地址
        uint32_t output_sram_addr,  // 输出数据在 SPM 中的地址
        uint16_t input_col_num,     // 输入列数 (Width - 1)
        uint16_t input_row_num      // 输入行数 (Height - 1)
    );

    // Test Interface
    void npu_dma_mvin_test(
        void* host_ptr,
        uint32_t sram_addr,
        uint16_t col_num,
        uint16_t row_num,
        uint16_t sram_stride,
        uint32_t dram_stride,
        uint8_t  precision,    // 2-bit
        uint8_t  input_type,   // 2-bit
        bool     dest,         // 1-bit: 0=SPM, 1=ACC
        bool     is_bias,      // 1-bit
        bool     is_quant,     // 1-bit
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
        uint32_t dram_stride,
        uint8_t  precision,    // 2-bit
        uint8_t  output_type,  // 2-bit
        bool     source,       // 1-bit: 0=SPM, 1=ACC
        bool     is_quant,     // 1-bit
        uint32_t quant_zero,
        uint16_t quant_scale,
        uint16_t quant_shift
    );
}

#endif // NPU_RUNTIME_H