# 当前 NPU 硬件规格（编译器与运行时合约）

> **状态：已验证默认硬件的正式接口合约，更新于 2026-08-07。** 本文件以 `rtl/top/npu_top.sv`、`rtl/ctrl/inst_ctrl.sv`、`doc/rdl/npu_inst_ctrl_regs.rdl` 及对应 RTL 回归为准。本文描述的默认配置可由编译器和运行时直接使用；历史 `sw/runtime/npu_regs.h` 的 pre-descriptor ABI 不适用于本硬件。

## 1. 硬件概览

| 项目 | 默认硬件配置 |
| --- | --- |
| 控制接口 | 32-bit 地址、64-bit 数据 AXI-Lite；寄存器偏移单位为 byte，所有 descriptor 均 8-byte 对齐。 |
| 外部存储接口 | 32-bit AXI 地址、256-bit 数据（32 byte/beat）；最大 burst 为 16 beat，读、写各最多 16 个 outstanding 事务。 |
| 标识 | `MAGIC=0x4e505547` (`NPUG`)，ABI version `1`，mode `1`（PREFILL），caps `0x00000001`。 |
| SA | `32 x 32` systolic array，signed INT8 x signed INT8，INT32 累加；单 K block 最大 `4096`。 |
| A/W 本地存储 | A0/A1、W0/W1 各 `2048 x 32 B = 64 KiB`。 |
| O 本地存储 | O0/O1 各 `2048 x 32 B = 64 KiB`；一个 word 可视为 32 个 INT8、16 个 BF16/FP16 或 8 个 INT32。 |
| metadata | normal metadata 与 QK metadata 各 `512 x 32 B = 16 KiB`。 |
| 可提交命令 | `MVIN_A`、`MVIN_W`、`AUX_MVIN`、`SA_COMPUTE`、`MVOUT`、`GEMV`、`VPU`，opcode 分别为 `0..6`。 |
| 目标 | ASIC 时钟目标为 `950 MHz`。实际板端频率以集成工程的时钟约束和实现报告为准。 |

### 1.1 单位、容量和地址

- A/W/O 和 metadata 的本地地址均为 **32-byte word index**；DRAM 地址、offset 与 stride 均为 **byte**。
- 本 ABI 的 DRAM 地址是 descriptor 中的 `dram_base[31:0]`：它是 AXI 地址空间内的 **32-bit byte address**，不是 32-byte word address，也不支持在该字段中编码 64-bit 物理地址。DMA AXI 数据宽度为 256 bit（32 byte/beat）；所有 base 及非零 DRAM row stride 必须满足 `addr[4:0]==0`，否则命令以 alignment error 结束。
- 除字段名含 `_m1` 外，`row_count`、`col_count`、`row_bytes`、`M/N/K` 和 SA 的 `d0..d3` 都是实际计数，`0` 为非法尺寸。
- 默认硬件的有效本地 word 地址范围为 A/W/O 的 `0..2047`、metadata 的 `0..511`。软件必须保证从 base 到最后一个被访问 word 完全位于此范围内。
- DRAM base 与非零 DRAM stride 必须 32-byte 对齐。最后一行或最后一个 word 可以不是 32 byte 整数倍，DMA 使用 keep/strobe 屏蔽尾部无效字节；软件仍须为读 DMA 分配向上取整到 32 byte 的可读空间。

`DESC0[63:32]` 不是 DRAM 地址的高半部分。它按命令分别表示 `dram_row_stride_bytes`（MVIN_A、AUX 的 O 搬运、MVOUT）、`dram_span_bytes`（MVIN_W，仅用于范围检查）或 `offset_bytes`（AUX metadata）。因此软件不得把超过 4 GiB 的地址拆分写入 DESC0 的上下半字。

## 2. 命令提交、状态和中断

1. 先写各命令的 `DESC0/DESC1/DESC2`，最后以一次 64-bit AXI-Lite 写向含 `start` 的 descriptor 置 W1S `start=1`。启动时硬件锁存该命令的 descriptor。
2. 写 start 返回 `OKAY` 表示寄存器写已接受；在复用数据或 bank 前，必须轮询对应 `STATUS.done/error` 或等待 IRQ。
3. 所有 `*_STATUS` 使用统一格式：`busy[0]`、W1C `done[1]`、W1C `error[2]`、`error_code[7:4]`、`accepted_count[63:32]`。accepted count 是接受次数，不是完成次数。
4. `IAR @0x100` 和 `ISR @0x118` 均以 W1C 清中断状态；`MER[0]` 是总使能，`IER[7:0]` 是掩码，`IPR=ISR & IER`，`irq_o = OR(ISR & IER)`。bit 0..4、5、6 分别为 A/W/AUX/SA/MVOUT、GEMV、VPU 完成，bit 7 为任意错误。
5. `GLOBAL_CLEAR @0x128[0]` 或 `GENERIC_CONTROL @0xf20[0]` 清除本地状态、pending 命令、IRQ 和后端状态。它不撤销已经发出的 AXI 事务；执行 clear 后，软件应重新初始化依赖的本地数据。

错误码：`0` none，`1` illegal shape，`2` start while busy，`3` resource conflict，`4` bank conflict，`5` bank not valid，`6` shape mismatch，`7` alignment，`8` unsupported mode，`10` internal assert/FIFO，`11` illegal flags。

### 2.1 通用寄存器

| 偏移 | 名称 | 含义 |
| --- | --- | --- |
| `0x100/0x108/0x110/0x118/0x120` | IAR/MER/IER/ISR/IPR | 中断确认、总使能、掩码、状态和 pending。 |
| `0x128` | GLOBAL_CLEAR | W1C 全局清除。 |
| `0x130` | PROFILE_CTRL | bit0 clear、bit1 enable；只有 elaboration 开启 `ENABLE_PROFILE` 时有效。 |
| `0x138..0x230` | PROFILE_* | global、各 API busy、任意/多路 busy 以及 4 路 AXI 的 beat/stall 64-bit 计数器。 |
| `0xf00/0xf08/0xf10/0xf18` | GENERIC_MAGIC/VERSION/MODE/CAPS | 只读硬件身份寄存器。 |
| `0xf20/0xf28/0xf30` | GENERIC_CONTROL/STATUS/ERROR | 控制、聚合 idle/busy/error 状态和 API error bitmap。 |

## 3. DMA 命令 ABI

### 3.1 MVIN_A、MVIN_W 和 AUX_MVIN

| 命令 | 寄存器 | 字段 |
| --- | --- | --- |
| MVIN_A | `0x000 DESC0` | `dram_base[31:0]`，`dram_row_stride_bytes[63:32]`。 |
|  | `0x008 DESC1` | `row_count[15:0]`，`col_count[31:16]`，`u8_minus_128[32]`，`start[33]`，`a_bank[34]`。 |
| MVIN_W | `0x020 DESC0` | `dram_base[31:0]`，`dram_span_bytes[63:32]`。该高 32 位仅参与 descriptor 检查；W 传输为连续布局，不提供 2D DRAM stride。 |
|  | `0x028 DESC1` | `row_count[15:0]`，`col_count[31:16]`，`w_bank[32]`，`start[33]`。 |
| AUX_MVIN | `0x040 DESC0` | `dram_base[31:0]`，`offset_bytes[63:32]`。 |
|  | `0x048 DESC1` | `value[31:0]`，`aux_type/o_bank[33:32]`，`start[34]`，`dest_o_bank[35]`，`metadata_bank[36]`。 |

MVIN_A/MVIN_W 的占用为 `row_count * ceil(col_count / 32)` 个本地 word。A 的 `u8_minus_128=1` 将每个有效输入 byte 解释为 unsigned 并减 128，再按 INT8 写入；为 0 时原样搬运。

`AUX_MVIN.dest_o_bank=0` 时搬运 metadata：`value` 是实际 byte count，`offset_bytes` 是 metadata byte offset，`metadata_bank` 选择 ping-pong bank。`dest_o_bank=1` 时搬运单物理 Res bank：`value[15:0]` 是 row count、`value[31:16]` 是 row bytes，`offset_bytes` 是 DRAM row stride；`aux_type/o_bank[33:32]` 为保留位，必须为 0。

#### 3.1.1 DMA 计数、stride 与旧接口迁移

当前 descriptor ABI 不使用旧 runtime 的通用 `col_num/row_num/sram_stride/dram_stride` 寄存器。对应关系如下；它们均为实际值，不采用旧接口的“值减一”编码。

| 旧字段 | 当前 descriptor 字段/硬件行为 | 位宽、单位和约束 |
| --- | --- | --- |
| `col_num` | MVIN_A/MVIN_W 的 `col_count`，MVOUT 的 `row_bytes`；即一行实际传输字节数。 | 16 bit，byte；`0` 非法。可非 32 的倍数，最后 AXI beat 以 keep/strobe 屏蔽。 |
| `row_num` | MVIN_A/MVIN_W/MVOUT 的 `row_count`。 | 16 bit，实际行数；`0` 非法。 |
| `dram_stride` | `dram_row_stride_bytes=DESC0[63:32]`（MVIN_A/MVOUT，以及 AUX 搬运 O）。下一行地址为 `dram_base + row_index * stride`。MVIN_W 固定连续布局，硬件将其 stride 置 0。 | 32 bit，byte；非零值必须 32-byte 对齐。AUX metadata 是单行搬运，`offset_bytes` 是本地 metadata 偏移，不是 DRAM stride。 |
| `sram_stride` | 无 descriptor 字段，也不支持任意本地 2D stride。A/W DMA 从所选 bank 的 word 0 开始，按 `ceil(col_count/32)` 个 32-byte word 连续排布各行；MVOUT 从所选 O bank 的 word 0 连续读取各行。 | 硬件推导，单位为 32-byte word。需要任意 local base/stride 时，须由软件重排、使用相应 compute base，或拆为多个命令。 |

旧 `input_type/output_type` 只属于 `sw/runtime/npu_runtime.*` 的 pre-descriptor API，当前 `inst_ctrl` 寄存器文件没有这两个字段，软件不得把它们写入本 ABI。旧 `input_type=IFM` 对应 `MVIN_A`，旧 `input_type=WEIGHT` 对应 `MVIN_W`；旧 `input_type=BIAS` 没有等价 DMA type，应通过 AUX metadata 路径加载 bias/scale。旧 `output_type=INT8/INT32` 也没有 MVOUT selector：当前 MVOUT 总是从 `o_bank` 导出，元素格式由产生该 O 数据的 SA `output_format` 及软件给出的 `row_bytes` 共同决定。


| 寄存器 | 字段 |
| --- | --- |
### 3.2 MVOUT

| `MVOUT_DESC0 @0x080` | `dram_base[31:0]`，`dram_row_stride_bytes[63:32]`。 |
| `MVOUT_DESC1 @0x088` | `row_count[15:0]`，`row_bytes[31:16]`，`metadata_base_byte[47:32]`，`qk_mode[48]`，`metadata_bank[49]`，`start[50]`，`o_bank[51]`（`0=O0`、`1=O1`），`reserved[52]`（必须为 0），`o_ischange[53]`，`qk_mask_enable[54]`。 |
| `MVOUT_STATUS @0x090` | 通用状态格式。 |

`o_bank[51]` 选择 `O0` 或 `O1`；`MVOUT_DESC1[52]` 是保留位，置 1 会以 illegal-flags error 拒绝命令。`qk_mode=0` 为 O bank raw data 写回；`qk_mode=1` 启用 attention QK logP correction，并使用指定 metadata bank/base 与可选 causal mask。每行传输 `ceil(row_bytes/32)` 个 AXI beat，最后一拍按 strobe 写回。

## 4. SA_COMPUTE ABI

| 寄存器 | 字段 |
| --- | --- |
| `DESC0 @0x060` | `d0/d1/d2/d3`，每个 16-bit 实际计数。GEMM/QK/PV 为 `M/N/K/0`；Conv 为 `IFM_H/IFM_W/Cin/Cout`。 |
| `DESC1 @0x068` | `a_base/w_base/o_base/acc_base/resadd_base`，每个 12-bit local-word 地址。 |
| `DESC2 @0x070` | `op_type[1:0]`、标志、输出格式、Conv 参数、metadata 地址、padding、bank 和 `start[45]`。 |
| `STATUS @0x078` | 通用状态格式。 |

`op_type`：`0=GEMM`、`1=Conv`、`2=attention QK`、`3=attention PV`。SA 自动将宏矩阵拆成最多 `32 x 32` 的 M/N tile，并处理尾 tile；`K` 不得超过 `4096`。

`DESC2` 的详细编码如下：

| 位 | 含义 |
| --- | --- |
| `[2]` / `[3]` / `[4]` | `accu_en` / `bias_en` / `resadd_en`。GEMM/Conv 中 `accu_en && bias_en` 非法。QK 中 `bias_en` 表示 causal mask，`accu_en=resadd_en=0`；PV 中 `accu_en=bias_en=0`，`resadd_en` 用于最终归一化结果。 |
| `[6:5]` | `output_format`：`0=raw INT32`、`1=tensor Q8.24 -> INT8`、`2=per-channel INT8`、`3=FP32 conversion`。 |
| `[10:7]` / `[12:11]` / `[17:13]` | Conv 的 `kernel_shape_m1`、`stride_m1`、`dilation_m1`；实际值均为 field+1。 |
| `[26:18]` / `[35:27]` | `bias_addr` 与 `scale_addr`，均为 9-bit metadata word index。mode1/mode3 的 scale 是 Q8.24 immediate 编码 `scale_addr << 4`；mode2 为逐通道 scale metadata base。QK 中两字段分别复用为 metadata base 与 gamma base。 |
| `[43:36]` | `pad_l/pad_r/pad_t/pad_b`，各 2 bit，填充值固定为 INT8 0。 |
| `[44]` / `[45]` | `irq_en` / W1S `start`。完成 IRQ 由 SA 完成源统一产生。 |
| `[53:46]` | `A[46]`、`W[47]`、`O[48]`、`ACC[50]` 为 bank 选择（均为 0/1）；`[49]`、`[51]` 为保留位且必须为 0。`RESADD[53:52]` 对应单物理 Res bank，必须为 0（即使 `resadd_en=0` 也必须清零）。 |
| `[60:54]` | bank lifecycle 字段：`a/w/o/acc/resadd/meta_ischange` 分别在 `[54]..[59]`，`meta_bank` 在 `[60]`。scale granularity 不再由这些位编码，而由 `output_format` 唯一决定：mode0 无 scale，mode1/mode3 为 per-tensor，mode2 为 per-channel。 |
| `[61]` | `relu_en`。 |

### 4.1 GEMM 形状、布局和尾块

对 `op_type=0`（GEMM），`DESC0.d0/d1/d2` 分别为实际 `M/N/K`，每个均为 16 bit；`d3` 保留并应写 0。计算为 `C[M,N] = A[M,K] × B[K,N]`，所有 A/W 元素为 signed INT8，累加器为 signed INT32。`M=0`、`N=0`、`K=0` 非法，单命令 `K <= 4096`；更大的 K 必须由软件分割为多条命令，并用 `accu_en` 管理 partial sum。

A 与 W 都以 32 个 INT8/word 存放，但 W 的 word 行是输出列而非逻辑矩阵行。令 `kw=ceil(K/32)`，对于从 `a_base/w_base` 开始的 bank word 地址：

```text
A[m][k] = A_bank[a_base + m*kw + floor(k/32)].byte[k mod 32]
B[k][n] = W_bank[w_base + n*kw + floor(k/32)].byte[k mod 32]
```

亦即 A 是逻辑 `[M,K]` row-major；W 必须预转置为 `[N,K]` row-major（按每个输出列连续保存 K chunk），不能把逻辑 `[K,N]` row-major 数据直接送入 W bank。输出 C 是 row-major；地址由 scheduler 自动按 `N` 和输出格式生成，`o_base/acc_base/resadd_base` 均为 32-byte word index。

M、N 由 scheduler 分成 `ceil(M/32) × ceil(N/32)` 个 tile，最后 tile 用 lane mask 屏蔽越界的行/列；软件不需要把 M/N 向上补齐。K 不会被 scheduler 自动分成多条 descriptor：K 尾块使用实际 `K` 控制最后一个 32-byte A/W word 的有效 lane，DMA 尾 beat 的无效字节不写入；为避免复用 bank 中的陈旧字节影响调试或非标准加载路径，软件应将最后 K word 的未使用 byte 预置为 0。

### 4.2 输出写回与量化

`SA_COMPUTE_DESC2[6:5]` 的可提交编码及 O-bank 格式如下：

| 编码 | 硬件行为 | O-bank 布局 |
| --- | --- | --- |
| `0` | raw INT32；可选 bias/accumulate/residual，未做量化。 | 每 word 8 个 little-endian INT32；行 stride `ceil(N/8)` words。 |
| `1` | per-tensor Q8.24 乘法、对称 round-to-nearest（远离零）并饱和为 INT8。scale 为 `scale_addr << 4`。 | 每 word 32 个 INT8；行 stride `ceil(N/32)` words。 |
| `2` | **descriptor 专用别名**：顶层将此值映射为 INT8 输出，并选择 per-channel Q8.24 scale；`scale_addr` 是 metadata word base，每个 32-byte word 含 8 个连续通道的 32-bit scale。它不是可提交的 BF16 格式。 | 每 word 32 个 INT8；行 stride `ceil(N/32)` words。 |
| `3` | 将 `INT32 × (scale_addr << 4)` 的 Q8.24 结果转换为 FP32。 | 每 word 8 个 IEEE-754 FP32；行 stride `ceil(N/8)` words。 |

`relu_en=1` 在量化/转换前将负 INT32 截为 0。内部枚举仍保留 BF16 值，但当前顶层对 descriptor 编码 `2` 的显式重映射使 BF16 不能由 SA_COMPUTE ABI 直接请求。MVOUT 不再做普通 GEMM 量化或类型转换：`qk_mode=0` 时它逐字节导出已有 O-bank 内容，软件必须用与上述格式匹配的 `row_bytes`（例如 INT8 为 `N`、INT32/FP32 为 `4*N`）并设置所需 DRAM row stride；只有 `qk_mode=1` 会进入 attention QK 的专用 logP correction 路径。


GEMV 是独立命令，复用 A/W/O/metadata bank，与 SA_COMPUTE 互斥。默认路径支持 W4A16：激活为 FP16，权重为 signed W4，weight scale 为 FP16，输出为 FP16；同时提供 W8、KV/PV、激活 scale、cache 与预载累加控制。

| 寄存器 | 字段 |
| --- | --- |
| `GEMV_DESC0 @0x098` | `m[15:0]`，`act_group_stride_bytes[31:16]`，`k[47:32]`，`act_scale_base[59:48]`。 |
| `GEMV_DESC1 @0x0a0` | `a_base[11:0]`，`w_base[23:12]`，`o_base[35:24]`，`act_scale2_base[47:36]`，`cache_cell_idx[63:48]`。 |
| `GEMV_DESC2 @0x0a8` | `metadata_base[8:0]`，metadata/A/W/O bank `[13:9]`，`mode[15:14]`，`group_count_m1[17:16]`，`kv_col_scale_en[18]`，`pv_prob_q24_en[19]`，`unit_weight_scale[20]`，`act_scale_en[21]`，`act_scale2_en[22]`，`preload_acc_en[23]`，preload `[46:24]`，`start[63]`。 |
| `GEMV_STATUS @0x0b0` | 通用状态格式。 |

`m`、`k` 是实际计数；`group_count_m1` 是 group count 减一。`mode=0` 选择 W4，`mode=1` 选择 W8；`kv_col_scale_en` 启用 KV 列 scale，`pv_prob_q24_en` 选择 Q24 概率输入，`unit_weight_scale` 固定 weight scale 为 1，两个 `act_scale_en` 分别启用一、二级 FP16 激活 scale。`cache_cell_idx` 不是内存地址：低 15 bit 是激活列缓存 key，bit15 是一次性 invalidate tag。preload 字段的格式为 `id[25:24]`、`row[30:26]`、`data[46:31]`。

## 6. VPU ABI

| 寄存器 | 字段 |
| --- | --- |
| `VPU_DESC0 @0x0c0` | `opcode[5:0]`，`funct[9:6]`，`sew[15:13]`，`vd[20:16]`，`vs1[25:21]`，`vs2[30:26]`，并在 `[47:32]` 编码 `cols_m1`、`[63:48]` 编码 `rows_m1`。 |
| `VPU_DESC1 @0x0c8` | `scalar[31:0]`，`out_inv_scale_q8_24[63:32]`。 |
| `VPU_DESC2 @0x0d0` | `src_o_bank[0]`，`reserved[1]`，`dst_o_bank[2]`，`reserved[3]`，`src_precision[5:4]`，`dst_precision[7:6]`，`dst_a_bank[8]`，`dst_a_bank_index[9]`，`dst_int8_high_half[10]`，`src_addr[31:16]`，`dst_addr[47:32]`，W1S `start[63]`。 |
| `VPU_STATUS @0x0e0` | 通用状态格式。 |

VPU 向量长度为 256 bit，含 32 个 8-bit、16 个 16-bit 或 8 个 32-bit lane；VREG 数量为 32。`sew=0/1/2` 对应 8/16/32 bit。precision 编码为 `0=INT8`、`1=BF16`、`2=FP32`、`3=FP16`。

`OPC_SPECIAL=0x10` 的 funct 定义为 `0=RMSNorm`、`1=LayerNorm`、`2=Softmax`、`3=GELU`、`4=Transpose`、`5=MaxPool`、`6=Sigmoid`。RMSNorm/LayerNorm/Softmax/GELU/Sigmoid 的矩阵形状来自 `DESC0`，即 `rows=rows_m1+1`、`cols=cols_m1+1`；其输入必须为 BF16，输出可为 BF16、FP16 或 INT8。Transpose 和 MaxPool 的形状来自 `DESC1.scalar`，其中 `rows_m1=scalar[31:16]`、`cols_m1=scalar[15:0]`，且输入、输出都必须为 INT8。RMSNorm/LayerNorm/Softmax 使用 FP32 跨 lane 累加；INT8 写回为 `saturate(round(y * out_inv_scale_q8_24), -128, 127)`，`0x01000000` 表示 1.0。`dst_a_bank=1` 时结果写入 A0/A1；INT8 的 `dst_int8_high_half=1` 将有效半 word 写至 256-bit word 的高 128 bit。

## 7. 数据布局和调度约束

- SA 的 A/W 以 32 个 INT8 为一个 32-byte word；O 的 raw INT32 为每 word 8 lane。GEMM、Conv、QK、PV 使用 K-inner 数据组织，编译器负责将 macro tile 切分为本地 bank 容量内的 command。
- Conv 输入为 NCHWc32 组织；kernel、stride、dilation 均通过 `*_m1` 字段编码，padding 使用 0 填充。
- mode1/mode3 的 Q8.24 scale 使用 `scale_addr << 4`；mode2 的每个 metadata word 含 8 个 32-bit Q8.24 scale，对应连续 8 个输出通道。
- QK 结果通过 QK MVOUT 进行 logP correction；PV 输入使用 QK 概率编码并在最终输出中完成归一化。
- 同一 bank 在未完成命令期间不得作为冲突命令的输入或输出。A/W、O 与 Acc 均为双 bank ping-pong；Res 是单物理 bank，仅由 AUX_MVIN 写入并由 ResAdd 读取。只有 O0/O1 可由 MVOUT 导出；软件必须按完成状态管理其生命周期。

## 8. 软件使用流程

1. 读取 `GENERIC_MAGIC/VERSION/MODE/CAPS`，确认值为 `0x4e505547/1/1/1`。
2. 将输入、权重和 metadata 按本文件的 32-byte word 布局放入可 DMA 的 DRAM 缓冲区，并完成 CPU cache 同步。
3. 使用 MVIN_A、MVIN_W 和 AUX_MVIN 填充所选 bank；等待各自完成。
4. 提交 SA_COMPUTE、GEMV 或 VPU 命令；对有数据相关或 bank 复用的命令等待 `done/error`。
5. 选择保存结果的 O0/O1 之一，以 MVOUT 取回数据，检查完成状态并以 W1C 清除 `done/error` 和中断位。
6. 任一错误都读取 `error_code`，执行 GLOBAL_CLEAR，并重新加载受影响的 bank 后再重试。

## 9. 寄存器索引

```text
0x000/008/010  MVIN_A_DESC0/DESC1/STATUS
0x020/028/030  MVIN_W_DESC0/DESC1/STATUS
0x040/048/050  AUX_MVIN_DESC0/DESC1/STATUS
0x060/068/070/078  SA_COMPUTE_DESC0/DESC1/DESC2/STATUS
0x080/088/090  MVOUT_DESC0/DESC1/STATUS
0x098/0a0/0a8/0b0  GEMV_DESC0/DESC1/DESC2/STATUS
0x0c0/0c8/0d0/0e0  VPU_DESC0/DESC1/DESC2/STATUS
0x100..0x230   interrupt and optional profile registers
0xf00..0xf30   generic identity, control and aggregate status
```

寄存器字段的机器可读定义见 `doc/rdl/npu_inst_ctrl_regs.rdl`；软件头文件应由同一寄存器定义生成或与其保持逐字段一致。
