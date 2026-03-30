# VPU (Vector Processing Unit) Architecture & ISA Specification

> **Project Name** : T_NPU
> **Module Name** : `vpu_top`
> **Version** : v1.4
> **Designer** : Jiakun Shen
> **Last Update** : 2026/01/28

---

## 1. 概述 (Overview)

VPU (Vector Processing Unit) 是 T_NPU 的向量协处理单元，旨在替代原有 SFU 模块，提供灵活的向量算术、逻辑、规约及特殊函数计算能力。

### 1.1 核心特性 (Key Features)

| 特性 | 描述 |
| :--- | :--- |
| **SIMD 架构** | 256-bit VLEN (32× INT8 / 16× INT16 / 16× BF16 / 8× INT32) |
| **混合精度** | 算术运算支持 INT8/INT16/BF16，累加与量化支持 INT32/FP32 |
| **独立存储** | **VRF**: 32 × 256b 向量寄存器堆 (支持 8/16/32b 元素)<br>**ACC**: 1 × 1024b 累加器堆 (32 × 32b) |
| **控制模式** | **MMIO 配置驱动** - 由 `inst_ctrl` 直连控制，无取指开销 |
| **单发射阻塞** | 通过 `busy` 信号串行执行，简化流水线设计 |

### 1.2 支持的计算类型

| 类别 | 操作 |
| :--- | :--- |
| 向量算术 | Add, Sub, Mul, MAC, Shift, Min/Max, Saturate |
| 向量逻辑 | And, Or, Xor |
| 向量规约 | Sum, Max |
| 特殊函数 | Exp2, Log2, Rsqrt, Recip |
| 类型转换 | INT8↔BF16, INT32↔BF16, Quantization |

---

## 2. 系统架构 (System Architecture)

### 2.1 顶层集成框图

```
      System Bus (AXI-Lite)
            │
            ▼
  ┌───────────────────────┐
  │      inst_ctrl        │  MMIO 配置 + 命令生成
  │   (AXI-Lite Slave)    │
  └───────────┬───────────┘
              │ cfg_vpu_* + vpu_req_en
              ▼
  ┌───────────────────────┐
  │       VPU Core        │
  │  ┌─────────────────┐  │
  │  │   Dispatcher    │  │  ← S1: 解码控制信号
  │  ├─────────────────┤  │
  │  │ VRF (32×256b)   │  │  ← S2: 读取源操作数 (VRF/ACC)
  │  │ ACC (1×1024b)   │  │
  │  ├─────────────────┤  │
  │  │ Execution Units │  │  ← S3: 执行 + 写回
  │  │ ALU│MAC│SFU│RED │  │
  │  └─────────────────┘  │
  └───────────┬───────────┘
              │ SPM R/W
              ▼
        ScratchPad Memory
```

### 2.2 指令编码实现方式

VPU 采用 **MMIO 寄存器映射** 方式实现指令分发。

工作流程：
1. Host CPU 通过 AXI-Lite 向 `inst_ctrl` 写入 VPU 配置寄存器。
2. `inst_ctrl` 将配置值以组合逻辑直连方式输出给 VPU。
3. Host 写入启动寄存器触发 `vpu_req_en` 脉冲。
4. VPU 执行并返回 `vpu_comp_done` 完成信号。

### 2.3 流水线结构

采用 **三级阻塞流水线**：

| 阶段 | 名称 | 功能 | 周期数 |
| :--: | :--- | :--- | :---: |
| S1 | Dispatch | 锁存配置，解析操作码，生成执行单元控制 | 1 |
| S2 | VRF Read | 从 VRF/ACC 读取源操作数 | 1 |
| S3 | Execute | ALU/MAC/SFU/Reduce 执行 + 结果写回 | 1-6 |

---

## 3. 接口定义 (Interface Definition)

### 3.1 控制接口 (与 inst_ctrl 握手)

| 信号名 | 方向 | 位宽 | 描述 |
| :--- | :--- | :--- | :--- |
| `vpu_req_en` | Input | 1 | 启动脉冲 (单周期高电平) |
| `vpu_busy` | Output | 1 | 忙状态 (执行期间保持高) |
| `vpu_comp_done` | Output | 1 | 完成脉冲 (触发中断) |

### 3.2 配置接口 (MMIO 寄存器映射)

这些信号由 `inst_ctrl` 内部寄存器直接驱动，构成 VPU 的"指令编码"。

| 信号名 | 位宽 | 描述 | 寄存器偏移建议 |
| :--- | :--- | :--- | :--- |
| `cfg_vpu_opcode` | 6 | 主操作码 | `slv_reg[VPU_BASE+0][5:0]` |
| `cfg_vpu_funct` | 4 | 子功能码 | `slv_reg[VPU_BASE+0][9:6]` |
| `cfg_vpu_sew` | 3 | 元素宽度 (00=8b, 01=16b, 10=32b) | `slv_reg[VPU_BASE+0][15:13]` |
| `cfg_vpu_vd` | 5 | 目标寄存器索引 | `slv_reg[VPU_BASE+0][20:16]` |
| `cfg_vpu_vs1` | 5 | 源寄存器 1 索引 | `slv_reg[VPU_BASE+0][25:21]` |
| `cfg_vpu_vs2` | 5 | 源寄存器 2 索引 | `slv_reg[VPU_BASE+0][30:26]` |
| `cfg_vpu_scalar` | 32 | 标量操作数 / 立即数 / SPM地址[17:0] | `slv_reg[VPU_BASE+2][31:0]` |

### 3.3 SPM 接口

| 信号名 | 方向 | 位宽 | 描述 |
| :--- | :--- | :--- | :--- |
| `vpu_spm_rd_en` | Output | 1 | 读使能 |
| `vpu_spm_rd_addr` | Output | SPM_ADDR_WIDTH | 读地址 |
| `vpu_spm_rd_data` | Input | SPM_DATA_WIDTH | 读数据 (256b) |
| `vpu_spm_wr_en` | Output | 1 | 写使能 |
| `vpu_spm_wr_addr` | Output | SPM_ADDR_WIDTH | 写地址 |
| `vpu_spm_wr_data` | Output | SPM_DATA_WIDTH | 写数据 (256b) |
| `vpu_spm_wr_mask` | Output | PE_WIDTH | 写掩码 (按元素) |

---

## 4. 指令集架构 (ISA)

### 4.1 汇编格式与约定

VPU 指令采用精简助记符，操作数位宽由 `SEW` (Selected Element Width) 配置决定，或由指令类型隐含。

```
mnemonic[.opt]  vd, vs2, vs1      # 双源操作
mnemonic[.opt]  vd, vs2, imm      # 立即数操作
mnemonic[.opt]  vd, (rs1)         # 访存操作 (rs1=SPM传址)
mnemonic[.opt]  vd, vs2           # 单源操作
```

**寄存器约定**：
- `v0-v31`: VRF 向量寄存器 (256b/寄存器)
- `acc`: 累加器 (1024b, 保存 32×32b 数据)
- `(rs1)`: SPM 访存地址 (由 `scalar[17:0]` 传递)
- `imm`: 立即数 (由 `scalar` 传递)

> **操作位宽约定**：
> - **[ACC]** 指令：始终操作 **32 个 Lane** (对应 ACC 的 32 个 32b 元素)。
> - **非 ACC** 指令：操作 **1 个向量寄存器** (长度为 VLEN，元素个数取决于 SEW)。

> **图例**:
> - `(*)` : 当前版本未实现的指令
> - `[ACC]` : 涉及 ACC 读写的指令

### 4.2 访存指令 (Load/Store)

支持向量与 ACC 的 SPM 存取。地址固定的 Unit-Stride 模式。

| Opcode | Funct | 助记符 | 汇编格式 | 操作 (Math) | 精度流 | SEW | 属性 |
| :---: | :---: | :--: | :--: | :--: | :--: | :--: | :--: |
| 0x00 | 0x0 | `vle8.v` | `vle8.v vd, (rs1)` | `vd[*] = Mem[rs1]` | 8b → 8b | 0 | - |
| 0x00 | 0x0 | `vle16.v` | `vle16.v vd, (rs1)` | `vd[*] = Mem[rs1]` | 16b → 16b | 1 | - |
| 0x00 | 0x0 | `vle32.v` | `vle32.v vd, (rs1)` | `vd[*] = Mem[rs1]` | 32b → 32b | 2 | - |
| 0x01 | 0x0 | `vse8.v` | `vse8.v vs3, (rs1)` | `Mem[rs1] = vs3[*]` | 8b → 8b | 0 | - |
| 0x01 | 0x0 | `vse16.v` | `vse16.v vs3, (rs1)` | `Mem[rs1] = vs3[*]` | 16b → 16b | 1 | - |
| 0x01 | 0x0 | `vse32.v` | `vse32.v vs3, (rs1)` | `Mem[rs1] = vs3[*]` | 32b → 32b | 2 | - |

### 4.3 整数算术指令 (Integer ALU)

包括基础算术、逻辑运算、移位与比较。

| Opcode | Funct | 助记符 | 汇编格式 | 功能描述 | 精度流 | SEW | 备注 |
| :---: | :---: | :--: | :--: | :--- | :--: | :--: | :--: |
| 0x02 | 0x0 | `vadd` | `vadd vd, vs2, vs1` | 加法 `vd = vs2 + vs1` | SEW → SEW | 0/1/2 | - |
| 0x02 | 0x1 | `vsub` | `vsub vd, vs2, vs1` | 减法 `vd = vs2 - vs1` | SEW → SEW | 0/1/2 | - |
| 0x02 | 0x2 | `vand` | `vand vd, vs2, vs1` | 按位与 | SEW → SEW | 0/1/2 | - |
| 0x02 | 0x3 | `vor` | `vor vd, vs2, vs1` | 按位或 | SEW → SEW | 0/1/2 | - |
| 0x02 | 0x4 | `vxor` | `vxor vd, vs2, vs1` | 按位异或 | SEW → SEW | 0/1/2 | - |
| 0x03 | 0x0 | `vsll` | `vsll vd, vs2, imm` | 逻辑左移 `vd = vs2 << imm` | SEW → SEW | 0/1/2 | - |
| 0x03 | 0x1 | `vsra` | `vsra vd, vs2, imm` | 算术右移 `vd = vs2 >>> imm` | SEW → SEW | 0/1/2 | - |
| 0x03 | 0x2 | `vsrl` | `vsrl vd, vs2, imm` | 逻辑右移 `vd = vs2 >> imm` | SEW → SEW | 0/1/2 | - |
| 0x03 | 0x3 | `vacc.sra` | `vacc.sra acc, imm` | ACC 算术右移 | 32b → 32b | - | [ACC] |
| 0x03 | 0x4 | `vacc.srl` | `vacc.srl acc, imm` | ACC 逻辑右移 | 32b → 32b | - | [ACC] |
| 0x04 | 0x0 | `vmin` | `vmin vd, vs2, vs1` | 有符号最小值 | SEW → SEW | 0/1/2 | - |
| 0x04 | 0x1 | `vmax` | `vmax vd, vs2, vs1` | 有符号最大值 | SEW → SEW | 0/1/2 | - |
| 0x04 | 0x2 | `vsadd` | `vsadd vd, vs2, vs1` | 饱和加法 | SEW → SEW | 0/1/2 | - |
| 0x04 | 0x3 | `vssub` | `vssub vd, vs2, vs1` | 饱和减法 | SEW → SEW | 0/1/2 | - |
| 0x04 | 0x4 | `vminu` (*) | `vminu vd, vs2, vs1` | 无符号最小值 | SEW → SEW | 0 | - |
| 0x04 | 0x5 | `vmaxu` (*) | `vmaxu vd, vs2, vs1` | 无符号最大值 | SEW → SEW | 0 | - |
| 0x04 | 0x6 | `vsaddu` (*) | `vsaddu vd, vs2, vs1` | 无符号饱和加 | SEW → SEW | 0 | - |
| 0x04 | 0x7 | `vssubu` (*) | `vssubu vd, vs2, vs1` | 无符号饱和减 | SEW → SEW | 0 | - |

### 4.4 整数乘法与累加 (Integer MAC)

支持扩展位宽乘法与 ACC 累加操作。

> **Broadcast**: 汇编使用 `vs1[idx]` 表示广播模式 (由 `scalar[31]`配置)。

| Opcode | Funct | 助记符 | 汇编格式 | 功能 (Math) | 精度流 | SEW | 属性 |
| :---: | :---: | :--: | :--: | :--- | :--: | :--: | :--: |
| 0x05 | 0x0 | `vwmul.s8` | `vwmul.s8 vd, vs2, vs1` | 8b×8b → 16b 乘法（广播） | 8*8 → 16b | 0 | - |
| 0x05 | 0x0 | `vwmul.s16` | `vwmul.s16 vd, vs2, vs1` | 16b×16b → 32b 乘法（广播） | 16*16 → 32b | 1 | - |
| 0x05 | 0x1 | `vwmac.s8` | `vwmac.s8 acc, vs2, vs1` | 8b×8b + 32b → 32b 累加（广播） | 8*8+32 → 32b | 0 | [ACC] |
| 0x05 | 0x1 | `vwmac.s16` | `vwmac.s16 acc, vs2, vs1` | 16b×16b + 32b → 32b 累加（广播） | 16*16+32 → 32b | 1 | [ACC] |
| 0x05 | 0x2 | `vwmac.u8` (*) | `vwmac.u8 acc, vs2, vs1` | 无符号累加 | 8*8+32 → 32b | 0 | [ACC] |
| 0x05 | 0x3 | `vwmac.mix` | `vwmac.mix acc, vs2, vs1` | 8b×16b + 32b → 32b 混合累加（广播） | 8*16+32 → 32b | - | [ACC] |
| 0x05 | 0x4 | `vacc.add` | `vacc.add acc, vs1` | ACC 累加: `acc += sext(vs1)` | SEW → 32b | 0/1 | [ACC] |
| 0x05 | 0x5 | `vacc.sub` | `vacc.sub acc, vs1` | ACC 减法: `acc -= sext(vs1)` | SEW → 32b | 0/1 | [ACC] |

### 4.5 浮点运算 (BF16 & FP32)

| Opcode | Funct | 助记符 | 汇编格式 | 功能描述 | 精度流 | 属性 |
| :---: | :---: | :--: | :--: | :--- | :--: | :--: |
| 0x06 | 0x0 | `vfadd` | `vfadd vd, vs2, vs1` | BF16 加法 | BF16 → BF16 | - |
| 0x06 | 0x1 | `vfsub` | `vfsub vd, vs2, vs1` | BF16 减法 | BF16 → BF16 | - |
| 0x06 | 0x2 | `vfacc.add` | `vfacc.add acc, vs1` | FP32 ACC 累加: `acc += fp32(vs1)` | BF16 → FP32 | [ACC] |
| 0x06 | 0x3 | `vfacc.sub` | `vfacc.sub acc, vs1` | FP32 ACC 减法: `acc -= fp32(vs1)` | BF16 → FP32 | [ACC] |
| 0x07 | 0x0 | `vfmul` | `vfmul vd, vs2, vs1` | BF16 乘法（广播） | BF16 → BF16 | - |
| 0x07 | 0x1 | `vfwmul` | `vfwmul vd, vs2, vs1` | BF16×BF16 → FP32 扩位乘法（广播） | BF16 → FP32 | - |
| 0x07 | 0x2 | `vfwmac` | `vfwmac acc, vs2, vs1` | BF16×BF16 + FP32 → FP32 累加（广播） | BF16 → FP32 | [ACC] |
| 0x07 | 0x3 | `vfmul.mix` | `vfmul.mix vd, vs2, vs1` | INT8×BF16 → BF16 混合乘（广播） | I8*BF16 → BF16 | - |
| 0x07 | 0x4 | `vfwmac.mix` | `vfwmac.mix acc, vs2, vs1` | INT8×BF16 + FP32 → FP32 混合累加（广播） | I8*BF16 → FP32 | [ACC] |
| 0x08 | 0x0 | `vfmax` | `vfmax vd, vs2, vs1` | BF16 最大值 | BF16 → BF16<br>FP32 → FP32 | - |
| 0x08 | 0x1 | `vfmin` | `vfmin vd, vs2, vs1` | BF16 最小值 | BF16 → BF16<br>FP32 → FP32 | - |

### 4.6 特殊函数与规约 (SFU & Reduction)

| Opcode | Funct | 助记符 | 汇编格式 | 描述 | 精度流 | SEW | 备注 |
| :---: | :---: | :--: | :--: | :--- | :--: | :--: | :--: |
| 0x09 | 0x0 | `vexp2` | `vexp2 vd, vs2` | 2^x | BF16 → BF16 | - | BF16 |
| 0x09 | 0x1 | `vlog2` | `vlog2 vd, vs2` | log2(x) | BF16 → BF16 | - | BF16 |
| 0x09 | 0x2 | `vrecip` | `vrecip vd, vs2` | 1/x (Approx) | BF16 → BF16 | - | BF16 |
| 0x09 | 0x3 | `vrsqrt` | `vrsqrt vd, vs2` | 1/sqrt(x) | BF16 → BF16 | - | BF16 |
| 0x0A | 0x0 | `vredsum` | `vredsum vd[0], vs2` | 整数求和规约 | SEW → 32b | 0/1/2 | Int SEW |
| 0x0A | 0x1 | `vredmax` | `vredmax vd[0], vs2` | 整数最大值规约 | SEW → SEW | 0/1/2 | Int SEW |
| 0x0B | 0x0 | `vfredsum` | `vfredsum vd[0], vs2` | 浮点求和规约 | BF16 → FP32 | 1/2 | BF16/FP32 |
| 0x0B | 0x1 | `vfredmax` | `vfredmax vd[0], vs2` | 浮点最大值规约 | BF16 → BF16 | 1/2 | BF16/FP32 |

### 4.7 数据搬运与转换 (Move & Convert)

| Opcode | Funct | 助记符 | 汇编格式 | 描述 | 精度流 | 属性 |
| :---: | :---: | :--: | :--: | :--- | :--: | :--: |
| 0x0C | 0x0 | `vfncvt.bf16` | `vfncvt.bf16 rd, acc` | 反量化: I32(ACC) → BF16 | I32 → BF16 | [ACC] |
| 0x0C | 0x1 | `vfncvt.f` | `vfncvt.f rd, acc` | 舍入: FP32(ACC) → BF16 | FP32 → BF16 | [ACC] |
| 0x0C | 0x2 | `vfwcvt.bf16` | `vfwcvt.bf16 rd, vs2` | 转换: I8 → BF16 | I8 → BF16 | - |
| 0x0C | 0x3 | `vfncvt.i8` | `vfncvt.i8 rd, vs2` | 转换: BF16 → I8 | BF16 → I8 | - |
| 0x0C | 0x4 | `vncvt.s16` | `vncvt.s16 rd, acc` | 饱和: I32(ACC) → I16 | I32 → I16 | [ACC] |
| 0x0C | 0x5 | `vwmv.s16` | `vwmv.s16 rd, vs2` | 扩位: I8 → I16 | I8 → I16 | - |
| 0x0C | 0x6 | `vncvt.s8` | `vncvt.s8 rd, vs2` | 饱和: I16 → I8 | I16 → I8 | - |
| 0x0C | 0x7 | `vfncvt.s8.f` | `vfncvt.s8.f rd, acc` | 饱和舍入: FP32(ACC) → I8 | FP32 → I8 | [ACC] |
| 0x0D | 0x0 | `vmv.i` | `vmv.i vd, imm` | 立即数广播 | - | - |
| 0x0D | 0x1 | `vmv.s` | `vmv.s vd, vs1[idx]` | 标量/元素广播 | - | - |
| 0x0D | 0x2 | `vmv.acc2x` | `vmv.acc2x vd` | ACC → VRF (详情见 4.7.1)<br>支持 8/16/32b 切片与全量 | - | [ACC] |
| 0x0D | 0x3 | `vmv.x2acc` | `vmv.x2acc vs2` | VRF → ACC (详情见 4.7.1)<br>支持 8/16/32b 切片与全量 | - | [ACC] |
| 0x0D | 0x4 | `vmerge` (*) | - | 掩码合并 (未实现) | - | - |
| 0x0E | - | `Mask Ops` (*) | - | 掩码指令 (未实现: vmseq, vmslt...) | - | - |

### 4.7.1 ACC 搬运指令详解 (ACC Transfer Detail)

根据 **SEW** 配置，支持全量搬运与切片读写。

**1. `vmv.acc2x vd` (ACC → VRF)**

| SEW | 模式 | 寄存器组 | 约束 | Scalar | 行为描述 (Lane Level) |
| :--: | :--- | :---: | :--- | :--- | :--- |
| **32b** | 全量 | 4 | `vd%4==0` | - | `v[vd...vd+3] = acc` (32b全搬, 4周期) |
| **16b** | 切片 | 2 | `vd%2==0` | `sel=s[0]` | `v[vd...vd+1] = acc.H[sel]` (16b切片, 2周期) |
| **8b** | 切片 | 1 | - | `sel=s[1:0]` | `v[vd] = acc.B[sel]` (8b切片, 1周期) |

**2. `vmv.x2acc vs2` (VRF → ACC)**

| SEW | 模式 | 寄存器组 | 约束 | Scalar | 行为描述 (Lane Level) |
| :--: | :--- | :---: | :--- | :--- | :--- |
| **32b** | 覆盖 | 4 | `vs2%4==0` | - | `acc = v[vs2...vs2+3]` (全覆盖, 4周期) |
| **16b** | 更新 | 2 | `vs2%2==0` | `sel=s[0]` | `acc.H[sel] = v[vs2...vs2+1]` (RMW, 2周期) |
| **8b** | 更新 | 1 | - | `sel=s[1:0]` | `acc.B[sel] = v[vs2]` (RMW, 1周期) |

---

## 5. 编码与寄存器映射 (Encoding)

### 5.1 MMIO 寄存器布局 (Updated)

```
┌─────────────────────────────────────────────────────────────────┐
│ VPU_REG0 (Offset 0x00) - 指令配置                               │
├───────┬───────┬───────┬───────┬─────────────┬───────┬──────────┤
│ 30:26 │ 25:21 │ 20:16 │ 15:13 │    12:10    │  9:6  │   5:0    │
│  vs2  │  vs1  │   vd  │  sew  │  reserved   │ funct │  opcode  │
└───────┴───────┴───────┴───────┴─────────────┴───────┴──────────┘

┌─────────────────────────────────────────────────────────────────┐
│ VPU_REG1 (Offset 0x04) - 标量参数                               │
├─────────────────────────────────────────────────────────────────┤
│                           31:0                                  │
│   scalar (立即数 / idx[4:0] / 广播使能[31] / SPM地址[17:0])     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 微架构 (Microarchitecture)

### 6.1 核心存储堆

| 模块 | 容量 | 描述 |
| :--- | :--- | :--- |
| **VRF** (Vector Register File) | 32 × 256 bit | 存放操作数。支持 Byte/Half/Word/Float 访问。<br>写端口: 1, 读端口: 2 |
| **ACC** (Accumulator File) | 1 × 1024 bit | 存放累加结果。<br>逻辑上视为 32 个 32-bit 寄存器。<br>支持 INT32/FP32 累加。 |

### 6.2 执行单元微架构详解 (Execution Unit Detail)

采用 **Lane-Based** 设计，VPU 包含 32 个并行 Lane，每个 Lane 绑定 1 个 32-bit ACC 切片，最大化硬件复用。

1.  **通用 MAC 核心 (Unified MAC Unit)**
    *   **乘法级**: 每个 Lane 包含 **17x17 bit 乘法器**。支持 INT16/INT8 (符号扩展) 及 BF16 (8-bit 尾数) 乘法。混合精度 `INT8 x BF16` 通过输入端动态转 BF16 复用该路径。
    *   **累加级 (Internal Adder)**: 乘法器后级成集成 **32-bit 宽位加法器**。它既是 INT32 的累加器，也是 FP32 累加操作的 **尾数加法器** (配合指数对齐逻辑)，实现了一套加法硬件对 INT32/FP32 的复用。

2.  **算术逻辑 (ALU & FPU)**
    *   **通用 ALU**: 独立的 **32-bit 分段加法器**，用于非 MAC 类的算术逻辑指令。支持 4xINT8 / 2xINT16 / 1xINT32 / 逻辑运算。
    *   **FPU 辅助**: 处理纯 BF16 加减 (非累加) 及格式转换。

3.  **后端处理 (Post-Processing)**
    *   **Quant/Sat**: 位于 VRF 写回路径，包含饱和截断逻辑，负责将 MAC/ALU 产生的宽位结果 (INT32/FP32) 转换为窄位目标格式 (INT8/BF16)。

4.  **访存单元 (LSU)**
    *   **Direct Path**: 独立于计算流水线，直接驱动 SPM 接口。
    *   **Unit-Stride**: 支持连续地址读写，单周期吞吐 256-bit。
    *   **Masking**: 写操作支持按元素掩码 (`vse8/16/32`) 控制 SPM 写使能。

5.  **共享资源 (Shared Resources)**
    *   **SFU Group**: 每 4 个 Lane 共享 1 个 SFU 实例（共 8 个）。采用 **LUT + 二阶插值** 算法实现高精度非线性函数 (`exp/log/rcp/rsqrt`)，以面积换取 4-Cycle 吞吐。
    *   **Reduction Tree**: 跨 Lane 的 **Logarithmic Tree** 互联网络 (延迟 5-Cycle)。`Sum` 规约复用 MAC 单元内的 **32-bit 宽位加法器** (支持 INT32/FP32)，`Max` 规约复用 ALU 逻辑。

---

## 7. NPU 集成指南

### 7.1 顶层模块接口 (SystemVerilog)

```systemverilog
module vpu_top 
    import npu_config_pkg::*;
#(
    parameter VLEN       = 256,
    parameter NUM_VREGS  = 32
) (
    input  logic                        clk,
    input  logic                        rst_n,
    
    // 控制接口
    input  logic                        vpu_req_en,
    output logic                        vpu_busy,
    output logic                        vpu_comp_done,
    
    // 配置接口 (Updated: 无 acc_sel)
    input  logic [5:0]                  cfg_vpu_opcode,
    input  logic [3:0]                  cfg_vpu_funct,
    input  logic [4:0]                  cfg_vpu_vs1,
    input  logic [4:0]                  cfg_vpu_vs2,
    input  logic [4:0]                  cfg_vpu_vd,
    input  logic [31:0]                 cfg_vpu_scalar,
    input  logic [2:0]                  cfg_vpu_sew,
    
    // SPM 接口
    output logic                        vpu_spm_rd_en,
    output logic [SPM_ADDR_WIDTH-1:0]   vpu_spm_rd_addr,
    input  logic [SPM_DATA_WIDTH-1:0]   vpu_spm_rd_data,
    output logic                        vpu_spm_wr_en,
    output logic [SPM_ADDR_WIDTH-1:0]   vpu_spm_wr_addr,
    output logic [SPM_DATA_WIDTH-1:0]   vpu_spm_wr_data,
    output logic [PE_WIDTH-1:0]         vpu_spm_wr_mask
);
```

---

## 8. 驱动编程参考

### 8.1 C 驱动宏

```c
#define VPU_REG0_OFFSET  0x60  // Config
#define VPU_REG1_OFFSET  0x64  // Scalar
#define VPU_START_BIT    6     

#define VPU_ENCODE_REG0(op, f, sew, vd, vs1, vs2) \
    (((vs2) << 26) | ((vs1) << 21) | ((vd) << 16) | \
     ((sew) << 13) | ((f) << 6) | (op))

static inline void vpu_exec(uint32_t reg0, uint32_t scalar) {
    mmio_write(VPU_REG0_OFFSET, reg0);
    mmio_write(VPU_REG1_OFFSET, scalar);
    mmio_write(START_REG_OFFSET, 1 << VPU_START_BIT);
    while (mmio_read(STATUS_REG) & VPU_BUSY_MASK);
}
```

### 8.2 GEMV 示例 (Updated)

```c
// Y = A * X
void vpu_gemv_int8(int8_t* A, int8_t* X, int32_t* Y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int k = 0; k < cols; k += 32) {
            // vle8.v v1, A_ptr
            vpu_exec(VPU_ENCODE_REG0(0x00, 0, 0, 1, 0, 0), (uint32_t)&A[r*cols + k]);
            // vle8.v v2, X_ptr
            vpu_exec(VPU_ENCODE_REG0(0x00, 0, 0, 2, 0, 0), (uint32_t)&X[k]);
            // vwmac.s8 acc, v1, v2
            vpu_exec(VPU_ENCODE_REG0(0x05, 0x1, 0, 0, 2, 1), 0);
        }
        
        // vse32.v acc, Y_ptr
        vpu_exec(VPU_ENCODE_REG0(0x01, 0, 2, 0, 0, 0), (uint32_t)&Y[r]);
    }
}
```

---

## 附录: Version History
- **v1.3**: Reduced ACC to 1. Simplified Mnemonics. Added ACC instructions.
- **v1.4**: Detailed Micro architecture and explanations for `vmv.acc2x` and `vmv.x2acc`. 
