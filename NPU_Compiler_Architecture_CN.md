# NPU 编译器架构笔记（面向快速定位与二次开发）

本文档基于当前仓库代码整理，目标有两点：
- 给开发者一份可直接定位代码的架构地图。
- 给后续 AI 一份高密度上下文，减少重复读码和 token 消耗。

profiling 相关设计与实现细节已单独整理到 `NPU_Profiling_Design_CN.md`。

---

## 1. 一眼看懂全链路

```text
ONNX
  -> convert-npu-onnx-to-linalg
     (把可下沉算子变成带 library_call 的 linalg.generic)
  -> npu-tiling
     (第一阶段分块：容量导向，含 Gemm Tm/Tn/Tk 计算与 loop_stage)
  -> npu-insert-dma
     (显式插入 npu_dma_mvin / npu_dma_mvout 逻辑节点)
  -> npu-op-splitting
     (第二阶段分块：硬件导向，Gemm N/M=32 + K<=2048 + split_stage)
  -> convert-onnx-to-krnl / bufferization / affine lowering
  -> convert-linalg-to-npux
     (linalg.generic -> npux.compute_run / npux.dma_*)
  -> npux-compute-fusion (可选)
  -> npu-memory-plan
     (给 npux sram/acc alloc 写 npu.offset)
  -> convert-npux-to-llvm
     (npux.* -> npu_runtime C API 调用)
  -> LLVM IR / 目标二进制
```

说明：
- 默认 `CompilerPasses.cpp` 的 NPU 管线与 `model_test/scripts/onnx_to_llvm.sh` 不完全一致。
- 日常调试 NPU 常用的是 `model_test` 脚本链路（分阶段落盘，便于看中间 IR）。

---

## 2. 关键目录与职责

- `src/Conversion/NpuPartition/`
  - ONNX -> Linalg(NPU) 的入口。
  - 典型文件：`ConvertONNXToLinalgNpu.cpp`、`Linalg/Gemm.cpp`。
- `src/Conversion/NpuTiling/`
  - 第一阶段 tiling、DMA 插入、第二阶段 splitting。
  - 典型文件：`Gemm.cpp`、`NpuInsertDma.cpp`、`NpuOpSplitting.cpp`。
- `src/Conversion/NpuBufferization/`
  - One-shot bufferization + DPS 结果外提。
- `src/Conversion/NpuToLLVM/`
  - linalg -> npux；npux -> LLVM；内存规划；地址空间收尾。
  - 典型文件：`ConvertLinalgToNpux.cpp`、`Npux/ComputeOpConvert.cpp`、`ConvertNpuxToLLVM.cpp`、`NpuMemPlan.cpp`。
- `src/Dialect/Npux/`
  - NPU 自定义 Dialect（`npux.compute_run`, `npux.dma_mvin`, `npux.mvin_bias` 等）。
- `model_test/runtime/`
  - C API 与驱动交互实现（`npu_gemm_run`, `npu_dma_mvin`, `npu_conv_run` 等）。

---

## 3. 数据与内存语义（非常关键）

### 3.1 Tensor encoding -> MemRef memory space

- 在 NPU 路径里，tensor encoding 被用于驱动 bufferization 的 memory space 选择。
- 常见空间语义：
  - `space=1`: NPU 侧 DRAM（Host 可见）
  - `space=2`: SPM
  - `space=3`: ACC

### 3.2 Npux 层内存算子

- `npux.sram_alloc/sram_free` 管理 SPM。
- `npux.acc_alloc/acc_free` 管理 ACC。
- `npu-memory-plan` 给这些 alloc 打 `npu.offset`，后续 lowering 直接取偏移拼 CAPI 地址。

### 3.3 最后地址空间清理

- `npu-erase-memory-space` 会把 `memref<..., 1>` 转回默认地址空间（0），避免 LLVM 阶段地址空间不匹配。

---

## 4. GEMM 端到端策略（当前实现）

下面是你最关心的部分。

### 4.1 ONNX -> linalg.generic 阶段

文件：`src/Conversion/NpuPartition/Linalg/Gemm.cpp`

- Gemm/MatMul 会生成 `library_call = "npu_gemm"` 或 `"npu_matmul"` 的 `linalg.generic`。
- 结果通常再接一个 `library_call = "mv_acc_to_spm"` 的 `linalg.generic` 表达 ACC->SPM 语义。

### 4.2 第一阶段分块（容量导向）

文件：`src/Conversion/NpuTiling/Gemm.cpp`

核心点：
- 触发点是 `mv_acc_to_spm`，并融合其 producer (`npu_gemm/npu_matmul`)。
- `getGemmTileSizes` 计算 `Tm/Tn/Tk`，并按迭代器顺序映射到 `[N, M, K]` 或 `[B, N, M, K]`。
- 自动 tile 算法 `calculateAutoGemmTile` 的优先级是：
  - 先尽量大 `Tk`（受 SPM/ACC 容量约束）
  - 再放大 `Tm`
  - 最后 `Tn`
- 当前代码里，第一阶段 **不再引入 2048 上限**；2048 约束交给第二阶段硬件切分。
- 第一阶段会做 K 维循环 peeling，并写 `npu.loop_stage = head/body/tail/single`。

### 4.3 第二阶段分块（硬件导向）

文件：`src/Conversion/NpuTiling/NpuOpSplitting.cpp`

`NpuGemmTilingPattern` 的逻辑：
- N/M 维固定按 32 切分（硬件 32x32 输出阵列限制）。
- 新增 K 维保护：`kMaxGemmKTile = 2048`。
- 当且仅当 `K` 是静态且 `K > 2048` 时：
  - 只在最后一维 K 上 `tileUsingSCF(step=2048)`。
  - K loop 打 `npu.split_dim = "K"`。
  - 对 K loop 进行 head/body/tail 或 single 标注：
    - `npu.split_stage = head/body/tail/single`。
- 该 K 分块逻辑会同时覆盖：
  - 简单 Gemm 路径（无 `mv_acc_to_spm` fuse）。
  - `mv_acc_to_spm + gemm` fuse 路径。

### 4.4 `loop_stage` 与 `split_stage` 的职责划分

文件：`src/Conversion/NpuToLLVM/Npux/ComputeOpConvert.cpp`

- `loop_stage`：第一阶段（容量分块）传递的“是否首块/末块”。
- `split_stage`：第二阶段（硬件分块）传递的“是否首块/末块”。
- 代码里会合并判断：
  - `isFirstCalculation = isLoopFirst && isSplitFirst`
  - `isLastCalculation  = isLoopLast  && isSplitLast`

因此：
- 只有“绝对第一块”会走 bias 首次加载（`npux.mvin_bias`）逻辑。
- 后续块走 psum 累加。
- 只有“绝对最后块”才允许最终激活/输出路径收尾。

这就是你描述的“首块 bias、后续累加、末块输出”的实现落点。

### 4.5 2048 约束的最后防线

同文件 `ComputeOpConvert.cpp` 在生成 `npux.compute_run` 前有硬件字段校验：
- `a_col_m1 <= 2047`
- `b_row_m1 <= 2047`

若仍超限会直接报错，说明前面 K split 没把 tile 压到 `K<=2048`。

### 4.6 MATADD（新增）

端到端路径：
- `ONNXAdd`（严格 QDQ、同形状、rank<=2、对称量化）  
  -> `linalg.generic {library_call="npu_matadd"}`  
  -> `npu-tiling`（`ElemWise.cpp`）  
  -> `npu-insert-dma`（两路 `mvin->ACC` + 一路 `mvout`）  
  -> `npux.matadd_run`  
  -> `npu_matadd_run` C API。

ACC 放不下时的分块策略（当前实现）：
- 在 `ElemWise.cpp` 中为 `npu_matadd` 使用容量上限：
  - `maxByAcc = acc_bytes / (2 * 4)`（A/B 两路在 ACC 中按 int32 计）
  - `maxBySpm = spm_bytes / out_elem_bytes`
  - `maxElems = min(maxByAcc, maxBySpm)`。
- 再按“内层优先”的贪心分块生成 tile。
- 额外强制 `row<=255`、`col<=255`，对应 MATADD 控制寄存器的 8-bit 字段限制。

---

## 5. DMA 与日志解读

### 5.1 为什么日志里常出现 `row_num=0`

先记住：DMA/GEMM 里很多字段是 `*_m1` 编码。
- `row_num=0` 实际代表 `1` 行。
- 例如 `col_num=2047,row_num=0` 对应形状约为 `1x2048`。

### 5.2 DMA 形状从哪里来

文件：`src/Conversion/NpuToLLVM/Npux/NpuMemoryManagement.cpp`

- `ConvertMemrefCopyToNpuxPattern` 会把 memref shape 扁平到 `(rows, cols)`。
- 然后生成 `npux.dma_mvin/mvout`，再在 LLVM lowering 中映射到 `npu_dma_mvin/mvout`。

---

## 6. 编译管线差异（默认 vs model_test）

### 6.1 默认编译器管线（`CompilerPasses.cpp`）

包含但不限于：
- `createONNXToLinalgNpuPass`
- `createNpuMergePass`
- `createNpuOutlinePass`
- nested `createNpuTilingPass`
- `createNpuDPSConversionPass`
- `createConvertLinalgToNpuPass`
- `createNpuMemPlanPass`

### 6.2 `model_test/scripts/onnx_to_llvm.sh` 管线

更偏调试和可观测：
- `--npu-tiling`
- `--npu-insert-dma`
- `--npu-op-splitting --npu-remove-redundant-dma`
- `--convert-linalg-to-npux`
- `--npux-compute-fusion`
- `--npu-memory-plan`

你现在观察到的 Gemm 分块行为，通常以这条脚本链路为准。

---

## 7. 常见修改入口（按需求反查）

- 改 ONNX 算子是否下沉 NPU：
  - `src/Conversion/NpuPartition/ConvertONNXToLinalgNpu.cpp`
- 改 Gemm 第一阶段容量分块（Tm/Tn/Tk 公式）：
  - `src/Conversion/NpuTiling/Gemm.cpp`
- 改 Gemm 第二阶段硬件分块（N/M/K、2048）：
  - `src/Conversion/NpuTiling/NpuOpSplitting.cpp`
- 改 bias/psum/accumulate 判定：
  - `src/Conversion/NpuToLLVM/Npux/ComputeOpConvert.cpp`
- 改最终 CAPI 参数映射：
  - `src/Conversion/NpuToLLVM/ConvertNpuxToLLVM.cpp`
- 改 runtime 寄存器写法与 CAPI 日志：
  - `model_test/runtime/npu_runtime.cpp`

---

## 8. 当前 GEMM 策略的简化结论

一句话版：
- 第一阶段按容量定大 tile（不含 2048 人工截断），第二阶段按硬件规则切成 N/M=32 与必要的 K=2048 子块，再通过 `loop_stage + split_stage` 驱动 bias 首块、psum 累加和末块输出。

---

## 9. 后续可继续补充（建议）

- 增加一份“从某条 `NPU_CAPI` 日志反推 IR 节点”的对照表。
- 增加一份“常见性能异常 checklist”（重复 mvin、偏置重复搬运、K split 失效、融合失败）。
- 增加一个固定模板脚本，自动从中间 IR 抽取：
  - `npu.loop_stage`
  - `npu.split_stage`
  - `npu.split_dim`
  - `library_call`
