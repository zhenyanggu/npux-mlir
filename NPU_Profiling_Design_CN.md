# NPU Profiling 设计说明（面向人类与 Agent）

本文档描述当前仓库中的 NPU/CPU 分层 profiling 设计，目标有三点：

- 给开发者快速说明 profiling 数据是如何在编译期和运行期串起来的。
- 给后续 Agent 一份可以直接落地修改的实现地图。
- 解释 `profile_manifest.json` 与 `profile_report.json` 的分工，以及为什么需要两份 JSON。

---

## 1. 一眼看懂设计

```text
ONNX / ONNX-MLIR
  -> convert-npu-onnx-to-linalg
     (可下沉 NPU 的算子被转成带 profiling 元数据的 linalg.generic)
  -> npu-profile-annotate
     (按 ONNX 逻辑层分组，插入 npu_profile_begin/end，并生成 profile_manifest.json)
  -> npu-tiling / dma / bufferization / npux / llvm lowering
     (profiling begin/end 调用跟随整条 lowering 链路继续向下)
  -> 目标二进制
  -> 运行时执行 npu_profile_begin/end
     (统计每层 wall-clock + NPU 子阶段时间)
  -> profile_report.json
     (按 layer_id 输出动态时间，并自动合并 manifest 的 layer_name/op type)
```

最核心的设计原则是：

- `manifest` 负责回答“这一层是谁”。
- `report` 负责回答“这一层跑了多久”。
- 两者通过稳定的 `layer_id` 关联。

---

## 2. 为什么需要两份 JSON

### 2.1 profile_manifest.json

编译期生成，保存静态信息，例如：

- `layer_id`
- `onnx_node_name`
- `origin_op_type`
- `device`
- `fused_ops`
- 输入/输出 shape
- dtype
- `input_bytes`
- `output_bytes`
- `weight_bytes`
- `macs`
- `fallback_reason`

本质上它是“层索引表”。

### 2.2 profile_report.json

运行期生成，保存动态信息，例如：

- `layer_id`
- `invocations`
- `total_ns`
- `avg_ns`
- `pct_total`
- `dma_in_ns`
- `compute_ns`
- `dma_out_ns`
- `layout_ns`
- `wait_irq_ns`
- `other_ns`
- `mvin_calls`
- `compute_calls`
- `mvout_calls`
- `layout_calls`

本质上它是“执行结果表”。

### 2.3 为什么不只保留 report

因为 layer 名称、融合关系、shape、fallback 原因等信息在编译期最清楚，运行期只知道“当前层 id 是多少”。如果把所有静态信息都重新编码进运行时调用接口，会让接口侵入性很强，也会让 lowering 更复杂。

所以当前方案采用：

- 编译期生成 manifest。
- 运行期 report 通过 `layer_id` 合并 manifest 的关键信息。

---

## 3. 编译期设计

### 3.1 profiling 元数据在哪一层挂上去

NPU lowering 会把同一逻辑层生成的 NPU op 标上统一属性：

- `npu.layer_name`
- `npu.layer_kind`
- `npu.fused_ops`
- `npu.origin_op_type`
- `onnx_node_name`

这些属性的目的不是给最终硬件直接使用，而是为了后续的：

- 逻辑层分组
- manifest 输出
- 可读性分析

### 3.2 关键文件

- `src/Conversion/NpuPartition/LinalgConversionHelper.hpp`
- `src/Conversion/NpuPartition/LinalgConversionHelper.cpp`

它们提供统一 helper，用于根据源 ONNX op 生成 profiling 属性，避免每个 lowering 文件手写一套命名逻辑。

### 3.3 已覆盖的 lowering 文件

当前已经补齐 profiling 属性的 NPU lowering 包括：

- `src/Conversion/NpuPartition/Linalg/Conv.cpp`
- `src/Conversion/NpuPartition/Linalg/Gemm.cpp`
- `src/Conversion/NpuPartition/Linalg/Binary.cpp`
- `src/Conversion/NpuPartition/Linalg/Unary.cpp`
- `src/Conversion/NpuPartition/Linalg/LayerNorm.cpp`
- `src/Conversion/NpuPartition/Linalg/UnaryUnquant.cpp`

覆盖的逻辑算子包括：

- `Conv`
- `Gemm`
- `MatMul`
- `Add`
- `Softmax`
- `Gelu`
- `LayerNorm`
- `Transpose`
- `MaxPool`
- `Resize`

### 3.4 npu-profile-annotate pass

关键文件：

- `src/Conversion/NpuPartition/NpuProfileAnnotate.cpp`

pass 名称：

- `--npu-profile-annotate`

pass 位置：

- 放在 `--convert-npu-onnx-to-linalg` 之后
- 放在 `--npu-tiling` 之前

当前在两条链路都已接入：

- `src/Compiler/CompilerPasses.cpp`
- `model_test/scripts/onnx_to_llvm.sh`

### 3.5 这个 pass 做了什么

它会顺序扫描函数 block，并按以下规则组层：

#### NPU 路径

- 连续出现、且 `npu.layer_name` 相同的 op，会被视作一个 ONNX 逻辑层。
- 例如一个 `Conv` 逻辑层内部可能包含：
  - `layout_in`
  - `compute`
  - `quant`
  - `layout_out`

#### CPU 路径

- 对没有被下沉到 NPU 的 ONNX 计算 op，按 op 本身建立 CPU 逻辑层。
- 为了让 profiling 边界在 CPU lowering 后仍能包住真实计算，CPU 层的
  begin/end 会吸收与该 op 直接相邻的 activation-side
  `DequantizeLinear` / `QuantizeLinear` helper 作为边界，但这些 helper
  仍不会单独生成层记录。
- 常量权重/偏置侧的 `DequantizeLinear` 不会再被直接拉到 layer begin 上；
  否则它们在 block 中被提前放置时，容易把后续 layer 的 begin 挪到前一层
  end 之前，形成嵌套 profiling 区间。
- 同一 block 内收集出的 layer 边界会在 pass 末尾做一次顺序归一化，保证
  begin/end 单调不交叉。这与运行期“同线程只维护一个 active layer”的语义
  保持一致。

#### 跳过的辅助 op

- `Constant`
- `QuantizeLinear`
- `DequantizeLinear`
- 其他纯辅助节点不进入主层列表

这里的“不进入主层列表”指的是：

- helper op 不会单独占用一个 `layer_id`
- 但在 CPU fallback 场景下，紧邻主算子的 Q/DQ helper 允许被吸收到该层的
  profiling 边界中，以避免后续 bufferization / memref lowering 把真实计算
  移到 begin/end 之外

#### 插入的调用

每层会插入：

- `func.call @npu_profile_begin(layer_id)`
- `func.call @npu_profile_end(layer_id)`

并在同一 pass 中输出：

- `profile_manifest.json`

### 3.6 为什么在高层插 begin/end

这是当前设计里非常关键的一点。

把 begin/end 放在高层 IR 的好处是：

- 不需要在 tiling、DMA、bufferization、npux lowering 每一层都继续手传层名。
- profiling 边界会自动包住后续 lowering 生成的完整执行序列。
- 对 CPU fallback 与 NPU 路径可以用同一套逻辑层定义。

代价是：

- 需要保证下游 pass 能接受 declaration-only 的 profiling helper 函数。

为此，当前已经在下面文件补了兼容：

- `src/Dialect/ONNX/ONNXDimAnalysis.cpp`

这里会显式跳过 declaration-only `func.func`，避免在 `buildFunctionArgsRes` 中访问空 region。

---

## 4. 运行期设计

### 4.1 新增的 C API

关键文件：

- `model_test/runtime/npu_runtime.h`
- `model_test/runtime/npu_runtime.cpp`

新增接口：

- `void npu_profile_begin(int64_t layer_id);`
- `void npu_profile_end(int64_t layer_id);`
- `void npu_profile_dump(const char *path);`

### 4.2 运行期状态机

运行期通过线程本地 `current layer` 维护当前活跃层：

- `npu_profile_begin(layer_id)` 开始计时
- `npu_profile_end(layer_id)` 结束该层 wall-clock 计时

同时，NPU 运行时内部会在更细粒度的函数上累计子阶段时间。

### 4.3 NPU 子阶段分类

当前固定子阶段为：

- `dma_in`
- `compute`
- `dma_out`
- `layout`
- `wait_irq`
- `other`

典型归类如下：

- `run_mvin` -> `dma_in`
- `run_mvout` -> `dma_out`
- `run_gemm / run_conv / run_sfu / run_matadd / run_resample` -> `compute`
- `run_transpose` -> `layout`
- `wait_irq` -> `wait_irq`

### 4.4 report 如何拿到 layer_name / op type

运行期会在写 `profile_report.json` 时，尝试加载 manifest 并合并以下字段：

- `layer_name`
- `onnx_node_name`
- `origin_op_type`
- `fused_ops`
- `fallback_reason`

也就是说，最终 report 是“动态时间 + 关键静态元数据”的合并结果。

### 4.5 manifest 查找顺序

当前运行期支持：

#### 显式指定

通过环境变量：

- `NPU_PROFILE_MANIFEST`

#### 自动推断

如果没有显式指定，运行期会按 report 输出路径推断：

- 同目录下的 `profile_manifest.json`
- 同目录下与 report 同前缀的 `*_profile_manifest.json`
- report 目录下的 `NpuPartition/profile_manifest.json`

这意味着：

- 如果最终输出目录里已经有 manifest，report 会自动补齐层名。
- 如果 report 被写到别的目录，可以显式设置 `NPU_PROFILE_MANIFEST`。

### 4.6 运行期环境变量

当前支持：

- `NPU_PROFILE_OUT`
  - 覆盖 report 输出路径。
- `NPU_PROFILE_MANIFEST`
  - 显式指定 manifest 路径。
- `NPU_PROFILE_DISABLE=1`
  - 禁用 profiling 输出。

### 4.7 _mlir_ciface 兼容

因为高层插入的是 `func.func @npu_profile_begin/end`，LLVM lowering 后会生成 `_mlir_ciface_npu_profile_begin/end` 相关符号。

为避免最终链接 unresolved symbol，运行时额外导出了：

- `_mlir_ciface_npu_profile_begin`
- `_mlir_ciface_npu_profile_end`

它们只是转发到：

- `npu_profile_begin`
- `npu_profile_end`

这属于链接兼容层，不属于业务逻辑层。

---

## 5. 输出文件与路径约定

### 5.1 编译期 manifest

在 `model_test/scripts/onnx_to_llvm.sh` 中，默认生成位置为：

- `NpuPartition/profile_manifest.json`

即相对当前模型工作目录：

- `<model_work_dir>/NpuPartition/profile_manifest.json`

### 5.2 最终输出目录中的 manifest

为了让板端或交叉编译后的产物更容易携带 profiling 元数据，当前已经在两处加入复制逻辑：

- `model_test/scripts/compile_to_zcu102.sh`
- `model_test/makefile`

当走 `make <model>.zcu102` 或直接调用 `compile_to_zcu102.sh` 时，会把：

- `<model_work_dir>/NpuPartition/profile_manifest.json`

复制到最终输出目录：

- `<output_dir>/profile_manifest.json`

### 5.3 运行期 report

默认输出：

- `profile_report.json`

如果指定 `NPU_PROFILE_OUT`，则按该路径写出。

---

## 6. 关键代码入口总表

### 编译器

- `src/Conversion/NpuPartition/NpuProfileAnnotate.cpp`
  - profiling annotate pass，本设计的编译期核心。
- `src/Conversion/NpuPartition/LinalgConversionHelper.cpp`
  - profiling 属性 helper。
- `src/Compiler/CompilerPasses.cpp`
  - 默认编译 pipeline 接入点。
- `src/Tools/onnx-mlir-opt/RegisterPasses.cpp`
  - pass 注册点。
- `src/Pass/Passes.hpp`
  - pass 声明。

### lowering 元数据接入点

- `src/Conversion/NpuPartition/Linalg/Conv.cpp`
- `src/Conversion/NpuPartition/Linalg/Gemm.cpp`
- `src/Conversion/NpuPartition/Linalg/Binary.cpp`
- `src/Conversion/NpuPartition/Linalg/Unary.cpp`
- `src/Conversion/NpuPartition/Linalg/LayerNorm.cpp`
- `src/Conversion/NpuPartition/Linalg/UnaryUnquant.cpp`

### 运行时

- `model_test/runtime/npu_runtime.h`
- `model_test/runtime/npu_runtime.cpp`

### 脚本/构建

- `model_test/scripts/onnx_to_llvm.sh`
- `model_test/scripts/compile_to_zcu102.sh`
- `model_test/makefile`

### 分析兼容修复

- `src/Dialect/ONNX/ONNXDimAnalysis.cpp`

---

## 7. 典型调试方法

### 7.1 看 manifest 是否生成

在模型工作目录中检查：

- `NpuPartition/profile_manifest.json`

### 7.2 看 begin/end 是否插入

检查中间 IR：

- `NpuPartition/NpuProfileAnnotate.mlir`

搜索：

- `npu_profile_begin`
- `npu_profile_end`

### 7.3 看 LLVM lowering 后是否还在

检查：

- `NpuToLLVM/llvm.mlir`

搜索：

- `llvm.call @npu_profile_begin`
- `llvm.call @npu_profile_end`

### 7.4 看 report 是否补齐层名

检查生成的 `profile_report.json`，确认每层除了 `layer_id` 外，还有：

- `layer_name`
- `origin_op_type`

如果没有，先排查：

1. report 生成时 manifest 是否存在。
2. report 与 manifest 是否位于同一输出目录。
3. 是否需要显式设置 `NPU_PROFILE_MANIFEST`。

---

## 8. 当前限制与注意事项

### 8.1 统计假设

当前实现默认假设：

- 单线程
- 非重叠执行
- 以 wall-clock 时间统计每层总时间

因此：

- 如果未来存在异步 overlap、双缓冲重叠、跨层流水，当前 `total_ns` 和子阶段之和之间的关系需要重新解释。

### 8.2 manifest 解析器是轻量实现

运行期为了避免引入额外 JSON 依赖，当前对 manifest 的读取采用轻量级、面向当前 schema 的解析方式。

这意味着：

- 如果未来 `profile_manifest.json` 的格式大改，`npu_runtime.cpp` 中的 manifest 解析逻辑也要同步更新。

建议后续 Agent 在修改 manifest schema 时，优先同步检查：

- `resolveProfileManifestPath`
- `loadManifestMetadata`
- `dumpProfilerReport`

### 8.3 npu_macs_share 仍为 null

当前 report 的 summary 中：

- `npu_macs_share`

仍未在运行时回填，因为 MACS 主要保存在 manifest 中，尚未做 summary 级聚合输出。

### 8.4 CPU/NPU device 优先级

当前 report 中每层 `device` 优先使用 manifest 的静态 `device`。

如果 manifest 不可用，则退回为运行时根据是否存在 NPU 子阶段样本来推断：

- 有 NPU 子阶段样本 -> `npu`
- 否则 -> `cpu`

---

## 9. 扩展一个新算子时应该改哪里

如果后续新增一个可下沉到 NPU 的逻辑算子，例如 `ReduceMean`，通常需要改四类位置：

### 9.1 lowering 打 profiling 元数据

在对应 lowering 文件中调用 helper，补齐：

- `npu.layer_name`
- `npu.layer_kind`
- `npu.fused_ops`
- `npu.origin_op_type`

### 9.2 如果它会映射到新的运行时硬件调用

需要在 `npu_runtime.cpp` 中决定它属于：

- `dma_in`
- `compute`
- `dma_out`
- `layout`
- `wait_irq`

或者新增阶段。

### 9.3 如果 manifest schema 需要扩展

修改：

- `src/Conversion/NpuPartition/NpuProfileAnnotate.cpp`

并同步修改运行时 manifest 解析。

### 9.4 如果最终输出目录需要带上新 sidecar 文件

检查：

- `model_test/scripts/compile_to_zcu102.sh`
- `model_test/makefile`

---

## 10. 当前状态（截至本仓库现状）

当前 profiling 方案已经具备以下能力：

- 编译期按 ONNX 逻辑层生成稳定 `layer_id`
- NPU/CPU 分层统计
- NPU 子阶段统计
- 编译期 manifest 输出
- 运行期 report 输出
- report 自动合并 `layer_name/op type`
- 输出目录自动携带 manifest

这套设计的推荐理解方式是：

- manifest 是“静态层地图”
- report 是“动态执行账本”
- 两者共同构成 profiling 结果
