# Versa-P 网络级 Descriptor 调度实施方案

## 1. 目标与边界

目标是在 `--npu-hardware-abi=versa-p-v1` 路径中，把已经完成 NPU
partition/tiling 的子图转换为可执行、可检查的 Versa-P descriptor 调度序列。
调度器必须显式管理 A0/A1、W0/W1、O0..O3 和 metadata ping-pong bank，
并将数据依赖映射为命令完成等待。

第一阶段只覆盖静态 shape 的 GEMM/MatMul 子图及其 DMA、SA_COMPUTE 和
MVOUT 命令。VPU、GEMV、attention QK/PV、动态 shape、板端 DMA/cache/IRQ
验证属于后续阶段，不能在第一阶段隐式启用。

本方案以 `NPUHardwareSpecificationTemplate_CN.md` 为 ABI 合约；RTL、RDL
和成功回归优先于历史 runtime 的 `npu_gemm_run` 语义。

### 1.1 冻结基线与变更规则

- 本计划的 ABI 基线为仓库根目录 `NPUHardwareSpecificationTemplate_CN.md`，
  SHA-256 为 `BDABEF06F134EDAF27F3A8F79CB8B4E7A11025B3E9D4341C772F22715E7FF4BE`
  （2026-08-07）。任何 descriptor 字段、bank 容量或完成语义的修改，必须先更新
  该合约及其哈希，再更新 encoder、scheduler、runtime 和本计划。
- `doc/rdl/npu_inst_ctrl_regs.rdl`、RTL 和 ABI 合约冲突时，以经 RTL 回归验证的
  RDL/RTL 为准；计划文档不得自行定义新的位域或状态语义。
- 阶段 1--5 的交付都必须保持默认关闭。只有
  `--npu-hardware-abi=versa-p-v1` 显式开启且静态资格检查通过时才可走新路径；其余
  情况继续走现有 C API，不允许因调度失败产生静默错误代码或不完整 descriptor。
- 每个阶段合入前记录输入 commit、ABI 哈希、执行命令、测试日志路径和已知限制。
  这些证据写入 `artifacts/versa_p_scheduler/`，但生成物不应提交到源码树。

## 2. 当前基线

- `VersaPDescriptor` 已提供 MVIN_A、MVIN_W、AUX_MVIN、MVOUT、GEMM 的字段编码和
  容量/对齐校验。
- `VersaPBankScheduler` 能为一个 `TileRequest` 生成 A/W/O/metadata bank
  选择、命令依赖、生命周期位和 GEMM tile 命令序列。
- 调度器尚未被 `NpuxComputeRunLowering` 调用；当前 lowering 仍生成历史
  `npu_gemm_run` C API 调用。因此现在不能产出整网 descriptor 调度结果。
- 硬件 ABI 支持 O0..O3，四个 O bank 都可被 MVOUT 导出；调度器必须将它们
  都作为候选，而不能退化为 O0/O1。

### 2.1 实现状态快照（2026-08-07）

下表以当前工作树为准，区分“代码已存在”和“端到端能力已完成”；前者不能作为阶段
验收通过的证据。

| 项目 | 现状 | 证据/缺口 |
| --- | --- | --- |
| ABI 选择 | 已定义 `legacy` 与 `versa-p-v1`，默认 `legacy` | `src/Compiler/NpuConfig.{hpp,cpp}`；开关尚未被 lowering 查询。 |
| descriptor encoder | 已实现 MVIN_A、MVIN_W、AUX_MVIN、MVOUT、GEMM 的字段编码及部分范围检查 | `VersaPDescriptorTest.cpp` 已在 143 隔离 worktree 通过 MVIN_W/AUX golden 和非法输入；其余 encoder 字段仍待补充测试。 |
| 单 tile 逻辑调度 | 已生成 LoadA/LoadW/LoadMetadata/GEMM/VPU/StoreO 的逻辑 DAG；MVIN_A 与 MVIN_W 使用独立 engine 状态，可作为并行候选 | `VersaPBankScheduler.cpp`；没有 DMA 地址、local base、descriptor 或完成状态字段，也缺自动化依赖测试。 |
| 编译集成 | encoder/scheduler 已列入 `NpuToLLVM` CMake 源列表，且显式依赖 `OMNpuxIncGen`/`OMNpuxEnums` | `NpuToLLVM`、CTest 与 `onnx-mlir-opt` 已在 143 隔离 worktree 构建；可执行文件已成功启动；尚未被 `ConvertNpuxToLLVM.cpp` 使用。 |
| LLVM lowering | `NpuxComputeRunLowering` 始终生成 `npu_gemm_run` | `ConvertNpuxToLLVM.cpp`；尚无 Versa-P 提交、等待或回退实现。 |
| 测试与 trace | `VersaPDescriptorTest` 已接入 CTest；尚无 scheduler 专项单测、lit test、mock runtime 或 JSON schema | G0 的 MVIN_W/AUX 子集已在 143 通过；G0 其余字段及 G1 以后仍必须新增验证，不能以人工审读替代。 |

本快照中的文件名仅用于定位；设计决策以 ABI 合约为准。每次改变上述状态时，必须同步
更新本节的日期和证据，避免计划与工作树再次漂移。

## 3. 目标架构

```text
npux.compute_run / npux.dma_* / npux.sfu_*
          |
          v
  VersaP schedule extraction
  - 静态形状、tile、buffer offset、量化与算子依赖
          |
          v
  NetworkSchedule
  - command DAG、bank 生命周期、descriptor 字段、等待条件
          |
          +--> 调试输出：MLIR 属性 + JSON trace
          |
          +--> descriptor lowering：64-bit MMIO descriptor 提交序列
                                      |
                                      v
                         runtime 等待 done/error 或 IRQ
```

`NetworkSchedule` 是 compiler 内部不可变调度结果。每条命令至少包含：
命令 id、算子/tile 来源、API 类型、descriptor 三个 64-bit word、读写 bank、
依赖命令 id、`*_ischange`、预期完成状态和失败诊断上下文。

### 3.1 调度对象的最小契约

当前 `VersaPBankScheduler` 仅生成逻辑 `ScheduledCommand`、bank 读写集和命令依赖；
`TileSchedule` **尚未**保存 DMA/SA/MVOUT 的编码 descriptor。阶段 1 的完成定义是将
下表字段补全，而不是把现有逻辑调度器视为已可提交实现。

| 对象 | 必填字段 | 不变量 |
| --- | --- | --- |
| `TileRequest` | 来源 op/tile 序号、A/W/O/metadata 的 DRAM byte 地址、local word base、shape/stride、元素类型、量化和生命周期需求 | 所有地址与尺寸在提取时已验证；不得以默认零地址表达未知地址。 |
| `ScheduledCommand` | 全局 id、kind/engine、读写 bank、依赖、descriptor、提交寄存器、完成状态寄存器 | 依赖 id 小于当前 id；读写集合和 descriptor 中的 bank 编码一致。 |
| `NetworkSchedule` | ABI 标识、子图 id、命令数组、fallback 决定、统计和诊断 | 命令 id 全局唯一、DAG 无环、同一 bank 的冲突访问存在完成依赖。 |
| `FallbackRecord` | op/tile、规则 id、原因、原始形状/地址 | 可稳定序列化；不得只记录泛化的“unsupported”。 |

命令依赖的语义固定为“前驱 `done` 且无 `error`”，不是 MMIO 写被接受
（`accepted_count`）的依赖。runtime 必须在开始复用一个 bank 前清除已消费的 W1C
状态；遇到 error 时该 `NetworkSchedule` 的剩余命令不得继续提交。

### 3.2 静态资格与回退边界

阶段 2 入口必须在生成 descriptor 之前执行，并对每个候选 NPU 子图给出一个可观察的
决定。第一阶段不支持项应当回退，不得部分发射。

| 条件 | Versa-P v1 行为 | 诊断/回退原因 |
| --- | --- | --- |
| GEMM/MatMul，M/N/K 均静态且 `0 < K <= 4096` | 进入 tile 提取 | -- |
| 动态维度、符号 stride 或无法折叠的 subview offset | 回退 | `dynamic-shape-or-offset` |
| DMA base 或非零 stride 非 32-byte 对齐 | 回退 | `dma-alignment` |
| tile 超出 A/W/O 2048 word 或 metadata 512 word | 重新切 tile；仍不能满足则回退 | `local-bank-capacity` |
| 未实现的算子、数据布局或量化组合 | 回退 | 对应稳定规则 id |
| `MAGIC/VERSION/MODE/CAPS` 不匹配或 runtime 报错 | 中止该 schedule，GLOBAL_CLEAR 后走安全路径 | `abi-or-runtime-error` |

## 4. 阶段 0：合约冻结与可编译基线

**实现内容**

- 将当前 `T_NPU/NPUHardwareSpecificationTemplate_CN.md` 同步到本仓库，
  并记录 SHA-256。
- 将 RDL 位定义和编译器 `VersaPDescriptor` 的 bit packing 做逐字段对照。
- 明确支持矩阵、数据类型、量化模式、bank 容量、对齐、K<=4096、错误码和
  不支持项。
- 在 143 的 `gpu01` 工作区建立干净构建目录，编译 `onnx-mlir-opt` 和受影响
  的 NPU lowering 库。

**验收**

- 规范副本哈希与 `T_NPU` 相同。
- descriptor 单元测试覆盖每个字段的 bit 位置及非法输入。
- 143 构建成功，日志保存到 `artifacts/versa_p_scheduler/phase0-build.log`。

## 5. 阶段 1：调度数据模型与单 Tile GEMM

**实现内容**

- 完善 `TileRequest`，携带 tile 的 DRAM 地址、local base、实际行列字节数、
  bias/scale metadata、累加/残差和输出消费信息。
- 完善 `TileSchedule`，可直接构建 MVIN_A、MVIN_W、AUX_MVIN、SA_COMPUTE、
  MVOUT descriptor，不再只返回抽象 bank 编号。
- 将 `EncodedDescriptor`、descriptor 提交寄存器和完成状态寄存器附着到每条
  `ScheduledCommand`；命令 kind 与 encoder 一一对应。descriptor 编码失败必须携带
  tile/op 上下文并转换为 fallback，而不是在 lowering 中重新猜测字段。
- 用 bank 读写集合和 API busy 约束构建命令 DAG；同一 bank 的冲突命令必须
  依赖对应 `done/error`，不同 bank 的无关命令保留并行候选。
- 正确设置 SA 的 A/W/O/ACC/RESADD/meta lifecycle 位，以及最终 MVOUT 的
  `o_ischange` 位。

**验收**

- 对 raw INT32、tensor INT8、per-channel INT8、FP32 四种 output mode 的
  descriptor 进行 golden bit 测试。
- 覆盖 bias 首块、K 分段 accumulate、resadd、metadata ping-pong、O0..O3
  导出和 bank 冲突拒绝。
- 断言独立 A/W bank 的 `LoadA` 与 `LoadW` 之间不存在伪 engine 依赖；两者都必须是
  后续 GEMM 的前驱。此项防止把独立的 MVIN_A/MVIN_W ABI 接口错误收敛为单 DMA engine。
- 生成单 GEMM tile 的 JSON trace，人工可确认命令顺序、依赖和 bank 变化。
- 为 scheduler 增加自检：拓扑排序后每个读取都由已完成的写入或显式 DRAM reload
  满足；每个 bank 的生命周期只在对应 done 之后结束；descriptor 中的 bank 与读写集
  完全相同。

## 6. 阶段 2：从 NPU IR 提取 GEMM tile

**实现内容**

- 在 NPU tiling/splitting 后读取 `npux.compute_run` 的静态 tile 形状、
  `npu.loop_stage`、`npu.split_stage`、`npu.split_dim` 和 memory-plan offset。
- 将首块 bias、后续 psum accumulate、末块输出的现有语义转成 `TileRequest`。
- 将 memref/subview 的 offset、shape、stride 和元素类型转换为 DMA 的 byte
  地址、row_count、row_bytes 和 local-word base。
- 对任何动态 shape、无法解析的 offset、未对齐地址或超 bank 容量的 tile，
  产生明确诊断并回退到既有路径。
- 明确“回退粒度”为完整 `npux.compute_run`（或完整可独立执行子图），不能让同一
  `compute_run` 的部分 tile 走 descriptor、其余 tile 走 `npu_gemm_run`，除非后续
  单独实现并验证两种路径间的 DRAM 可见性和量化一致性。

**验收**

- lit 测试验证输入 IR 到 `NetworkSchedule` 的 tile 数、K 分段和 bank 分配。
- 选择一个静态 GEMM ONNX 模型，比较旧 tiling 结果和 schedule trace 的每个
  tile 形状。
- 错误 case 覆盖 K>4096、metadata 越界、非 32-byte 对齐和动态维度。

## 7. 阶段 3：网络级 DAG 与跨算子 bank 生命周期

**实现内容**

- 引入 `NetworkScheduleBuilder`，按 NPU 子图的拓扑顺序累积 TileSchedule，
  维护跨 tile 和跨算子的 bank valid/busy/last-use 状态。
- 支持 producer 输出直接被后继算子或 VPU 消费；只有需要 DRAM 可见或没有
  本地可复用路径时插入 MVOUT。
- 在选择 O bank 时同时考虑结果、accumulate、residual 和 post-process 的
  生命周期，禁止未完成命令上的读写冲突。
- 建模 bank 状态为 `Empty -> Loading -> Valid -> Reading/Writing -> Reusable`，状态
  转移由命令完成驱动；`lastUse` 只是选择候选的排序键，不能单独作为 bank 有效性的
  证明。为每个状态转移记录 producer、最后 consumer 和触发完成命令。
- 对资源限制形成可复现的退化策略：等待最早依赖、插入 MVOUT/reload，或
  按成本模型回退到 DRAM 路径。

**验收**

- 多层 GEMM 网络产生单一 DAG，命令 id 全局唯一，所有依赖均指向已存在命令。
- 自检器验证无 bank overlap、无循环依赖、所有 descriptor 范围合法。
- JSON trace 能显示每个 bank 的时间线和每次生命周期变更。

## 8. 阶段 4：Descriptor Lowering 与 runtime 合约

**实现内容**

- 新增 Versa-P 专用 lowering，将 `NetworkSchedule` 映射为 runtime 的
  64-bit descriptor 提交接口，而非调用历史 `npu_gemm_run`。
- runtime 按 ABI 先检查 `GENERIC_MAGIC/VERSION/MODE/CAPS`，再按“字段先写、
  start 最后写”提交命令。
- 将 lowering 分为两层：编译器只发射稳定的 `submit(desc0, desc1, desc2, api)`、
  `wait(api)`、`clear_status(api)` 和 `global_clear()` runtime 调用；寄存器偏移、
  轮询/IRQ、W1C 和 cache 同步由 runtime 封装。这样 LLVM IR 可检查依赖顺序，板端
  驱动也可在不改变 compiler ABI 的条件下切换轮询与 IRQ。
- error recovery 的最小语义为：停止提交后继命令，读取 error code，GLOBAL_CLEAR，
  标记所有本地 bank 无效；仅在完整输入仍在 DRAM 且上层选择重试时从头重新调度/加载。
  不允许在 clear 后假设任何 A/W/O/metadata bank 保留内容。
- 依赖边转为轮询 `STATUS.done/error` 或 IRQ 等待；错误时读取 error code，
  执行 GLOBAL_CLEAR，并让调用方重新加载失效 bank。
- 保留旧 C API 路径作为不支持算子、动态 shape 和未启用 Versa-P ABI 的回退。

**验收**

- LLVM IR/FileCheck 验证生成的 runtime 调用顺序、64-bit 参数和等待点。
- mock runtime 验证 command DAG 的提交顺序和 error recovery。
- 在 143 完成 `onnx-mlir-opt` 构建并运行新增 lit 测试。

## 9. 阶段 5：可观测性与用户交付

**实现内容**

- 新增编译选项，例如 `--npu-versa-p-schedule-trace=<path>`，默认关闭。
- 输出稳定 JSON：硬件 ABI 信息、子图、tile、command、descriptor、bank 读写、
  依赖、等待、fallback 原因和统计数据。
- 在中间 MLIR 上添加仅调试用途的调度属性，便于 `FileCheck` 与人工定位。
- 输出汇总报告：命令数、DMA 字节数、bank 等待次数、并行候选数、MVOUT/reload
  次数、回退节点和失败诊断。

**验收**

- 同一输入与相同编译选项生成字节稳定的 trace。
- JSON schema 校验通过，并能由脚本渲染 bank 时间线。
- 用户能够从一个 ONNX 模型命令得到 schedule JSON 和摘要报告。

## 10. 阶段 6：扩展算子与硬件闭环

**实现内容**

- VPU：将 SFU/GELU/Softmax/RMSNorm/LayerNorm 的 source/destination bank、
  precision、tail 和写回规则接入 DAG。
- GEMV：接入 W4A16/W8 的独立 API，与 SA_COMPUTE 建立互斥资源约束。
- QK/PV：接入 QK metadata、QK MVOUT logP correction、PV 归一化和 causal mask。
- ZCU102：完成 32-bit DMA 地址检查、cache 同步、IRQ/timeout、device tree 和
  descriptor-DMA-readback 数值回归。

**验收**

- 每个新 API 均有 descriptor golden、scheduler DAG、RTL 或 mock-runtime
  回归和明确 fallback 条件。
- 板端回归验证 descriptor 到 DRAM readback 的数值正确性和异常路径。

## 11. 143 服务器构建与验证流程

143 是共享 GPU 服务器，编译前必须先检查 `gpu01` 工作区、CPU/内存负载、
`cmake`/`ninja`/`clang` 与 conda 环境。编译不应占用 GPU，也不得终止其他用户
的进程。

### 11.1 已完成的预检（2026-08-07）

- 两跳 SSH 可达 `gpu01`；可见 7 块 RTX 2080 Ti，其中 GPU0..GPU6 空闲，GPU7
  的驱动句柄查询报错，编译任务不依赖该 GPU。
- 远程工作树为 `/home/lqma/npux-mlir`，已有 `build/CMakeCache.txt`，并可在
  `npux-mlir` conda 环境中使用 `/usr/bin/cmake` 与 ninja。
- 该工作树当前提交为 `7014dc15b48f1886ff041273e2ca01fcc6ba3149`，没有
  `VersaPBankScheduler.cpp`，且含其他未提交变更。因此不能直接用它验证本地
  调度器实现，也不能覆盖该工作树。

在阶段 0 编译前，必须从远程工作树创建独立 worktree，或在独立目录检出与本地
相同的提交并同步本功能涉及的文件；同步范围和目标目录须先明确，避免污染共享
工作树。之后再执行下面的构建命令。

建议命令如下，实际路径以服务器检查结果为准：

```bash
ssh -p 6801 lqma@10.134.142.143
ssh gpu01
cd <npux-mlir-worktree>
conda activate npux-mlir
source scripts/activate_env.sh
cmake --build build --target onnx-mlir-opt -j8
cmake --build build --target check-onnx-lit -j8
```

每一阶段只运行受影响的测试；阶段 4 及以后增加 Versa-P descriptor 和
schedule-trace lit 测试。完整 backend/数值/板端回归只在对应能力闭环后运行。

在已同步 Versa-P 源码并完成 configure 的隔离 worktree 中，G0 的最小验证命令为：

```bash
cmake --build build --target VersaPDescriptorTest -j8
ctest --test-dir build --output-on-failure -R '^VersaPDescriptorTest$'
```

第一条命令必须先成功，第二条必须报告一个通过的 CTest；缺少该 target、测试未发现或
只完成 configure 都不构成 G0 验收证据。

### 11.2 隔离 worktree 的已验证配置前置条件

2026-08-07 的实际检查表明，`gpu01` 的非交互 SSH 环境可使用 `/usr/bin/cmake`、
`/usr/bin/ninja` 和 `/home/lqma/anaconda3`，但不会自动提供 `conda` 命令。独立
worktree 也不会自动检出 submodule。因此在该机器上新建验证目录时，执行顺序必须为：

```bash
cd /home/lqma/npux-mlir
git worktree add --detach /home/lqma/npux-mlir-versa-p-<id> <commit>
cd /home/lqma/npux-mlir-versa-p-<id>
git submodule update --init --recursive
source /home/lqma/anaconda3/etc/profile.d/conda.sh
conda activate npux-mlir
source scripts/activate_env.sh
cmake -G Ninja -S . -B build \
  -DONNX_MLIR_CCACHE_BUILD=OFF \
  -DMLIR_DIR=/home/lqma/.local/npux-env/build/llvm-project/lib/cmake/mlir
```

上述 `ccache` 与 `MLIR_DIR` 选项来自该服务器的实际 CMake 诊断：`ccache` 未安装，
而项目虽在环境脚本中显示 `MLIR_DIR`，仍要求显式传入 CMake cache。路径可能随用户
环境变化；执行前应确认 `activate_env.sh` 的输出并更新 `-DMLIR_DIR`。submodule 初始化
可能耗时较长，完成前不得开始配置或构建。所有这些操作仅限独立 worktree，不能在共享
`/home/lqma/npux-mlir` 中执行。

该流程已在 `/home/lqma/npux-mlir-versa-p-g0` 成功完成 CMake configure；configure
阶段有 Java 禁用、Protobuf 与运行时 `libz` 搜索路径警告，但未阻止生成 Ninja 文件。
不要以 `cmake --build build --target NpuToLLVM` 代替计划中的 `onnx-mlir-opt`：前者在
此配置下不会先生成 `src/Dialect/Npux/NpuxDialect.hpp.inc`，会在 `NpuxOps.hpp` 编译处
失败。验证应构建 `onnx-mlir-opt`（或先显式构建其 dialect 生成依赖），并将完整日志保存
到 `artifacts/versa_p_scheduler/phase0-build.log`。SSH 前台命令受客户端时限影响；需要
长构建时应使用经确认的远端作业/日志机制，而不是把连接超时误记为编译失败或成功。

## 12. 最终阶段交付物

完成阶段 5 后，用户可得到以下结果：

1. 对静态 GEMM NPU 子图，编译器自动生成可执行的 Versa-P descriptor 命令 DAG。
2. 每个命令包含可提交的 64-bit descriptor、bank 分配和 `done/error` 依赖。
3. `schedule.json`：完整网络调度、bank 时间线、DMA/SA/MVOUT 统计和 fallback。
4. 生成的 LLVM IR/runtime 调用：可审查的 descriptor 提交与等待顺序。
5. lit/mock-runtime/143 构建日志：可复现的功能验证证据。

阶段 6 完成后，以上能力扩展到 VPU、GEMV 和 attention 路径，并以 ZCU102
descriptor-DMA-readback 数值回归作为最终硬件闭环证据。

## 13. 阶段门禁与优先级

实施顺序以可验证的最小闭环为准，禁止在前一阶段没有证据时并行宣称后续功能完成。

| 门禁 | 进入条件 | 退出条件 |
| --- | --- | --- |
| G0 合约冻结 | ABI 文档、RDL/RTL 版本已确认 | 哈希记录，descriptor 单测逐字段通过 |
| G1 单 tile | `TileRequest` 可取得静态地址和形状 | 生成完整 descriptor/DAG，mock runtime 完成 GEMM+MVOUT |
| G2 IR 接入 | G1 通过，资格检查可判定 | 一个静态 ONNX GEMM 经 lowering 产出稳定 trace；不合格 case 可回退 |
| G3 网络 DAG | G2 通过 | 多 tile/多 op 无 bank overlap、无环，trace 可复现 |
| G4 runtime 闭环 | G3 通过，runtime 接口存在 | LLVM FileCheck、mock error recovery、143 构建和受影响 lit 通过 |
| G5 用户交付 | G4 通过 | JSON schema、摘要报告和可复制的一键样例齐全 |

近期优先级为 G0 -> G1：先补齐 `ScheduledCommand` 的 descriptor/状态字段和单 tile
golden 测试，再把 scheduler 接入 `NpuxComputeRunLowering`。在 G1 前不得修改默认
lowering 行为，也不应开始网络级跨算子复用优化。

## 14. 文件级实施与测试矩阵

每项工作只能在对应测试先失败、实现后通过时标记完成。测试路径可因仓库既有 lit 布局
调整，但必须在实现提交中同时落地，且保持下表所列覆盖意图。

| 阶段 | 主修改位置 | 必须新增/更新的验证 | 完成判据 |
| --- | --- | --- | --- |
| G0 | `VersaPDescriptor.{hpp,cpp}`、`VersaPDescriptorTest.cpp`、`NpuToLLVM/CMakeLists.txt` | CTest `VersaPDescriptorTest`：MVIN_W 多行尾部 span、AUX metadata/O3 golden 与非法值 | encoder 输出与 ABI 表逐字段一致；非法输入返回 `llvm::Error`。 |
| G1 | `VersaPBankScheduler.{hpp,cpp}` | `test/.../VersaPBankSchedulerTest.cpp`：四种 output mode、bias/accumulate/resadd、O0..O3、metadata ping-pong、A/W 无伪依赖 | 每个逻辑命令具备完整 descriptor、bank 集合和 done/error 依赖；自检无冲突。 |
| G2 | `ConvertNpuxToLLVM.cpp` 及必要的 helper | `test/.../versa-p-gemm.mlir` FileCheck：资格通过、每种回退原因、稳定 schedule 属性/trace | 开关开启时仅合格完整 op 进入 Versa-P；legacy 输出保持不变。 |
| G3 | `NetworkScheduleBuilder`（新增）及 trace writer | 多 tile/多 op trace fixture 与 schema 验证 | id、依赖和 bank 时间线字节稳定；验证器拒绝环和非法生命周期。 |
| G4 | compiler runtime 声明、NPU runtime 实现、mock runtime | LLVM IR FileCheck、mock runtime success/error/clear 测试 | descriptor 写入遵守 start-last；等待的是 done/error；clear 后无 bank 复用。 |
| G5 | 命令行选项、JSON schema、文档样例 | 从 ONNX 到 JSON 的端到端 smoke test | 同输入/选项的 trace 可复现，摘要与 JSON 相互一致。 |

`test/...` 是 G1 及以后待选择的现有测试根路径占位符，并非可以省略的任务。G0 已在
`src/Conversion/NpuToLLVM/VersaPDescriptorTest.cpp` 通过根 CMake 的 CTest 接入；它
不是孤立测试，但在远端完整源码树构建并运行前仍不能宣称 G0 验收通过。实施 G1 时先
确定 scheduler 单测和 lit 的真实根路径，再用真实相对路径替换本表。

## 15. 首个实现切片（G0/G1）

首个可合入切片只处理一个静态、单一 `npux.compute_run` GEMM，并强制 `storeToDram`。
它必须覆盖 `MVIN_A -> MVIN_W -> [AUX_MVIN] -> SA_COMPUTE -> MVOUT`，但不承担
跨算子 bank 保留、VPU、GEMV、QK/PV 或板端 IRQ。切片的输入和输出定义如下：

- 输入为已解析的静态 `TileRequest`，其中 A/W/O 的 DRAM 地址、row stride、local
  base、M/N/K、输出模式和 metadata 使用情况都不可缺省。
- 输出为拓扑顺序确定的命令数组。每个数组元素包含 `EncodedDescriptor`、API kind、
  `DESC0/DESC1/DESC2` 写入值、start 写位置、状态寄存器、读写 bank 和前驱 id。
- MVIN_A 与 MVIN_W 可并行候选；SA 必须等待两者及可选 AUX_MVIN 完成；MVOUT 必须
  等待 SA 完成。bank 重用只能发生在消费者的 done 已被观察并清除之后。
- 任何 encoder、资格或自检失败都返回具名 fallback 原因，并保证没有 Versa-P descriptor
  已提交。该性质需要由 mock runtime 测试验证。

G1 的非目标是从 MLIR 解析地址，或把命令真正发给硬件。这样能先独立证明 descriptor
编码、DAG 和 bank 生命周期；G2/G4 分别负责 IR 提取和实际提交。

## 16. 风险、证据与停止条件

下表用于阶段评审。风险未关闭时，只能继续收集证据或实现隔离代码，不能将相应门禁
标记通过。

| 风险 | 触发信号 | 当前隔离措施 | 关闭所需证据 |
| --- | --- | --- | --- |
| ABI 漂移 | ABI 哈希、RDL 或 RTL 字段变化 | 冻结 SHA-256；encoder 仅按合约定义字段 | 更新后的字段逐位 golden、RDL/RTL 回归和新哈希记录。 |
| 虚假并行或 bank 冲突 | 命令缺失前驱，或无关命令被错误串行 | bank 读写集与独立 MVIN_A/MVIN_W engine 状态 | scheduler 单测证明 A/W 无伪依赖、GEMM 等待全部 producer、冲突访问有 done/error 前驱。 |
| 部分发射破坏语义 | 一个 `compute_run` 同时走 legacy 与 descriptor | G2 规定完整 op/子图粒度 fallback | FileCheck 与 mock runtime 证明失败前无 descriptor 提交，legacy 输出保持不变。 |
| 本地数据在 clear 后被误用 | error 后仍读取 A/W/O/metadata bank | runtime 设计要求 GLOBAL_CLEAR 后全部 bank 失效 | mock error recovery 证明后继提交停止，重试从 DRAM 全量 reload。 |
| 测试基础设施不可用 | `test/*` 未检出或未接入构建 | 不把源码审读视为 G0/G1 通过 | 恢复真实测试根、CMake/lit 接入和可重复的测试日志。 |
| 远端验证不可复现 | 共享树脏、submodule 未锁定、前台 SSH 超时 | detached worktree、固定 conda/MLIR_DIR、独立日志 | worktree commit/submodule status、完整 `onnx-mlir-opt` 日志和产物哈希。 |
| 本地源码未进入远端验证树 | gpu01 基线缺 Versa-P 文件 | 禁止覆盖共享树；仅允许隔离 worktree | 获得明确同步授权后，记录文件清单、校验和和远端 `git diff`。 |

当出现 ABI 不匹配、scheduler 自检失败、runtime error 或任何未分类 fallback 时，应停止
继续提交该 schedule，并在 trace 中记录规则 id、命令 id（若已生成）和原始 tile 信息。
仅资源不足导致的等待、MVOUT/reload 或完整 op 的 legacy fallback 可以继续编译；其余
错误必须先修复或得到新的 ABI 合约。

## 17. G0 descriptor Golden 向量

以下向量来自冻结 ABI，可直接作为 `VersaPDescriptorTest` 的最小位级用例。所有未列出
的 descriptor word 必须为零；测试应同时确认输入 `config` 未被 encoder 修改。

| 用例 | 输入 | 期望 `DESC0` | 期望 `DESC1` |
| --- | --- | --- | --- |
| MVIN_W 两行尾部 | `dramBase=0x3000`，`dramSpanBytes=0x80`，`rowCount=2`，`colCount=33`，`bank=1` | `0x0000008000003000` | `0x0000000300210002` |
| AUX metadata | `dramBase=0x1000`，`offsetBytes=0x20`，`value=0x40`，`auxTypeOrOBank=0`，`destinationIsOBank=false`，`metadataBank=1` | `0x0000002000001000` | `0x0000001400000040` |
| AUX O3 | `dramBase=0x2000`，`offsetBytes=0x80`，`rowCount=2`，`rowBytes=64`（`value=0x00400002`），`auxTypeOrOBank=3`，`destinationIsOBank=true` | `0x0000008000002000` | `0x0000000F00400002` |

`MVIN_W 两行尾部` 验证连续 W 传输的最小 span 是
`rowCount * ceil(colCount / 32) * 32`；本例为 `2 * 2 * 32 = 128` byte，`96` byte
必须被拒绝。`AUX metadata` 用例验证 `start[34]`、`metadata_bank[36]` 和 byte offset；
`AUX O3` 用例验证 `o_bank[33:32]`、`start[34]`、`dest_o_bank[35]`、打包的 row
count/row bytes 以及 O0..O3 的上界。三者都应返回 `desc2 == 0`。

非法输入至少包含：未对齐 `dramBase`、`auxTypeOrOBank=4`、metadata `value=0`、
metadata `offsetBytes + value > 16384`、O 目的地的零 row count/row bytes、O stride
小于 row bytes 或非 32-byte 对齐。每项都必须返回 `llvm::Error`，且错误文本包含
`AUX_MVIN`，以便 trace/fallback 保留可定位原因。

### 17.1 已执行证据（2026-08-07）

在 `gpu01` 隔离 worktree `/home/lqma/npux-mlir-versa-p-g0` 中，使用
`npux-mlir` conda 环境、Ninja 和显式 `MLIR_DIR` 完成了以下实际验证：

```text
cmake --build build --target VersaPDescriptorTest -j8   # passed
ctest --test-dir build --output-on-failure -R VersaPDescriptorTest
# 1/1 Test #1: VersaPDescriptorTest ... Passed
```

日志保存在该 worktree 的
`artifacts/versa_p_scheduler/phase0-versap-descriptor-test.log`。这证明当前
MVIN_W 两行尾部 span、AUX metadata/O3 golden 和列出的 AUX 非法输入可构建且通过；
它**不**证明 MVIN_A、MVOUT、GEMM 的逐字段测试已完成，也不替代
`onnx-mlir-opt`、lit 或 runtime 验证。因此 G0 阶段门禁仍保持未通过。

同一 worktree 已完成
`cmake --build build --target onnx-mlir-opt -j8` 的 539 个构建步骤，生成
`build/Debug/bin/onnx-mlir-opt`（711,867,336 byte），并在激活的 conda 环境中成功
执行 `--version`，报告 LLVM `22.0.0git`。完整构建日志为
`artifacts/versa_p_scheduler/phase0-onnx-mlir-opt.log`。这关闭了 G0 的“可编译基线”
子项，但没有扩大 descriptor 测试覆盖范围。

本次 143 构建使用 `7014dc15` 的 detached 基线加上表中列出的 Versa-P/CMake 文件；
为保护共享工作树和避免扩大同步范围，**没有**同步本地其他未提交文件。因此该日志证明
这组 Versa-P 修改能与该基线构建，不能替代对完整本地 worktree 或未同步改动的构建证明。

构建前已逐项核对本地与隔离 worktree 的 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `VersaPDescriptor.cpp` | `16564AE1545FBA6994A2C4ECBB27BADB4413468394DD7F6D48FFA90540D107AA` |
| `VersaPDescriptor.hpp` | `42F9CA47DD7F36FD7779941DE066FC26390FA43BB446E6950A49A7E884966A51` |
| `VersaPDescriptorTest.cpp` | `BAE20A6991978F9A0AC1BF2E86D6BF904DC9E48C63A2F1D1349222DE8048D29A` |
| `NpuToLLVM/CMakeLists.txt` | `5EDF6F1376AA5CB9DB936EB7BA89BB2AD9FB34ED58D7E5CB7FB09E6982D81F5A` |

## 18. 2026-08-08 G1 单 tile 实测证据

- 在 `gpu01` 的隔离 worktree `/home/lqma/npux-mlir-versa-p-g0` 中，ABI 合约
  `NPUHardwareSpecificationTemplate_CN.md` 已按二进制 SHA-256 校验为
  `BDABEF06F134EDAF27F3A8F79CB8B4E7A11025B3E9D4341C772F22715E7FF4BE`。
- `VersaPBankScheduler` 现在要求每个 G1 tile 显式提供 A/W/MVOUT 的静态 DMA 信息和
  SA local-word base；输出的每条 `ScheduledCommand` 带有编码后的 `DESC0/1/2`、提交
  descriptor 寄存器偏移、`STATUS` 偏移以及只表示 `done && !error` 的前驱 id。
- 新增 CTest `VersaPBankSchedulerTest` 覆盖 A/W 独立 load、GEMM/MVOUT 的完成依赖、
  raw INT32/tensor INT8/per-channel INT8/FP32、bias、accumulate、resadd、metadata
  ping-pong、O0..O3 选择和未知 DMA 的拒绝。远端执行：

  ```text
  cmake --build build --target VersaPBankSchedulerTest -j8
  ctest --test-dir build --output-on-failure -R '^VersaP(BankScheduler|Descriptor)Test$'
  # 2/2 Passed
  ```

  日志：`artifacts/versa_p_scheduler/phase1-versap-bank-scheduler-matrix.log`。
- 这不是 G1 门禁关闭证据：尚未有 mock runtime 验证 descriptor 提交、`done/error`、W1C
  清除以及失败后无提交；`NpuxComputeRunLowering` 仍未调用 scheduler，默认 legacy C API
  路径保持不变。
- 2026-08-08 G1 mock runtime：VersaPBankSchedulerTest 已验证 done/error、W1C、停止后续提交和 GLOBAL_CLEAR；日志为 rtifacts/versa_p_scheduler/phase1-versap-mock-runtime.log。真实 runtime 与 lowering 仍未接入。

### 18.1 G2 DMA 区域与运行时地址 patch 约束（2026-08-08）

`npux.compute_run` 的 buffer operand 只保存 NPU SRAM/ACC local offset；descriptor 所需的
DRAM base 来自同一 block 的 `npux.dma_mvin`/`npux.dma_mvout` host memref，并且是运行时
地址。因此 G2 不得将 local offset 或编译期零值写入 `dram_base`。`VersaPDmaRegion` 只识别
唯一、顺序正确的 `DMA(A) -> DMA(W) -> GEMM -> MVOUT` SSA region；缺失或歧义必须以
`incomplete-subgraph` fallback 回到完整 legacy 子图。

G4 runtime 接口必须支持 `runtime-address-patch-required`：编译器保存 descriptor 的静态字段与
动态地址字段位置，runtime 在每次提交前从 memref base/byte offset 填充 DRAM base，随后遵守
`DESC0/DESC1/DESC2` 先写、start 最后写和 `done/error` 等待语义。没有该接口时，G2 不得
发射部分 descriptor 或替换 legacy DMA/compute 调用。
### 18.2 G2/G4 运行时地址翻译契约（2026-08-08）

npux.dma_mvin/npux.dma_mvout 的 host memref 在 LLVM lowering 后是虚拟指针，而
Versa-P DESC0[31:0] 需要的是 32 位 DMA/物理地址。因此 descriptor lowering 必须先调用
npu_versa_p_translate_dram_address(const void *host_ptr, uint32_t *dram_base)。

不得将 host 指针直接截断为 uint32_t。板端 runtime 必须在此接口完成 DMA/IOMMU 映射；
当前 mock 仅为 VersaPRuntimeTest 提供低 32 位的可观察模拟，不能作为硬件地址转换实现。
运行时翻译失败属于编译后事件：已选择 descriptor 的完整区域必须 GLOBAL_CLEAR 并 fail-closed
中止，不能继续提交后续 descriptor；编译期无法证明静态资格的区域才保留完整 legacy 路径。该接口与空指针拒绝行为已在 gpu01 隔离 worktree 的 VersaPRuntimeTest 中验证通过。
### 18.3 G2/G4 已接入的安全 lowering（2026-08-08）

`npux-versa-p-region-mark` 现在只对完整且静态合格的
`DMA(A) -> DMA(W) -> GEMM -> MVOUT` 写入同一 region id、角色、API 和
`DESC0/1/2` 属性。`ConvertNpuxToLLVM` 只在这四类属性齐全时替换 legacy 调用：

- A/W/MVOUT 调用 `npu_versa_p_submit_wait_host_or_abort`；运行时先将 host 指针映射为
  DMA 地址，再只 patch DMA descriptor 的 `DESC0[31:0]`。
- GEMM/SA 调用 `npu_versa_p_submit_wait_static_or_abort`；其 `DESC0` 原样提交，不能被
  DMA base 覆盖，因为其中编码的是 M/N/K。
- `submit_wait_*` 的每次提交均等待 completion、成功后 W1C 清状态；失败时先
  `GLOBAL_CLEAR`，随后 fail-closed 中止，保证同一已替换区域不会继续执行并使用陈旧 local
  bank 状态。未标记、动态或不合格区域仍保持完整 legacy lowering，绝不产生混合子图。

`GemmDmaRegion` 也已显式携带 `ComputeRunOp`，资格构建器不再依赖 block 邻接扫描来反查
GEMM。2026-08-08 在 `gpu01` 隔离 worktree
`/home/lqma/npux-mlir-versa-p-g0` 上重新构建 `onnx-mlir-opt`，并执行：

```text
ctest --test-dir build --output-on-failure -R VersaP.*Test
# VersaPDescriptorTest, VersaPBankSchedulerTest, VersaPRuntimeTest: 3/3 Passed
```

`VersaPRuntimeTest` 额外验证 SA 的 `0x1122334455667788` 原样保留，覆盖了此前
`submit_wait_static` 错把 DESC0 低 32 位当作 DMA base 的风险。该证据证明 C++ 编译、descriptor
编码、scheduler 与 mock runtime 合约；仍需以真实 NPU MLIR fixture/lit 和 ZCU102 板端 DMA
映射实现来关闭 G2/G4 全门禁。