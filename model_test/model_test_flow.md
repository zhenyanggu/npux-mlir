# 通用模型端到端测试流程规范

本文件旨在说明 `model_test` 目录下的文件结构及其具体含义，并规范从模型准备到生成硬件可执行文件的端到端测试流程。

**核心导向**：算子级测试（Unit Test）是为了验证局部算子的可识别与正确性，其输入 shape 应贴近真实模型分布；模型级测试（Integration Test）是最终目的，用于验证整网端到端任务的正确性与精度指标。

## 1. 目录结构

按以下层次组织文件，确保每个测试模型（或单算子网络）享有独立的工作空间：

```text
model_test/
├── makefile                     # 自动化构建脚本，一键完成从模型处理到交叉编译
├── scripts/                     # 核心编译脚本目录
│   ├── onnx_to_llvm.sh          # ONNX-MLIR 编译至 LLVM IR
│   ├── conv_tile.py             # 探索并生成卷积分块策略 tile_config.json
│   └── compile_to_zcu102.sh     # LLVM IR 交叉编译至 ZCU102 可执行文件
├── runtime/                     # ZCU102 运行时源码与头文件
│   ├── npu_runtime.cpp / .h
│   └── npu_regs.h / _compat.h
├── models/                      # 测试模型资源池（每个模型独立目录）
│   └── [model_name]/            
│       ├── model.py / .onnx     # 模型生成脚本或原始权重
│       └── main.cpp             # 模型宿主测试程序（必需）
├── build/                       # 编译工作区（make 时自动生成）
│   ├── NPU_[model_name]/        # NPU 独立中间产物目录
│   │   ├── model.onnx / model.onnx.mlir
│   │   ├── tile_config.json
│   │   ├── RecomposeONNX/RecomposeONNX.mlir
│   │   ├── NpuPartition/ConvertONNXToLinalgNpu.mlir
│   │   ├── NpuToLLVM/llvm.mlir
│   │   ├── Codegen/model.ll
│   │   └── Codegen/model.o
│   └── CPU_[model_name]/        # CPU 独立中间产物目录
│       ├── model.onnx / model.onnx.mlir
│       ├── tile_config.json
│       ├── RecomposeONNX/RecomposeONNX.mlir
│       ├── CpuToLLVM/llvm.mlir
│       ├── Codegen/model.ll
│       └── Codegen/model.o
└── output/                      # 最终可执行与测试输入输出
    ├── NPU/
    │   └── [model_name]/
    │       ├── [model_name]_zcu102
    │       ├── [model_name]_input.bin
    │       └── [model_name]_output_golden.bin
    └── CPU/
        └── [model_name]/
            ├── [model_name]_zcu102
            ├── [model_name]_input.bin
            └── [model_name]_output_golden.bin

```

### 1.1 Docker 约定

- 当前推荐编译环境在 docker 容器 `my-npux-dev` 中。
- 仓库根目录在容器内挂载为 `/workspace`。
- 所有文档中的命令优先写成下面这种形式，便于直接复制执行：
  - `docker exec my-npux-dev /bin/bash -lc 'cd /workspace && ...'`

## 2. 编译与转换链路

以下为标准的模型转换与编译链路。各步骤的输入输出需严格对应：

1. **模型准备**：生成或下载预训练的 `.onnx` 模型。输入 shape 需符合真实场景（如 CNN 的 NCHW）。
2. **量化与校准**：对模型进行 INT8 对称量化。插入 QDQ（`QuantizeLinear` / `DequantizeLinear`）节点。
3. **策略探索 (可选)**：运行 `conv_tile.py`，生成 `tile_config.json`（无卷积算子可跳过）。
4. **MLIR 转换**：使用 `onnx-mlir --EmitONNXIR` 将 ONNX 降级为 ONNX Dialect 的 MLIR。
5. **高层模式重组**：通过 `onnx_to_llvm.sh` 中的 `RecomposeONNX` 阶段，把已展开的高层模式重新识别成 `onnx.Gelu` / `LayerNormalization` 等。
6. **LLVM IR 编译**：通过 `onnx_to_llvm.sh` 将 MLIR 进一步 Lowering 为 LLVM IR。
7. **交叉编译**：通过 `compile_to_zcu102.sh` 生成 `Codegen/model.ll`、目标文件和 ZCU102 可执行文件。

### 2.1 docker 中的标准命令

1. `docker exec my-npux-dev /bin/bash -lc 'cd /workspace/model_test && make -B llvm MODEL=bert_base'`
2. `docker exec my-npux-dev /bin/bash -lc 'cd /workspace/model_test && make -B zcu102 MODEL=bert_base'`
3. `docker exec my-npux-dev /bin/bash -lc 'cd /workspace && grep -c "library_call = \"npu_gelu\"" model_test/build/NPU_bert_base/NpuPartition/ConvertONNXToLinalgNpu.mlir'`
4. `docker exec my-npux-dev /bin/bash -lc 'cd /workspace && grep -c "@npu_sfu_run(i8 1," model_test/build/NPU_bert_base/Codegen/model.ll'`

### 2.2 执行边界与验证规范

* **自动化边界**：代码/脚本生成后，构建与运行步骤（如 `make`）由开发者手动触发。
* **图结构要求**：默认要求目标算子满足 `DequantizeLinear -> 目标算子 -> QuantizeLinear` 的相邻模式。
* **当前 GELU 特例**：对 unary `Gelu`，NPU partition 额外支持 `DequantizeLinear -> Reshape? -> Gelu -> Reshape? -> QuantizeLinear`。
* **当前未覆盖场景**：若 `Gelu` 后面直接接浮点算子而没有输出侧 `QuantizeLinear`，则仍会保留在 CPU/浮点路径。
* **日志输出底线**：`main.cpp` 在结束前，**最后一行必须**输出以下固定格式，且其后禁止打印任何内容：
> `@@MODEL_TEST_RESULT@@ errors=<错误点数量或误分类数>/<总点数量或总样本数> status=<PASS|FAIL>`

### 2.3 bert_base 的 GELU 验收口径

- 不要只检查 `model.onnx.mlir`。`bert_base` 的源 ONNX 本身就可能把 GELU 展开成 `Div -> Erf -> Add -> Mul -> Mul(0.5)`。
- 首先检查：
  - `model_test/build/NPU_bert_base/RecomposeONNX/RecomposeONNX.mlir`
  - 这里应能看到重组后的 `onnx.Gelu`。
- 然后检查：
  - `model_test/build/NPU_bert_base/NpuPartition/ConvertONNXToLinalgNpu.mlir`
  - 当前应有 12 个 `library_call = "npu_gelu"`。
- 最终检查：
  - `model_test/build/NPU_bert_base/Codegen/model.ll`
  - 当前应有 24 个 `@npu_sfu_run(i8 1, ...)`，对应 12 个 encoder GELU 在 tiling 后拆成两次 SFU 调用。
- 说明：
  - `cls/predictions/transform` 的 GELU 目前仍保留为浮点 `onnx.Gelu`，因为它后面直接接 `LayerNormalization`，不满足当前 unary GELU 的输出侧量化约束。


## 3. 测试进度与层级目标

### 3.1 算子级测试 (Unit Tests)

**目标**：验证单个或局部算子在 NPU 上的可识别性、替换率及数值准确性（逐元素误差比对）。

* **已支持与验证算子**：Conv, MaxPool, MatMul, Gemm, Add(MatAdd), Transpose, Relu, Gelu, Softmax, LayerNorm。
* **验证逻辑**：比对张量级差异。小张量全量打印，大张量打印误差最大的 Top-K。

### 3.2 模型级测试通用规范 (Integration Tests)

**目标**：以完整模型（如 VGG, ResNet, BERT, YOLO 等）为测试对象，验证整网端到端的任务指标（如分类准确率、mAP 等）。

| 规范维度 | 具体要求 |
| --- | --- |
| **数据与预处理** | 必须使用真实的验证集/测试集数据（禁止全随机张量）。<br>

<br>输入 shape 需与真实推理一致。<br>

<br>`main.cpp` 中的预处理逻辑（如归一化、类型转换）必须与 Golden 端严格对齐。 |
| **量化与校准** | 必须执行静态量化（推荐 INT8 对称量化，QDQ 格式）。<br>

<br>校准数据**必须来自训练集**，校准样本数建议 $\ge 64$。 |
| **指标与验收** | 核心验收标准为任务级指标（如 Top-1 Accuracy）。<br>

<br>需约定精度退化阈值（例如：相比 FP32 Baseline 下降 $\le 1.0\%$）。<br>

<br>禁止以“单样本跑通”作为最终结论，建议评测规模 $\ge 1000$ 样本。 |
| **编译回退检查** | 必须记录 NPU 替换覆盖率。若关键骨干算子大面积回退到 CPU，即使精度达标也应标记为“风险/FAIL”。 |
| **日志与明细** | `main.cpp` 必须打印：总样本数、正确数、核心指标（Accuracy/mAP 等）、误分类样本明细（Index, Pred, Golden）。 |

### 3.3 模型级测试交付清单 (Minimum Deliverables)

每个进入模型级测试的 Network 应当包含以下标准产物：

1. `model.onnx`（量化后的标准图）。
2. `[model_name]_input_data.bin`（如原始图像二进制或文本张量）。
3. `[model_name]_output_golden.bin`（预期输出）。
4. 评测标签文件及数据预处理脚本。
5. 符合日志规范的 `main.cpp` 测试源码。

---

针对这份精炼后的通用流程，您是否需要我为您调整 `makefile` 的核心逻辑，使其能够通过传递参数（如 `make test MODEL=resnet`）来兼容这种标准化的目录结构和链路？
