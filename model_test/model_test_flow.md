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
└── build/                       # 编译工作区（make 时自动生成）
    └── [model_name]/
        ├── model.onnx / model.onnx.mlir
        ├── tile_config.json
        ├── NpuToLLVM/llvm.mlir
        ├── Codegen/model.o
        └── output/
            ├── [model_name]_zcu102
            ├── [model_name]_input.bin
            └── [model_name]_output_golden.bin

```

## 2. 编译与转换链路

以下为标准的模型转换与编译链路。各步骤的输入输出需严格对应：

1. **模型准备**：生成或下载预训练的 `.onnx` 模型。输入 shape 需符合真实场景（如 CNN 的 NCHW）。
2. **量化与校准**：对模型进行 INT8 对称量化。插入 QDQ（`QuantizeLinear` / `DequantizeLinear`）节点。
3. **策略探索 (可选)**：运行 `conv_tile.py`，生成 `tile_config.json`（无卷积算子可跳过）。
4. **MLIR 转换**：使用 `onnx-mlir --EmitONNXIR` 将 ONNX 降级为 ONNX Dialect 的 MLIR。
5. **LLVM IR 编译**：通过 `onnx_to_llvm.sh` 将 MLIR 进一步 Lowering 为 LLVM IR。
6. **交叉编译**：通过 `compile_to_zcu102.sh` 链接 runtime，生成目标硬件（ZCU102）可执行文件。

### 2.1 执行边界与验证规范

* **自动化边界**：代码/脚本生成后，构建与运行步骤（如 `make`）由开发者手动触发。
* **图结构要求**：被测试的目标算子必须满足 `DequantizeLinear -> 目标算子 -> QuantizeLinear` 的相邻模式，严禁仅有 `Quantize -> 目标 -> Dequantize` 的反向测试图。
* **日志输出底线**：`main.cpp` 在结束前，**最后一行必须**输出以下固定格式，且其后禁止打印任何内容：
> `@@MODEL_TEST_RESULT@@ errors=<错误点数量或误分类数>/<总点数量或总样本数> status=<PASS|FAIL>`



## 3. 测试进度与层级目标

### 3.1 算子级测试 (Unit Tests)

**目标**：验证单个或局部算子在 NPU 上的可识别性、替换率及数值准确性（逐元素误差比对）。

* **已支持与验证算子**：Conv, MaxPool, MatMul, Gemm, Transpose, Relu, Gelu, Softmax, LayerNorm。
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