# 模型测试流程
本文件旨在说明 model_test 目录下的文件结构及其具体含义，并规范从模型准备到生成硬件可执行文件的端到端测试流程。

目标补充：当前算子级（Unit Test）测试不是终点，而是为后续模型级（Integration Test）打基础。
因此，单算子测试所使用的输入形状（shape）应尽量贴近真实模型中的常见尺寸分布，而不是仅使用过小的 toy shape。

# 1. 目录结构
建议按以下层次组织文件，确保每个测试模型享有独立的工作空间：

```Plaintext
model_test/
├── makefile                     # 自动化构建脚本，一键完成从模型处理到交叉编译
├── scripts/                     # 核心编译脚本目录
│   ├── onnx_to_llvm.sh          # ONNX-MLIR 编译至 LLVM IR
│   ├── conv_tile.py             # 对模型进行卷积分块并生成 tile_config.json
│   └── compile_to_zcu102.sh     # LLVM IR 交叉编译至 ZCU102 可执行文件
├── runtime/                     # ZCU102 运行时源码与头文件
│   ├── npu_runtime.cpp
│   ├── npu_runtime.h
│   ├── npu_regs.h
│   └── npu_regs_compat.h
├── models/                      # 测试模型资源池
│   └── [model_name]/            # 每个模型一个独立目录
│       ├── model.py / model.onnx
│       └── main.cpp             # 模型宿主程序（必需）
└── build/                       # 编译工作区（make 时自动生成）
    └── [model_name]/
        ├── model.onnx
                                ├── model.onnx.mlir
        ├── tile_config.json
        ├── NpuToLLVM/llvm.mlir
        ├── Codegen/model.o
        ├── Linking/
        └── output/
            ├── [model_name]_zcu102
            ├── [model_name]_input.bin
            └── [model_name]_output_golden.bin
```
# 2. 编译流程
以下为标准的模型转换与编译链路。各步骤的输入输出需严格对应：

模型准备: 使用 PyTorch 生成所需的测试模型，或直接下载预训练的 .onnx 模型文件。

尺寸约束（新增）：
- 单算子测试的 shape 需来自真实模型常见维度（如 CNN 的 `NCHW`、Transformer 的 `batch/seq/hidden` 组合）。
- 选择 shape 时应优先考虑后续要验证的目标模型（如 VGG/ResNet/BERT）中的典型尺寸。
- 禁止仅使用“方便调试但不具代表性”的极小尺寸作为唯一测试样本。

量化与校准: 对模型进行 INT8 对称量化（当前阶段暂不进行严格校准）。

策略探索 (可选): 运行 conv_tile.py 脚本，针对模型探索最优的卷积分块策略，并生成 tile_config.json。（若模型中无卷积算子，可忽略此步）。

MLIR 转换: 将 ONNX 模型降级为 ONNX Dialect 的 MLIR 文件。

执行命令: onnx-mlir --EmitONNXIR [模型文件.onnx]

LLVM IR 编译: 使用 scripts/onnx_to_llvm.sh，将上一步生成的 MLIR 文件进一步编译（Lowering）为 LLVM IR。

交叉编译: 使用 scripts/compile_to_zcu102.sh，结合 runtime 运行库，将 LLVM IR 编译成可在 ZCU102 上运行的可执行文件。

说明：
- `compile_to_zcu102.sh` 默认输入为当前工作目录下 `NpuToLLVM/llvm.mlir`。
- 第二个参数可指定输出文件名；未指定时默认输出 `[model_name]_zcu102`。
- `npu_runtime.cpp` 优先从模型工作目录读取；若不存在则自动回退到 `model_test/runtime/npu_runtime.cpp`。
- `LLVM_SRC_ROOT`、`RUNTIME_LIB_DIR`、`CROSS_CXX` 等路径/工具均可通过环境变量覆盖。
- 默认后端为 ZCU102，可直接使用 `make [model_name]` 一键跑完整链路。
- 运行时数据命名统一为模型名前缀：`[model_name]_input.bin` 与 `[model_name]_output_golden.bin`。

执行边界（新增）：
- AI 完成代码/脚本生成后，不自动执行 `make` 或其他构建命令。
- 构建与运行步骤由开发者手动触发，AI 仅负责产物准备与说明。

验证输出要求（新增）：
- `main.cpp` 在结果比对时应打印错误对比明细（至少包含 index、actual、golden、abs diff）。
- 小张量建议全量打印；大张量建议打印误差最大的 Top-K 样本。
- 需要统计错误数据的数量，格式（错误点数量/总点数量）
- 测试日志最后一行必须使用统一且固定的脚本识别格式输出：`@@MODEL_TEST_RESULT@@ errors=<错误点数量>/<总点数量> status=<PASS|FAIL>`
- 该统一格式必须作为程序最后输出；在该行之后禁止再打印其他日志。
- 被测试的目标算子在量化图中必须满足 `DequantizeLinear -> 目标算子 -> QuantizeLinear` 的直接相邻模式，供 NPU 编译器识别并替换为 NPU 算子。
- 禁止使用仅有 `QuantizeLinear -> 目标算子 -> DequantizeLinear` 的形式作为最终测试图，这种形式不满足当前 NPU pattern 识别要求。
- 建议在生成 `model.onnx.mlir` 后，检查目标算子附近的 MLIR，确认存在 `onnx.DequantizeLinear -> onnx.[Op] -> onnx.QuantizeLinear`。
- 鼓励按算子增加定制化验证：例如 Softmax 额外关注 Top-K 一致性与概率和约束。

# 3. 测试进度与目标
3.1 算子级测试 (Unit Tests)
目前支持并正在测试的算子包括（x代表已经测试）：

- [ ] Conv
- [x] MaxPool
<!-- - [] AveragePool硬件不支持 -->
- [ ] MatMul
- [ ] Gemm
- [x] Transpose
- [ ] Relu（和卷积放在一起测试）
- [x] Gelu
- [x] Softmax(精度较低Top-1 match: 89 / 128)
- [x] LayerNorm（精度较低，错误：2061/98304）
<!-- - [ ] Resize硬件不支持 -->

3.2 模型级测试 (Integration Tests)
简单模型验证: MNIST

目标模型验证: VGG, ResNet, BERT, YOLO