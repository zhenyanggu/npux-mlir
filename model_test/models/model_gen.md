# AI 算子级模型生成与 main.cpp 规范（以 Gelu 为例）

本文档用于指导其他 AI 在 `model_test` 流程中生成**可编译、可验证、可复现**的算子级测试模型与配套 `main.cpp`。  
目标是避免“单算子被优化掉、无法识别”的常见问题，并统一输入输出对齐规则。
更重要的是：算子级测试需要为后续整模型验证打基础，因此 shape 选型必须贴近真实模型的常见尺寸。

---

## 1. 适用范围

- 目录：`model_test/models/[model_name]/`
- 场景：算子级测试（Unit Test），如 `Gelu`、`Relu`、`LayerNorm` 等
- 产物：
  - 模型文件（`.onnx`）
  - 测试输入（二进制）
  - golden 输出（二进制）
  - 宿主程序 `main.cpp`

---

## 2. 硬性要求（必须遵守）

### 2.1 不允许“裸算子”模型

不要只保留单个目标算子（如仅 `Gelu(x)`），否则在导出/优化阶段可能被融合或消除，导致后端识别不到目标算子。

### 2.2 必须使用“包裹结构”保护目标算子

推荐最小结构：

1. 前置轻量算子（如 `Add`）
2. 目标算子（如 `Gelu`）
3. 后置轻量算子（如 `Sub`）

即：`x -> Add -> Gelu -> Sub -> y`

### 2.3 输入输出契约必须固定

- 输入 shape、dtype 在模型生成和 `main.cpp` 中必须完全一致
- 输入/输出二进制文件名在生成端与消费端必须一致
- 二进制文件建议统一使用模型名前缀：`[model_name]_input.bin`、`[model_name]_output_golden.bin`
- 最终运行产物统一放在 `build/[model_name]/output/`
- 建议固定随机种子，保证可复现

### 2.4 量化策略要显式声明

- 若仅测试目标算子量化，必须指定 `op_types_to_quantize=[目标算子]`
- 对称量化场景需显式设置：
  - `ActivationSymmetric=True`
  - `WeightSymmetric=True`
- 建议保留 zero point 检查（INT8 应为 0）
- 最终用于编译验证的目标算子必须落在 `DequantizeLinear -> [OP_NAME] -> QuantizeLinear` 模式中。
- 若生成结果是 `QuantizeLinear -> [OP_NAME] -> DequantizeLinear`、`QLinear[OP_NAME]` 或其他不满足该模式的形式，则该测试视为不合规，需要继续调整模型生成方式。
- 建议在 `model.onnx.mlir` 中人工或脚本检查目标算子附近 IR，确认 pattern 成立后再交付。

### 2.5 测试尺寸必须贴近真实模型

- 单算子测试不得只使用“教学用小尺寸”作为唯一样本。
- shape 需来自目标模型族的常见配置，例如：
    - CNN 场景：`[N, C, H, W]` 中 `C/H/W` 参考常见骨干网络阶段尺寸。
    - Transformer 场景：`[batch, seq, hidden]` 中 `hidden` 建议使用 `384/512/768/1024` 等常见值。
- 若为了调试保留小尺寸样本，必须额外提供至少一个“真实尺寸样本”。

### 2.6 交付后不自动执行构建

- AI 在完成代码、模型与文档生成后，不执行 `make` 或其他构建/运行命令。
- 仅输出“建议执行命令”和检查要点，实际执行由开发者手动完成。

---

## 3. Gelu 参考拓扑（推荐）

以输入 `x` 为例：

1. `x = x + 0.001`（前置包裹）
2. `x = Gelu(x)`（目标算子）
3. `x = x - 0.001`（后置包裹）

说明：

- 前后算子尽量简单、稳定，不引入额外复杂语义
- 常量偏移值可小幅设置（如 `1e-3`），避免改变整体数值分布过大

---

## 4. 模型与数据文件命名规范

建议在每个模型目录下使用如下命名：

- `model.onnx` 
- `[model_name]_input.bin` 
- `[model_name]_output_golden.bin` 
- `main.cpp`

最终归档目录：`build/[model_name]/output/`。

关键点：命名可自定义，但必须在整个链路中一致。为避免多模型混放冲突，推荐固定采用模型名前缀。

---

## 5. main.cpp 编写规范（必须覆盖）

`main.cpp` 的职责是：读取输入 -> 调用编译模型 -> 读取 golden -> 比对结果。

### 5.1 输入读取

- 以二进制方式读取 `float32` 数据
- 按“元素个数 × 4 字节”校验文件大小
- 输入 shape 与模型导出一致（例如 `[1, 768]`）
- 建议从“可执行文件所在目录”解析输入路径，避免依赖当前工作目录
- 建议优先读取带前缀命名，兼容回退到旧命名（`input.bin` / `output_golden.bin`）

### 5.2 构造运行时输入

- 使用 ONNX-MLIR Runtime 的 `OMTensor` / `OMTensorList`
- 输入数据类型与模型入口对齐（常见为 `ONNX_TYPE_FLOAT`）

### 5.3 执行模型

- 调用编译后导出的入口函数（如 `run_main_graph`）
- 判断返回值是否为空，空指针直接报错退出

### 5.4 输出处理

- 从输出 `OMTensor` 获取数据指针与 shape
- 计算输出总元素个数，用于读取 golden 文件

### 5.5 结果比对

- 计算至少两项指标：
  - `Max Absolute Error`
  - `Mean Squared Error`
- 必须打印“误差对比明细”：至少包含 index、actual、golden、abs diff
- 小张量建议全打印；大张量至少打印误差最大的 Top-K 项（建议 K=10）
- 对量化模型使用合理阈值（例如 `max_diff < 0.1`）
- 日志最后一行必须打印统一脚本识别格式：`@@MODEL_TEST_RESULT@@ errors=<错误点数量>/<总点数量> status=<PASS|FAIL>`
- `PASS/FAIL` 必须与同一阈值下的错误点统计保持一致，且该行之后不得再输出任何内容

### 5.6 可选的算子定制化验证（推荐）

- 在通用误差指标之外，鼓励按算子语义补充定制检查。
- 例如 Softmax：
    - 检查 Top-K 索引一致性（如 Top-1 / Top-5）
    - 检查概率和是否接近 1（允许小数值误差）
- 例如 LayerNorm：检查均值/方差统计是否落在预期范围。
- 例如 MatMul/Gemm：可增加相对误差（relative error）统计，避免仅看绝对误差。

### 5.7 资源释放

- 正确释放 `OMTensorList` 等运行时对象
- 保证异常路径不泄漏关键资源

---

## 6. 最小交付清单（AI 生成完成后自检）

生成内容至少应包含：

1. 包含目标算子的“包裹结构”模型（非裸算子）
2. 对齐 shape/dtype 的输入二进制
3. 对应的 golden 输出二进制
4. 可直接用于推理与比对的 `main.cpp`
5. 文件名一致性检查结果
6.（若量化）对称量化参数说明与 zero point 检查说明
7. `build/[model_name]/output/` 目录内 3 个最终文件名清单

---

## 7. 给其他 AI 的固定指令模板（可直接复用）

你要生成一个“算子级测试模型”，要求如下：

1. 目标算子为 `[OP_NAME]`，禁止只生成裸算子，必须使用 `前置轻量层 -> 目标算子 -> 后置轻量层` 结构，防止被优化掉。
2. 输入 shape 必须选自真实模型常见尺寸（而非仅 toy shape），dtype 为 `float32`，并固定随机种子。
3. 生成 `.onnx` 模型，以及输入二进制与 golden 输出二进制；二进制采用 `[model_name]_input.bin`、`[model_name]_output_golden.bin` 命名。
4. 若做量化，仅量化 `[OP_NAME]`，并显式给出对称量化配置。
5. 同时给出 `main.cpp`：
    - 按可执行文件名推导 `model_name` 前缀并读取输入二进制
   - 构造 OMTensor 并调用模型
   - 读取 golden 输出并计算误差
    - 打印 max diff / mse 与误差明细（小张量全量或大张量 Top-K）
    - 如算子适用，增加定制化验证（例如 Softmax 的 Top-K 一致性）
   - 正确释放资源
6. 输出最终自检清单，确认文件名、shape、dtype、阈值一致，并确认 `model.onnx.mlir` 中目标算子满足 `DequantizeLinear -> [OP_NAME] -> QuantizeLinear`。
7. 给出 `build/[model_name]/output/` 最终交付清单（可执行文件 + input + golden）。
8. 不执行 `make`，仅给出建议命令，由开发者手动执行。

---

## 模板参考
1. model.py
```python
import torch
import torch.nn as nn
import torch.onnx
from torch.onnx import register_custom_op_symbolic
import onnx
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod
)
import onnxruntime as ort
import numpy as np
import os

# ==========================================
# 1. 模型定义 (极简版：Add -> Gelu -> Sub)
# ==========================================

class SimpleGeluModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 目标核心算子
        self.gelu = nn.GELU()

    def forward(self, x):
        # 1. 前置简单加法 (模拟预处理)
        x = x + 0.001 
        
        # 2. 核心 Gelu (目标量化层)
        x = self.gelu(x) 
        
        # 3. 后置简单减法 (模拟后处理)
        x = x - 0.001
        return x

# 注册自定义符号，确保 Gelu 导出为标准的 ONNX Gelu 节点
# 注意：opset 20 中 Gelu 是标准算子，但保留此函数以兼容旧习惯或特定需求
def my_gelu_symbolic(g, self, approximate='none'):
    return g.op("Gelu", self)

register_custom_op_symbolic("aten::gelu", my_gelu_symbolic, opset_version=20)

# ==========================================
# 2. 数据校准器
# ==========================================
class RandomDataReader(CalibrationDataReader):
    def __init__(self, input_name, input_shape):
        self.input_name = input_name
        self.input_shape = input_shape
        self.data = self._create_data()
        self.enum_data = iter(self.data)

    def _create_data(self):
        # 模拟 10 组校准数据
        # 使用 randn 产生正态分布数据，覆盖正负值，测试 Gelu 的非线性区间
        return [{self.input_name: np.random.randn(*self.input_shape).astype(np.float32)} 
                for _ in range(10)]

    def get_next(self):
        return next(self.enum_data, None)

# ==========================================
# 3. 主流程
# ==========================================
def run_pipeline():
    # 路径配置
    model_name = "gelu"
    fp32_model_path = "simple_gelu_fp32.onnx"
    quant_model_path = "simple_gelu_quant_symmetric.onnx"
    input_bin_path = f"{model_name}_input.bin"
    output_bin_path = f"{model_name}_output_golden.bin"
    
    # 输入形状建议使用更贴近真实模型的常见维度
    # 例如 Transformer 家族常见 hidden size: 768
    input_shape = (1, 768)

    # --- Step A: 导出 FP32 模型 ---
    print("\n[Step 1] 导出 Simple FP32 模型...")
    model = SimpleGeluModel()
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    torch.onnx.export(
        model, dummy_input, fp32_model_path,
        input_names=['input'], output_names=['output'],
        opset_version=20, 
        do_constant_folding=True
    )
    print(f" -> 导出成功: {fp32_model_path}")

    # --- Step B: 执行对称量化 (Target: Only Gelu, Int8 Symmetric) ---
    print("\n[Step 2] 开始对称量化...")
    dr = RandomDataReader('input', input_shape)

    # 对称量化核心配置：强制 ZeroPoint 为 0
    extra_options = {
        'ActivationSymmetric': True, 
        'WeightSymmetric': True
    }

    quantize_static(
        model_input=fp32_model_path,
        model_output=quant_model_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ, # 插入 QuantizeLinear/DequantizeLinear 节点
        
        # 只量化 Gelu，前后的 Add/Sub 保持 FP32
        op_types_to_quantize=['Gelu'], 
        
        # 使用 QInt8 (Signed 8-bit, -128 ~ 127)
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        
        # MinMax 配合 Symmetric 最合适
        calibrate_method=CalibrationMethod.MinMax,
        
        extra_options=extra_options
    )
    print(f" -> 对称量化成功: {quant_model_path}")

    # --- Step C: 生成验证数据 (Golden Data) ---
    print("\n[Step 3] 生成验证数据...")
    
    # 1. 生成固定输入数据
    np.random.seed(123) 
    test_input = np.random.randn(*input_shape).astype(np.float32)
    
    # 2. 保存输入
    test_input.tofile(input_bin_path)
    
    # 3. 使用 ORT 运行量化后的模型
    sess = ort.InferenceSession(quant_model_path)
    input_name = sess.get_inputs()[0].name
    ort_inputs = {input_name: test_input}
    
    # 运行推断
    ort_outputs = sess.run(None, ort_inputs)
    golden_output = ort_outputs[0] 

    # 4. 保存输出
    golden_output.tofile(output_bin_path)
    print(f" -> 输入 ({input_bin_path}) 与 输出 ({output_bin_path}) 已保存")
    
    print("\n[Data Preview]")
    # 打印前 5 个数值，方便直观感受 Add -> Gelu -> Sub 的过程
    print(f"Input Raw:    {test_input.flatten()}")
    # 这里的 Input to Gelu = Input Raw + 1.5
    print(f"Output Final: {golden_output.flatten()}")

    # --- (Optional) 验证 Zero Point ---
    print("\n[Verification] 检查 Zero Point (应为 0)...")
    model_onnx = onnx.load(quant_model_path)
    zp_check_pass = True
    
    for initializer in model_onnx.graph.initializer:
        if "zero_point" in initializer.name:
            if initializer.data_type == onnx.TensorProto.INT8:
                raw_data = initializer.raw_data
                zp_values = np.frombuffer(raw_data, dtype=np.int8)
                if not np.all(zp_values == 0):
                    print(f" [!] Warning: Non-zero ZP found in {initializer.name}: {zp_values}")
                    zp_check_pass = False
    
    if zp_check_pass:
        print(" -> 验证通过: 所有 INT8 Zero Point 均为 0 (Symmetric)")

if __name__ == "__main__":
    run_pipeline()
```

2. main.cpp
```cpp
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>

// 引入 ONNX-MLIR 运行时头文件
// 确保编译时能找到这个头文件
#include "OnnxMlirRuntime.h"

// 声明外部生成的模型入口函数 (由 onnx-mlir 编译出的库提供)
extern "C" OMTensorList *run_main_graph(OMTensorList *);

// ==========================================
// 辅助函数：读取二进制文件到 vector
// ==========================================
std::vector<float> load_binary_file(const std::string& filename, int64_t expected_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 检查文件大小是否匹配预期 (float 是 4 字节)
    if (size != expected_size * sizeof(float)) {
        std::cerr << "Error: File size mismatch for " << filename 
                  << ". Expected bytes: " << expected_size * sizeof(float)
                  << ", Actual bytes: " << size << std::endl;
        exit(1);
    }

    std::vector<float> data(expected_size);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        std::cerr << "Error: Failed to read data from " << filename << std::endl;
        exit(1);
    }
    return data;
}

// ==========================================
// 辅助函数：比较结果
// ==========================================
void verify_output(const float* actual, const std::vector<float>& golden, int64_t size) {
    float max_diff = 0.0f;
    float mse = 0.0f;
    
    std::cout << "\n=== Verification Report ===" << std::endl;
    std::cout << std::setw(10) << "Index" << std::setw(15) << "NPU Output" 
              << std::setw(15) << "Golden" << std::setw(15) << "Diff" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    for (int64_t i = 0; i < size; ++i) {
        float diff = std::abs(actual[i] - golden[i]);
        max_diff = std::max(max_diff, diff);
        mse += diff * diff;

        // 打印所有数据点 (因为只有 16 个，可以直接全部打印出来分析)
        std::cout << std::setw(10) << i 
                  << std::setw(15) << actual[i] 
                  << std::setw(15) << golden[i] 
                  << std::setw(15) << diff << std::endl;
    }
    mse /= size;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Max Absolute Error: " << max_diff << std::endl;
    std::cout << "Mean Squared Error: " << mse << std::endl;

    // 判定阈值：由于包含 Int8 量化，允许一定程度的精度损失
    // 如果 Python 脚本生成的 golden 也是量化后的结果，那么误差应该极小
    if (max_diff < 0.1) { 
        std::cout << "✅ RESULT: PASS" << std::endl;
    } else {
        std::cout << "⚠️ RESULT: WARNING (Error too large)" << std::endl;
    }
}

int main(int argc, char **argv) {
    // ==========================================
    // 1. 定义形状 (与 Python 脚本保持一致)
    // ==========================================
    // Python: input_shape = (1, 768)
    int64_t dim_0 = 1;
    int64_t dim_1 = 768;
    int64_t input_elements = dim_0 * dim_1;

    // 根据可执行文件名推导模型前缀，例如 gelu_zcu102 -> gelu
    std::string model_prefix = "gelu";
    if (argc > 0) {
        std::string exe = argv[0];
        auto pos = exe.find_last_of('/');
        if (pos != std::string::npos) exe = exe.substr(pos + 1);
        const std::string suffix = "_zcu102";
        if (exe.size() > suffix.size() &&
            exe.compare(exe.size() - suffix.size(), suffix.size(), suffix) == 0) {
            model_prefix = exe.substr(0, exe.size() - suffix.size());
        }
    }

    std::string input_filename = model_prefix + "_input.bin";
    std::string golden_filename = model_prefix + "_output_golden.bin";

    // ==========================================
    // 2. 加载输入数据
    // ==========================================
    std::cout << "Loading " << input_filename << "..." << std::endl;
    std::vector<float> input_data = load_binary_file(input_filename, input_elements);

    // ==========================================
    // 3. 构造 OMTensor (输入)
    // ==========================================
    OMTensor *inputTensors[1];
    int64_t input_shape[] = {dim_0, dim_1};
    int64_t rank = 2; // Rank = 2, Shape = [1, 768]
    
    // 创建输入 Tensor，类型为 FLOAT (onnx-mlir 内部会处理到 int8 的转换)
    OMTensor *tensor = omTensorCreate(input_data.data(), input_shape, rank, ONNX_TYPE_FLOAT);
    inputTensors[0] = tensor;
    
    // 创建 Tensor List
    OMTensorList *tensorListIn = omTensorListCreate(inputTensors, 1);

    // ==========================================
    // 4. 执行推理 (调用编译好的模型)
    // ==========================================
    std::cout << "Running inference on Compiled Model..." << std::endl;
    OMTensorList *tensorListOut = run_main_graph(tensorListIn);

    if (!tensorListOut) {
        std::cerr << "Error: Inference failed (returned NULL)." << std::endl;
        return 1;
    }

    // ==========================================
    // 5. 获取输出
    // ==========================================
    // 获取第一个输出 Tensor
    OMTensor *output_tensor = omTensorListGetOmtByIndex(tensorListOut, 0);
    float *output_data = (float *)omTensorGetDataPtr(output_tensor);
    
    // 验证输出形状
    const int64_t *out_shape_ptr = omTensorGetShape(output_tensor);
    int64_t out_rank = omTensorGetRank(output_tensor);
    int64_t total_output_elements = 1;
    
    std::cout << "Output Shape: [";
    for(int i=0; i<out_rank; i++) {
        std::cout << out_shape_ptr[i] << (i < out_rank - 1 ? " x " : "");
        total_output_elements *= out_shape_ptr[i];
    }
    std::cout << "]" << std::endl;

    // ==========================================
    // 6. 加载 Golden Data 并进行比对
    // ==========================================
    std::cout << "Loading " << golden_filename << " for verification..." << std::endl;
    std::vector<float> golden_data = load_binary_file(golden_filename, total_output_elements);
    
    verify_output(output_data, golden_data, total_output_elements);

    // ==========================================
    // 7. 清理内存
    // ==========================================
    omTensorListDestroy(tensorListIn); 
    omTensorListDestroy(tensorListOut);

    return 0;
}
```