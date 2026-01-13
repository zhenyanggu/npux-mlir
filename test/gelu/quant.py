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
# 1. 模型定义 & 自定义导出
# ==========================================

class FullOpsModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (N, 1, 8, 8)
        self.conv = nn.Conv2d(1, 4, 3, padding=1, bias=True) # -> (N, 4, 8, 8)
        self.flatten = nn.Flatten()                          # -> (N, 256)
        self.fc1 = nn.Linear(256, 16, bias=True)             # -> (N, 16)
        self.ln = nn.LayerNorm(16)                           # -> (N, 16)
        self.gelu = nn.GELU()                                # -> (N, 16) [目标量化层]
        self.fc2 = nn.Linear(16, 10, bias=True)              # -> (N, 10)
        self.softmax = nn.Softmax(dim=1)                     # -> (N, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ln(x)
        x = self.gelu(x) 
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# 注册自定义符号，确保 Gelu 导出为标准的 ONNX Gelu 节点
def my_gelu_symbolic(g, self, approximate='none'):
    return g.op("Gelu", self)

register_custom_op_symbolic("aten::gelu", my_gelu_symbolic, opset_version=17)

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
        return [{self.input_name: np.random.randn(*self.input_shape).astype(np.float32)} 
                for _ in range(10)]

    def get_next(self):
        return next(self.enum_data, None)

# ==========================================
# 3. 主流程：导出 -> 量化 -> 生成验证数据
# ==========================================
def run_pipeline():
    # 路径配置
    fp32_model_path = "model_fp32.onnx"
    quant_model_path = "model_quant_only_gelu.onnx"
    input_bin_path = "input.bin"
    output_bin_path = "output_golden.bin"
    
    # 输入形状 (Batch=1, Channel=1, H=8, W=8)
    input_shape = (1, 1, 8, 8)

    # --- Step A: 导出 FP32 模型 ---
    print("\n[Step 1] 导出 FP32 模型...")
    model = FullOpsModel()
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    torch.onnx.export(
        model, dummy_input, fp32_model_path,
        input_names=['input'], output_names=['output'],
        opset_version=20, # 使用较高的 opset 以支持 LayerNorm/Gelu 原生算子
        do_constant_folding=True
    )
    print(f" -> 导出成功: {fp32_model_path}")

    # --- Step B: 执行量化 ---
    print("\n[Step 2] 开始量化 (Target: Only Gelu)...")
    dr = RandomDataReader('input', input_shape)

    quantize_static(
        model_input=fp32_model_path,
        model_output=quant_model_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,     # 生成 Q-DQ 节点对，适合 NPU 编译器处理
        op_types_to_quantize=['Gelu'],    # 【关键】只量化 Gelu
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print(f" -> 量化成功: {quant_model_path}")

    # --- Step C: 生成验证数据 (Golden Data) ---
    print("\n[Step 3] 生成验证数据 (For C++ Verification)...")
    
    # 1. 生成固定输入数据 (float32)
    # 使用随机数，但固定下来保存到文件
    np.random.seed(42) # 固定随机种子以保证复现
    test_input = np.random.randn(*input_shape).astype(np.float32)
    
    # 2. 保存输入到二进制文件
    test_input.tofile(input_bin_path)
    print(f" -> 输入数据已保存: {input_bin_path} (Shape: {test_input.shape}, Bytes: {os.path.getsize(input_bin_path)})")

    # 3. 使用 ONNX Runtime 运行量化后的模型作为 "Golden Standard"
    # 我们对比的是：NPU执行量化模型 vs CPU(ORT)执行量化模型
    # 这样能排除 "量化本身带来的精度损失"，专注于验证 "NPU编译器/硬件执行的正确性"
    sess = ort.InferenceSession(quant_model_path)
    
    input_name = sess.get_inputs()[0].name
    ort_inputs = {input_name: test_input}
    ort_outputs = sess.run(None, ort_inputs)
    golden_output = ort_outputs[0] # numpy array

    # 4. 保存标准输出到二进制文件
    golden_output.tofile(output_bin_path)
    print(f" -> 标准输出已保存: {output_bin_path} (Shape: {golden_output.shape}, Bytes: {os.path.getsize(output_bin_path)})")
    
    # 打印部分数据供目视检查
    print("\n[Preview Data]")
    print(f"Input [0,0,0,:5]: {test_input.flatten()[:5]}")
    print(f"Output[:5]:       {golden_output.flatten()[:5]}")

if __name__ == "__main__":
    run_pipeline()