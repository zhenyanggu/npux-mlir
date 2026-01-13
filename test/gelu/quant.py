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
        return [{self.input_name: np.random.randn(*self.input_shape).astype(np.float32)} 
                for _ in range(10)]

    def get_next(self):
        return next(self.enum_data, None)

# ==========================================
# 3. 主流程：导出 -> 对称量化 -> 生成验证数据
# ==========================================
def run_pipeline():
    # 路径配置
    fp32_model_path = "model_fp32.onnx"
    quant_model_path = "model_quant_symmetric_gelu.onnx" # 修改文件名以区分
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
        opset_version=20, 
        do_constant_folding=True
    )
    print(f" -> 导出成功: {fp32_model_path}")

    # --- Step B: 执行对称量化 (Symmetric Quantization) ---
    print("\n[Step 2] 开始对称量化 (Target: Only Gelu, Int8 Symmetric)...")
    dr = RandomDataReader('input', input_shape)

    # 配置对称量化的关键参数
    # 1. ActivationSymmetric=True: 激活值强制对称 (ZeroPoint=0)
    # 2. WeightSymmetric=True: 权重强制对称 (ZeroPoint=0)
    extra_options = {
        'ActivationSymmetric': True, 
        'WeightSymmetric': True
    }

    quantize_static(
        model_input=fp32_model_path,
        model_output=quant_model_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=['Gelu'],
        
        # 【关键修改 1】使用 QInt8 (Signed 8-bit)，范围 -128 到 127
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        
        # 【关键修改 2】MinMax 方法最适合对称量化 (取 abs max)
        calibrate_method=CalibrationMethod.MinMax,
        
        # 【关键修改 3】传入 extra_options 强制开启对称
        extra_options=extra_options
    )
    print(f" -> 对称量化成功: {quant_model_path}")

    # --- Step C: 生成验证数据 (Golden Data) ---
    print("\n[Step 3] 生成验证数据 (For C++ Verification)...")
    
    # 1. 生成固定输入数据 (float32)
    np.random.seed(42) 
    test_input = np.random.randn(*input_shape).astype(np.float32)
    
    # 2. 保存输入
    test_input.tofile(input_bin_path)
    print(f" -> 输入数据已保存: {input_bin_path}")

    # 3. 使用 ORT 运行量化后的模型
    sess = ort.InferenceSession(quant_model_path)
    input_name = sess.get_inputs()[0].name
    ort_inputs = {input_name: test_input}
    
    # 运行推断
    ort_outputs = sess.run(None, ort_inputs)
    golden_output = ort_outputs[0] 

    # 4. 保存输出
    golden_output.tofile(output_bin_path)
    print(f" -> 标准输出已保存: {output_bin_path}")
    
    print("\n[Preview Data]")
    print(f"Input Sample: {test_input.flatten()[:5]}")
    print(f"Output Sample: {golden_output.flatten()[:5]}")

    # --- (Optional) 验证 Zero Point 是否为 0 ---
    print("\n[Verification] 检查量化参数是否为对称 (Zero Point 必须为 0)...")
    model_onnx = onnx.load(quant_model_path)
    zp_found = False
    for initializer in model_onnx.graph.initializer:
        if "zero_point" in initializer.name:
            # 读取 TensorProto 数据
            if initializer.data_type == onnx.TensorProto.INT8:
                raw_data = initializer.raw_data
                zp_values = np.frombuffer(raw_data, dtype=np.int8)
                if np.all(zp_values == 0):
                    zp_found = True
                else:
                    print(f"Warning: Found non-zero ZP in {initializer.name}: {zp_values}")
    
    if zp_found:
        print(" -> 验证通过: 检测到 Zero Point 为 0 的量化参数节点。")
    else:
        print(" -> 注意: 未检测到显式的 ZP 节点 (可能是因为全是 0 被折叠或者未量化成功)")

if __name__ == "__main__":
    run_pipeline()