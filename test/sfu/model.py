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
# 1. 模型定义 (针对 NPU 激活函数测试优化)
# ==========================================

class ActivationTestModel(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()
        self.input_dim = input_dim
        
        # 1. 头部 Add 层：防止 Input 直接进入 LayerNorm 被编译器融合
        # 使用 Parameter 确保它被视为模型权重的一部分
        self.head_bias = nn.Parameter(torch.randn(input_dim) * 0.1)
        
        # 2. 待测试的核心算子
        self.ln = nn.LayerNorm(input_dim)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)
        
        # 3. 尾部 Add 层：防止 Softmax 直接输出，增加优化难度
        self.tail_bias = nn.Parameter(torch.randn(input_dim) * 0.1)

    def forward(self, x):
        # Head Add
        x = torch.add(x, self.head_bias)
        
        # Core Activations
        x = self.ln(x)
        x = self.gelu(x)
        x = self.softmax(x)
        
        # Tail Add
        x = torch.add(x, self.tail_bias)
        return x

# 注册自定义符号，确保 Gelu 导出为标准的 ONNX Gelu 节点 (而非 Tanh 近似子图)
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
        # 模拟 20 组校准数据，增加数据量以覆盖 Softmax 的分布
        return [{self.input_name: np.random.randn(*self.input_shape).astype(np.float32)} 
                for _ in range(20)]

    def get_next(self):
        return next(self.enum_data, None)

# ==========================================
# 3. 主流程：导出 -> 对称量化 -> 生成验证数据
# ==========================================
def run_pipeline():
    # 路径配置
    fp32_model_path = "act_test_fp32.onnx"
    quant_model_path = "act_test_quant_sym.onnx"
    input_bin_path = "input_act.bin"
    output_bin_path = "output_act_golden.bin"
    
    # 输入形状 (Batch=1, Dim=32)
    # 使用一维向量更能体现 Softmax 和 LayerNorm 的特性
    input_dim = 32
    input_shape = (1, input_dim)

    # --- Step A: 导出 FP32 模型 ---
    print("\n[Step 1] 导出 FP32 模型 (Add -> LN -> Gelu -> Softmax -> Add)...")
    model = ActivationTestModel(input_dim=input_dim)
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    torch.onnx.export(
        model, dummy_input, fp32_model_path,
        input_names=['input'], output_names=['output'],
        # Opset 17+ 支持标准的 LayerNormalization 算子
        opset_version=20, 
        do_constant_folding=True
    )
    print(f" -> 导出成功: {fp32_model_path}")

    # --- Step B: 执行对称量化 ---
    print("\n[Step 2] 开始对称量化 (Target: All Ops, Int8 Symmetric)...")
    dr = RandomDataReader('input', input_shape)

    extra_options = {
        'ActivationSymmetric': True, 
        'WeightSymmetric': True
    }

    # 指定需要尝试量化的算子类型
    # 注意：Softmax 和 LayerNormalization 在某些 ORT 版本中可能默认保留为 Float，
    # 但我们显式列出它们，期望生成 QDQ 节点包围它们。
    target_ops = [ 'Gelu', 'LayerNormalization', 'Softmax']

    quantize_static(
        model_input=fp32_model_path,
        model_output=quant_model_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=target_ops, # 扩大量化范围
        
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options=extra_options
    )
    print(f" -> 对称量化成功: {quant_model_path}")

    # --- Step C: 生成验证数据 (Golden Data) ---
    print("\n[Step 3] 生成验证数据...")
    
    np.random.seed(1024) 
    # 扩大输入范围，确保 Softmax 不会全输出 0 或 1，增加测试难度
    test_input = (np.random.randn(*input_shape) * 2.0).astype(np.float32)
    
    test_input.tofile(input_bin_path)
    print(f" -> 输入数据已保存: {input_bin_path}")

    # 使用 ORT 运行量化模型
    sess = ort.InferenceSession(quant_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    ort_outputs = sess.run([output_name], {input_name: test_input})
    golden_output = ort_outputs[0] 

    golden_output.tofile(output_bin_path)
    print(f" -> 标准输出已保存: {output_bin_path}")
    
    print("\n[Data Sample]")
    print(f"Input  (First 5): {test_input.flatten()[:5]}")
    print(f"Output (First 5): {golden_output.flatten()[:5]}")

    # --- (Optional) 检查特定节点的量化情况 ---
    print("\n[Graph Inspection]")
    model_onnx = onnx.load(quant_model_path)
    node_counts = {}
    for node in model_onnx.graph.node:
        node_counts[node.op_type] = node_counts.get(node.op_type, 0) + 1
    
    print("模型算子统计:", node_counts)
    
    if node_counts.get('QuantizeLinear', 0) > 0:
        print(" -> 检测到 QDQ 节点，量化已插入。")
        print("    (注: Softmax/LayerNorm 可能被 QDQ 包围，也可能内部仍为 Float，取决于 ORT 策略)")
    else:
        print(" -> Warning: 未检测到量化节点。")

if __name__ == "__main__":
    run_pipeline()