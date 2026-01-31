import torch
import torchvision.models as models
import torch.nn as nn
from transformers import BertModel
from ultralytics import YOLO
import os
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod
)

# ==========================================
# 1. 辅助类与函数定义
# ==========================================

# 自定义 LeNet (保持不变)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class GenericDataReader(CalibrationDataReader):
    """
    通用校准数据读取器。
    支持单输入 (如 ResNet) 和多输入 (如 BERT)。
    """
    def __init__(self, input_specs, data_count=10):
        """
        :param input_specs: 字典，格式为 {'input_name': (shape, dtype_str)}
                            dtype_str 支持 'float32' 或 'int64'
        :param data_count: 校准数据的批次数
        """
        self.input_specs = input_specs
        self.data_count = data_count
        self.data = self._create_data()
        self.enum_data = iter(self.data)

    def _create_data(self):
        data_list = []
        for _ in range(self.data_count):
            input_feed = {}
            for name, (shape, dtype) in self.input_specs.items():
                if dtype == 'int64':
                    # 针对 BERT 等需要整数输入的模型 (模拟 token ID)
                    # 假设 vocab size 足够大，这里随机生成 0-1000 之间的整数
                    input_feed[name] = np.random.randint(0, 1000, size=shape, dtype=np.int64)
                else:
                    # 针对图像模型，生成 float32 数据
                    input_feed[name] = np.random.randn(*shape).astype(np.float32)
            data_list.append(input_feed)
        return data_list

    def get_next(self):
        return next(self.enum_data, None)

def quantize_to_int8_symmetric(fp32_model_path, output_model_path, data_reader):
    """
    执行 Int8 对称量化
    """
    print(f"   -> 正在量化模型: {fp32_model_path} ...")
    
    # 对称量化关键配置
    extra_options = {
        'ActivationSymmetric': True,  # 激活值对称 (ZP=0)
        'WeightSymmetric': True       # 权重对称 (ZP=0)
    }

    try:
        quantize_static(
            model_input=fp32_model_path,
            model_output=output_model_path,
            calibration_data_reader=data_reader,
            quant_format=QuantFormat.QDQ,     # QDQ 格式通常对推理引擎更友好
            
            # 使用 QInt8 (Signed 8-bit, -128 to 127) 配合 MinMax 策略是对称量化的标准做法
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            
            extra_options=extra_options
        )
        print(f"   ✅ 量化完成: {output_model_path}")
    except Exception as e:
        print(f"   ❌ 量化失败: {e}")

# ==========================================
# 2. 主流程
# ==========================================

def export_and_quantize():
    os.makedirs("static_onnx_models", exist_ok=True)
    device = torch.device("cpu")

    # --- 1. CV 模型 (Batch Size = 1) ---
    cv_models = {
        "resnet50": models.resnet50(weights=None), 
        "mobilenet_v2": models.mobilenet_v2(weights=None),
        "vgg16": models.vgg16(weights=None),
        "lenet": LeNet()
    }

    for name, model in cv_models.items():
        print(f"\n[处理模型: {name}]")
        model.eval()
        
        # 定义形状
        input_shape = (1, 1, 28, 28) if name == "lenet" else (1, 3, 224, 224)
        dummy_input = torch.randn(input_shape).to(device)
        
        fp32_path = f"static_onnx_models/{name}_static.onnx"
        quant_path = f"static_onnx_models/{name}_int8_sym.onnx"
        
        # 1.1 导出 FP32
        torch.onnx.export(
            model, dummy_input, fp32_path,
            export_params=True, opset_version=12,
            do_constant_folding=True,
            input_names=['input'], output_names=['output']
        )
        
        # 1.2 量化
        # 构造校准器
        input_specs = {'input': (input_shape, 'float32')}
        dr = GenericDataReader(input_specs)
        
        # 执行量化
        quantize_to_int8_symmetric(fp32_path, quant_path, dr)

    # --- 2. BERT 模型 ---
    print("\n[处理模型: BERT]")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()
    
    fixed_batch = 1
    fixed_seq_len = 128
    dummy_input_ids = torch.ones(fixed_batch, fixed_seq_len, dtype=torch.long)
    dummy_mask = torch.ones(fixed_batch, fixed_seq_len, dtype=torch.long)
    
    fp32_path = "static_onnx_models/bert_static.onnx"
    quant_path = "static_onnx_models/bert_int8_sym.onnx"

    # 2.1 导出 FP32
    torch.onnx.export(
        bert_model, (dummy_input_ids, dummy_mask), fp32_path,
        export_params=True, opset_version=12,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state', 'pooler_output']
    )

    # 2.2 量化
    # BERT 需要两个输入，且类型为 int64
    input_specs = {
        'input_ids': ((fixed_batch, fixed_seq_len), 'int64'),
        'attention_mask': ((fixed_batch, fixed_seq_len), 'int64')
    }
    dr = GenericDataReader(input_specs)
    quantize_to_int8_symmetric(fp32_path, quant_path, dr)

    # --- 3. YOLO 模型 ---
    print("\n[处理模型: YOLO]")
    # 注意：YOLO export 可能会下载模型
    yolo_model = YOLO("yolov8n.pt") 
    
    # 3.1 导出 FP32
    # ultralytics 的 export 会自动在当前目录或 weights 目录生成文件，我们需要获取该路径
    # 默认情况下，format='onnx' 会生成 'yolov8n.onnx' 在当前工作目录
    exported_path = yolo_model.export(format="onnx", opset=12, dynamic=False, imgsz=[640, 640])
    print(f"YOLO 原生导出路径: {exported_path}")
    
    # 确保我们知道导出文件在哪 (通常是 string path)
    if isinstance(exported_path, str):
        fp32_path = exported_path
    else:
        # Fallback，防止返回值变化
        fp32_path = "yolov8n.onnx"

    quant_path = "static_onnx_models/yolov8n_int8_sym.onnx"

    # 3.2 量化
    # YOLOv8 默认输入名为 'images'，形状 (1, 3, 640, 640)
    # 若不确定输入名，可以用 onnx.load 查看，这里按标准 YOLOv8 处理
    input_specs = {'images': ((1, 3, 640, 640), 'float32')}
    dr = GenericDataReader(input_specs)
    quantize_to_int8_symmetric(fp32_path, quant_path, dr)

if __name__ == "__main__":
    export_and_quantize()