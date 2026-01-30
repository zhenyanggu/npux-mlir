import torch
import torchvision.models as models
import torch.nn as nn
from transformers import BertModel
from ultralytics import YOLO
import os

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

def export_static_onnx():
    os.makedirs("static_onnx_models", exist_ok=True)
    device = torch.device("cpu")

    # --- 1. CV 模型 (Batch Size = 1) ---
    cv_models = {
        "resnet50": models.resnet50(weights=None), # 演示用，不加载权重速度更快
        "mobilenet_v2": models.mobilenet_v2(weights=None),
        "vgg16": models.vgg16(weights=None),
        "lenet": LeNet()
    }

    for name, model in cv_models.items():
        model.eval()
        # 定义确定的输入形状
        input_shape = (1, 1, 28, 28) if name == "lenet" else (1, 3, 224, 224)
        dummy_input = torch.randn(input_shape).to(device)
        
        path = f"static_onnx_models/{name}_static.onnx"
        
        # 移除 dynamic_axes 参数即为静态导出
        torch.onnx.export(
            model, dummy_input, path,
            export_params=True, 
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'], 
            output_names=['output']
            # dynamic_axes=None  <-- 默认就是 None
        )
        print(f"✅ 已成功导出静态模型: {path}")

    # --- 2. BERT 模型 (Batch Size = 1, Seq_Len = 128) ---
    print("正在导出静态 BERT...")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()
    
    # 定义固定的输入维度
    fixed_batch = 1
    fixed_seq_len = 128
    dummy_input_ids = torch.ones(fixed_batch, fixed_seq_len, dtype=torch.long)
    dummy_mask = torch.ones(fixed_batch, fixed_seq_len, dtype=torch.long)
    
    torch.onnx.export(
        bert_model, (dummy_input_ids, dummy_mask), "static_onnx_models/bert_static.onnx",
        export_params=True, opset_version=12,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state', 'pooler_output']
        # 不配置 dynamic_axes，模型将只接受 (1, 128) 形状
    )
    print("✅ 已成功导出静态模型: static_onnx_models/bert_static.onnx")

    # --- 3. YOLO 模型 ---
    print("正在导出静态 YOLO...")
    yolo_model = YOLO("yolov8n.pt") 
    # 将 dynamic 设为 False，并指定 imgsz
    yolo_model.export(format="onnx", opset=12, dynamic=False, imgsz=[640, 640]) 
    print("✅ YOLO 已导出为静态形状 (默认 Batch=1, Size=640x640)")

if __name__ == "__main__":
    export_static_onnx()