import torch
import torch.nn as nn
import torch.ao.quantization as quantization
import os

class SymmetricFakeQuant(nn.Module):
    def __init__(self):
        super().__init__()
        self.fake_quant = quantization.FakeQuantize.with_args(
            observer=quantization.MinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False
        )()
        
    def forward(self, x):
        return self.fake_quant(x)

class SpecificQuantModel(nn.Module):
    def __init__(self, input_res=224):
        super().__init__()
        self.input_res = input_res
        
        # 1. Conv (FP32)
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False)
        
        # 2. Conv 输出后的量化 (Q->DQ)
        self.qdq_conv_out = SymmetricFakeQuant()
        
        # 3. GELU
        self.gelu = nn.GELU()
        
        # 4. GELU 输出后的量化 (Q->DQ)
        self.qdq_gelu_out = SymmetricFakeQuant()
        
        # 5. Linear (FP32)
        # 计算逻辑: 4 channels * input_res * input_res
        self.fc_input_dim = 4 * self.input_res * self.input_res
        self.fc = nn.Linear(self.fc_input_dim, 10, bias=False)

    def forward(self, x):
        # x shape: [Batch, 1, 224, 224]
        x = self.conv(x)
        
        # Q -> DQ
        x = self.qdq_conv_out(x)
        
        x = self.gelu(x)
        
        # Q -> DQ
        x = self.qdq_gelu_out(x)
        
        # Reshape 为 [Batch, 4*224*224]
        x = x.reshape(x.size(0), -1)
        
        # Linear (FP32)
        x = self.fc(x)
        
        return x

def main():
    # 设置输入分辨率
    RES = 224
    model = SpecificQuantModel(input_res=RES)
    
    model.train()

    print(f"Running calibration with input resolution {RES}x{RES}...")
    with torch.no_grad():
        # 模拟校准过程
        for _ in range(5):
            # 这里的输入尺寸改为 [1, 1, 224, 224]
            model(torch.randn(1, 1, RES, RES))

    model.eval()
    model.apply(quantization.disable_observer)

    onnx_path = "model.onnx"
    print(f"Exporting to {onnx_path}...")
    
    # 修改 Dummy Input 尺寸
    dummy_input = torch.randn(1, 1, RES, RES)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=13,
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True
    )
    
    print(f"Done! Final Linear Layer Input Dim: {model.fc_input_dim}")
    print("Graph: Conv -> Q/DQ -> GELU -> Q/DQ -> Flatten -> Linear")

if __name__ == "__main__":
    main()