import torch
import torchvision.models as models
import numpy as np
import os

def generate_vgg16_test_data():
    # 1. 加载模型 (确保和你量化时用的权重一致，这里用预训练权重演示)
    # 注意：如果你量化的是无权重的模型，这里也得由随机权重生成，否则输出是无意义的
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.eval()

    # 2. 创建 VGG16 标准输入 (1, 3, 224, 224)
    # 这里使用随机数据，或者你可以加载一张真实的图片并进行预处理
    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(input_shape, dtype=torch.float32)

    # 3. 获取标准输出 (Golden Output)
    with torch.no_grad():
        output = model(dummy_input)
    
    # 转为 numpy
    input_np = dummy_input.numpy()
    output_np = output.numpy()

    print(f"Input shape: {input_np.shape}")
    print(f"Output shape: {output_np.shape}")

    # 4. 保存为二进制文件 (.bin) 供 C++ 读取
    input_np.tofile("vgg16_input.bin")
    output_np.tofile("vgg16_golden.bin")
    
    print("✅ 已生成 vgg16_input.bin 和 vgg16_golden.bin")
    
    # 打印 Top-5 预测结果供参考
    top5_prob, top5_catid = torch.topk(output, 5)
    print("Python Top-1 Class ID:", top5_catid[0][0].item())

if __name__ == "__main__":
    generate_vgg16_test_data()