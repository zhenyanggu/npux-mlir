#include <iostream>
#include <vector>
#include <numeric>

// 引入 ONNX-MLIR 运行时头文件
#include "OnnxMlirRuntime.h"

// 声明推理入口函数
// onnx-mlir 编译生成的默认入口点通常是 run_main_graph
extern "C" OMTensorList *run_main_graph(OMTensorList *);

int main() {
    // 1. 准备输入数据
    // 模型输入形状: 1x3x224x224
    int64_t batch = 1;
    int64_t channels = 3;
    int64_t height = 224;
    int64_t width = 224;
    int64_t input_size = batch * channels * height * width;

    // 使用 vector 管理内存，避免栈溢出
    std::vector<float> input_data(input_size);
    
    // 初始化输入数据 (这里为了演示简单填充为 1.0f，你可以改为读取图片或随机数)
    for (int64_t i = 0; i < input_size; ++i) {
        input_data[i] = 1.0f; 
        // input_data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // 或者随机数
    }

    // 2. 创建 OMTensor
    int inputNum = 1;
    OMTensor *inputTensors[inputNum];
    
    int64_t rank = 4;
    int64_t shape[] = {batch, channels, height, width};
    
    // 注意：omTensorCreate 不会拷贝数据，所以 input_data 必须在推理期间保持有效
    OMTensor *tensor = omTensorCreate(input_data.data(), shape, rank, ONNX_TYPE_FLOAT);
    
    inputTensors[0] = tensor;
    OMTensorList *tensorListIn = omTensorListCreate(inputTensors, inputNum);

    // 3. 执行推理
    std::cout << "Running inference..." << std::endl;
    OMTensorList *tensorListOut = run_main_graph(tensorListIn);

    // 4. 清理输入 (不再需要)
    omTensorListDestroy(tensorListIn);

    // 5. 获取并分析输出
    // 模型输出: tensor<1x16x224x224xf32>
    OMTensor *output_tensor = omTensorListGetOmtByIndex(tensorListOut, 0);
    float *output_data = (float *)omTensorGetDataPtr(output_tensor);
    
    // 获取输出形状进行验证
    int64_t out_rank = omTensorGetRank(output_tensor);
    const int64_t *out_shape = omTensorGetShape(output_tensor);
    int64_t total_output_elements = 1;
    
    std::cout << "Output Shape: [";
    for(int i=0; i<out_rank; i++) {
        std::cout << out_shape[i] << (i < out_rank - 1 ? " x " : "");
        total_output_elements *= out_shape[i];
    }
    std::cout << "]" << std::endl;

    // 6. 打印部分结果用于验证 (由于数据量太大，只打印前10个值)
    std::cout << "First 10 output values:" << std::endl;
    for (int i = 0; i < 10 && i < total_output_elements; i++) {
        std::cout << "output[" << i << "] = " << output_data[i] << std::endl;
    }

    // 7. 清理输出
    omTensorListDestroy(tensorListOut);

    return 0;
}