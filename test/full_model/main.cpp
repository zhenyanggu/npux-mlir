#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm> // for std::max_element

// 引入 ONNX-MLIR 运行时头文件
#include "OnnxMlirRuntime.h"

// 声明入口函数
extern "C" OMTensorList *run_main_graph(OMTensorList *);

// ==========================================
// 1. 辅助函数
// ==========================================

// 读取二进制文件
std::vector<float> load_binary_file(const std::string& filename, int64_t expected_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }
    
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

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

// 获取最大值的索引 (ArgMax)
int get_argmax(const float* data, int64_t size) {
    return std::distance(data, std::max_element(data, data + size));
}

// 验证输出 (针对 VGG16 优化)
void verify_output(const float* actual, const std::vector<float>& golden, int64_t size) {
    float max_diff = 0.0f;
    float mse = 0.0f;
    
    // 计算误差
    for (int64_t i = 0; i < size; ++i) {
        float diff = std::abs(actual[i] - golden[i]);
        max_diff = std::max(max_diff, diff);
        mse += diff * diff;
    }
    mse /= size;

    // 获取预测类别
    int actual_class = get_argmax(actual, size);
    int golden_class = get_argmax(golden.data(), size);

    std::cout << "\n=== Verification Report (VGG16) ===" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Data Size          : " << size << " (Usually 1000 for ImageNet)" << std::endl;
    std::cout << "Max Absolute Error : " << max_diff << std::endl;
    std::cout << "Mean Squared Error : " << mse << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "NPU Predicted Class: " << actual_class << " (Value: " << actual[actual_class] << ")" << std::endl;
    std::cout << "Golden Class       : " << golden_class << " (Value: " << golden[golden_class] << ")" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    // 判定逻辑
    // 1. 类别必须一致
    // 2. 数值误差允许有一定的量化损失
    if (actual_class == golden_class) {
        std::cout << "✅ RESULT: PASS (Classification Match)" << std::endl;
    } else {
        std::cout << "❌ RESULT: FAIL (Classification Mismatch)" << std::endl;
        // 如果分类错了，但也可能是因为两个类别的概率非常接近（Top-5 check 更好，这里简化为 Top-1）
        std::cout << "   Note: Quantization noise might flip close probabilities." << std::endl;
    }
}

// ==========================================
// 2. 主函数
// ==========================================

int main(int argc, char** argv) {
    // 文件名配置
    std::string input_filename = "vgg16_input.bin";
    std::string golden_filename = "vgg16_golden.bin";

    // 如果命令行提供了文件名，则使用命令行的
    if (argc >= 2) input_filename = argv[1];
    if (argc >= 3) golden_filename = argv[2];

    // ------------------------------------------------
    // 1. 定义 VGG16 输入形状 (NCHW)
    // ------------------------------------------------
    int64_t batch = 1;
    int64_t channels = 3;
    int64_t height = 224;
    int64_t width = 224;
    
    int64_t input_shape[] = {batch, channels, height, width};
    int64_t rank = 4;
    int64_t input_elements = batch * channels * height * width; // 150528

    // ------------------------------------------------
    // 2. 加载数据
    // ------------------------------------------------
    std::cout << "[Runner] Loading input: " << input_filename << " (" << input_elements << " floats)" << std::endl;
    std::vector<float> input_data = load_binary_file(input_filename, input_elements);

    // ------------------------------------------------
    // 3. 构造 OMTensor
    // ------------------------------------------------
    OMTensor *inputTensors[1];
    
    // 注意：onnx-mlir 的入口通常接受 float 输入，
    // 即使模型内部量化了，输入层通常会插入 QuantizeLinear 算子来处理 float->int8
    OMTensor *tensor = omTensorCreate(input_data.data(), input_shape, rank, ONNX_TYPE_FLOAT);
    inputTensors[0] = tensor;
    
    OMTensorList *tensorListIn = omTensorListCreate(inputTensors, 1);

    // ------------------------------------------------
    // 4. 执行推理
    // ------------------------------------------------
    std::cout << "[Runner] Running VGG16 inference..." << std::endl;
    OMTensorList *tensorListOut = run_main_graph(tensorListIn);

    if (!tensorListOut) {
        std::cerr << "Error: Inference failed." << std::endl;
        return 1;
    }

    // ------------------------------------------------
    // 5. 解析输出
    // ------------------------------------------------
    // VGG16 只有一个输出 Tensor
    OMTensor *output_tensor = omTensorListGetOmtByIndex(tensorListOut, 0);
    float *output_data = (float *)omTensorGetDataPtr(output_tensor);
    
    // 获取输出形状
    const int64_t *out_shape_ptr = omTensorGetShape(output_tensor);
    int64_t out_rank = omTensorGetRank(output_tensor);
    int64_t total_output_elements = 1;

    std::cout << "[Runner] Output Shape: [";
    for(int i=0; i<out_rank; i++) {
        std::cout << out_shape_ptr[i] << (i < out_rank - 1 ? " x " : "");
        total_output_elements *= out_shape_ptr[i];
    }
    std::cout << "]" << std::endl;

    // ------------------------------------------------
    // 6. 验证结果
    // ------------------------------------------------
    std::cout << "[Runner] Verifying against: " << golden_filename << std::endl;
    // 确保 Golden 文件存在，否则可以跳过验证只打印结果
    std::ifstream f(golden_filename.c_str());
    if (f.good()) {
        std::vector<float> golden_data = load_binary_file(golden_filename, total_output_elements);
        verify_output(output_data, golden_data, total_output_elements);
    } else {
        std::cout << "⚠️ Golden file not found. Skipping verification." << std::endl;
        int pred_class = get_argmax(output_data, total_output_elements);
        std::cout << "Predicted Class: " << pred_class << std::endl;
    }

    // ------------------------------------------------
    // 7. 内存清理
    // ------------------------------------------------
    omTensorListDestroy(tensorListIn);
    omTensorListDestroy(tensorListOut);

    return 0;
}