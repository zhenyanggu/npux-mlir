#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>

// 引入 ONNX-MLIR 运行时头文件
#include "OnnxMlirRuntime.h"

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

        // 打印前10个或者误差较大的值用于调试
        if (i < 10) {
            std::cout << std::setw(10) << i 
                      << std::setw(15) << actual[i] 
                      << std::setw(15) << golden[i] 
                      << std::setw(15) << diff << std::endl;
        }
    }
    mse /= size;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Max Absolute Error: " << max_diff << std::endl;
    std::cout << "Mean Squared Error: " << mse << std::endl;

    // 判定阈值 (根据业务需求调整，量化模型通常允许 1e-2 到 1e-4 的误差)
    if (max_diff < 1e-3) { 
        std::cout << "✅ RESULT: PASS (Consistent)" << std::endl;
    } else {
        std::cout << "⚠️ RESULT: WARNING (High Deviation)" << std::endl;
    }
}

int main() {
    // 1. 定义形状
    int64_t batch = 1;
    int64_t channels = 1;
    int64_t height = 8;
    int64_t width = 8;
    int64_t input_elements = batch * channels * height * width;

    // 2. 加载输入数据 (从 Python 生成的 bin 文件)
    std::cout << "Loading input.bin..." << std::endl;
    std::vector<float> input_data = load_binary_file("input.bin", input_elements);

    // 3. 构造 OMTensor
    OMTensor *inputTensors[1];
    int64_t input_shape[] = {batch, channels, height, width};
    int64_t rank = 4;
    
    OMTensor *tensor = omTensorCreate(input_data.data(), input_shape, rank, ONNX_TYPE_FLOAT);
    inputTensors[0] = tensor;
    OMTensorList *tensorListIn = omTensorListCreate(inputTensors, 1);

    // 4. 执行推理
    std::cout << "Running inference on NPU/Model..." << std::endl;
    OMTensorList *tensorListOut = run_main_graph(tensorListIn);

    // 5. 获取输出
    OMTensor *output_tensor = omTensorListGetOmtByIndex(tensorListOut, 0);
    float *output_data = (float *)omTensorGetDataPtr(output_tensor);
    
    // 验证输出形状
    // 你的模型 FullOpsModel 结尾是 fc2(16->10) -> softmax
    // 所以输出形状应该是 [1, 10]
    const int64_t *out_shape_ptr = omTensorGetShape(output_tensor);
    int64_t out_rank = omTensorGetRank(output_tensor);
    int64_t total_output_elements = 1;
    
    std::cout << "Output Shape: [";
    for(int i=0; i<out_rank; i++) {
        std::cout << out_shape_ptr[i] << (i < out_rank - 1 ? " x " : "");
        total_output_elements *= out_shape_ptr[i];
    }
    std::cout << "]" << std::endl;

    // 6. 加载 Golden Data 并进行比对
    std::cout << "Loading output_golden.bin for verification..." << std::endl;
    // 注意：如果文件不存在，程序会在此处报错退出
    std::vector<float> golden_data = load_binary_file("output_golden.bin", total_output_elements);
    
    verify_output(output_data, golden_data, total_output_elements);

    // 7. 清理
    omTensorListDestroy(tensorListIn); // 注意：input_data vector 自动析构
    omTensorListDestroy(tensorListOut);

    return 0;
}