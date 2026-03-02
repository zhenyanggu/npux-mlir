#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "OnnxMlirRuntime.h"

extern "C" OMTensorList *run_main_graph(OMTensorList *);

namespace {
namespace fs = std::filesystem;

struct ErrorItem { int64_t index; float actual; float golden; float absDiff; };

bool endsWith(const std::string &value, const std::string &suffix) {
  if (value.size() < suffix.size()) return false;
  return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string modelPrefixFromExecutable(const fs::path &exePath) {
  std::string name = exePath.filename().string();
  const std::string suffix = "_zcu102";
  if (endsWith(name, suffix)) name.resize(name.size() - suffix.size());
  return name.empty() ? "model" : name;
}

std::string resolveDataFile(const fs::path &exeDir, const std::string &prefix, const std::string &logical) {
  const fs::path prefixed = exeDir / (prefix + "_" + logical);
  if (fs::exists(prefixed)) return prefixed.string();
  const fs::path legacy = exeDir / logical;
  if (fs::exists(legacy)) return legacy.string();
  return prefixed.string();
}

std::vector<float> loadBinaryFloatFile(const std::string &filename, int64_t expectedElements) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: cannot open file: " << filename << std::endl;
    std::exit(1);
  }
  file.seekg(0, std::ios::end);
  const std::streamsize bytes = file.tellg();
  file.seekg(0, std::ios::beg);
  const auto expectedBytes = static_cast<std::streamsize>(expectedElements * sizeof(float));
  if (bytes != expectedBytes) {
    std::cerr << "Error: file size mismatch for " << filename << ", expected " << expectedBytes
              << ", got " << bytes << std::endl;
    std::exit(1);
  }
  std::vector<float> data(expectedElements);
  if (!file.read(reinterpret_cast<char *>(data.data()), bytes)) {
    std::cerr << "Error: failed to read file: " << filename << std::endl;
    std::exit(1);
  }
  return data;
}

void compareOutputs(const float *actual, const std::vector<float> &golden, int64_t count,
    float threshold = 0.1f, int64_t topK = 10) {
  float maxAbsError = 0.0f;
  double mse = 0.0;
  int64_t errorCount = 0;
  std::vector<ErrorItem> errors;
  errors.reserve(static_cast<size_t>(count));

  for (int64_t i = 0; i < count; ++i) {
    const float diff = std::abs(actual[i] - golden[static_cast<size_t>(i)]);
    maxAbsError = std::max(maxAbsError, diff);
    mse += static_cast<double>(diff) * static_cast<double>(diff);
    if (diff > threshold)
      ++errorCount;
    errors.push_back({i, actual[i], golden[static_cast<size_t>(i)], diff});
  }
  mse /= static_cast<double>(count);

  std::cout << "\n=== Verification Report ===\n";
  std::cout << std::setw(12) << "index" << std::setw(16) << "actual" << std::setw(16)
            << "golden" << std::setw(16) << "abs diff" << std::endl;

  if (count <= 128) {
    for (const auto &item : errors) {
      std::cout << std::setw(12) << item.index << std::setw(16) << item.actual
                << std::setw(16) << item.golden << std::setw(16) << item.absDiff << std::endl;
    }
  } else {
    const int64_t k = std::min<int64_t>(topK, count);
    std::partial_sort(errors.begin(), errors.begin() + k, errors.end(),
        [](const ErrorItem &a, const ErrorItem &b) { return a.absDiff > b.absDiff; });
    for (int64_t i = 0; i < k; ++i) {
      const auto &item = errors[static_cast<size_t>(i)];
      std::cout << std::setw(12) << item.index << std::setw(16) << item.actual
                << std::setw(16) << item.golden << std::setw(16) << item.absDiff << std::endl;
    }
  }

  std::cout << "Max Absolute Error: " << maxAbsError << std::endl;
  std::cout << "Mean Squared Error: " << mse << std::endl;
  std::cout << "Error Count (错误点数量/总点数量): " << errorCount << "/" << count
            << std::endl;
  std::cout << (maxAbsError < threshold ? "RESULT: PASS" : "RESULT: WARNING (max error exceeds threshold)")
            << std::endl;
}

void verifyShape(const int64_t *shape, int64_t rank) {
  bool ok = rank == 2 && shape[0] == 128 && shape[1] == 1024;
  std::cout << "[Gemm Shape Check] " << (ok ? "PASS" : "WARNING") << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  constexpr int64_t d0 = 128, d1 = 768;
  constexpr int64_t inputElements = d0 * d1;

  const fs::path exePath = (argc > 0) ? fs::path(argv[0]) : fs::path();
  const fs::path exeDir = exePath.has_parent_path() ? exePath.parent_path() : fs::path(".");
  const std::string prefix = modelPrefixFromExecutable(exePath);

  const std::string inputFile = resolveDataFile(exeDir, prefix, "input.bin");
  const std::string goldenFile = resolveDataFile(exeDir, prefix, "output_golden.bin");

  std::vector<float> inputData = loadBinaryFloatFile(inputFile, inputElements);

  int64_t inputShape[] = {d0, d1};
  OMTensor *inputTensor = omTensorCreate(inputData.data(), inputShape, 2, ONNX_TYPE_FLOAT);
  OMTensor *inputs[] = {inputTensor};
  OMTensorList *inputList = omTensorListCreate(inputs, 1);
  OMTensorList *outputList = run_main_graph(inputList);
  if (!outputList) {
    std::cerr << "Error: inference failed" << std::endl;
    omTensorListDestroy(inputList);
    return 1;
  }

  OMTensor *outputTensor = omTensorListGetOmtByIndex(outputList, 0);
  float *outputData = reinterpret_cast<float *>(omTensorGetDataPtr(outputTensor));
  const int64_t *shape = omTensorGetShape(outputTensor);
  const int64_t rank = omTensorGetRank(outputTensor);

  int64_t outputElements = 1;
  for (int64_t i = 0; i < rank; ++i) outputElements *= shape[i];

  std::vector<float> golden = loadBinaryFloatFile(goldenFile, outputElements);
  compareOutputs(outputData, golden, outputElements);
  verifyShape(shape, rank);

  omTensorListDestroy(inputList);
  omTensorListDestroy(outputList);
  return 0;
}
