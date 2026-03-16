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

struct VerificationSummary {
  float maxAbsError;
  double mse;
  int64_t errorCount;
  int64_t totalCount;
  bool passed;
};

struct ErrorItem {
  int64_t index;
  float actual;
  float golden;
  float absDiff;
};

bool endsWith(const std::string &value, const std::string &suffix) {
  if (value.size() < suffix.size())
    return false;
  return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string modelPrefixFromExecutable(const fs::path &exePath) {
  std::string name = exePath.filename().string();
  const std::string suffix = "_zcu102";
  if (endsWith(name, suffix))
    name.resize(name.size() - suffix.size());
  return name.empty() ? "model" : name;
}

std::string resolveDataFile(const fs::path &exeDir, const std::string &modelPrefix,
    const std::string &logicalName) {
  const fs::path prefixed = exeDir / (modelPrefix + "_" + logicalName);
  if (fs::exists(prefixed))
    return prefixed.string();
  const fs::path legacy = exeDir / logicalName;
  if (fs::exists(legacy))
    return legacy.string();
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
    std::cerr << "Error: file size mismatch for " << filename << ", expected: "
              << expectedBytes << ", got: " << bytes << std::endl;
    std::exit(1);
  }
  std::vector<float> data(expectedElements);
  if (!file.read(reinterpret_cast<char *>(data.data()), bytes)) {
    std::cerr << "Error: failed to read file: " << filename << std::endl;
    std::exit(1);
  }
  return data;
}

void printErrorDetails(const std::vector<ErrorItem> &errors, int64_t count, int64_t topK = 10) {
  std::cout << "\n=== Verification Report ===" << std::endl;
  std::cout << std::setw(12) << "index" << std::setw(16) << "actual"
            << std::setw(16) << "golden" << std::setw(16) << "abs diff" << std::endl;

  if (count <= 128) {
    for (const auto &item : errors) {
      std::cout << std::setw(12) << item.index << std::setw(16) << item.actual
                << std::setw(16) << item.golden << std::setw(16) << item.absDiff << std::endl;
    }
    std::cout << "Displayed all " << count << " elements." << std::endl;
    return;
  }

  const int64_t k = std::min<int64_t>(topK, count);
  std::vector<ErrorItem> sorted = errors;
  std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(),
      [](const ErrorItem &a, const ErrorItem &b) {
        if (a.absDiff == b.absDiff)
          return a.index < b.index;
        return a.absDiff > b.absDiff;
      });

  std::cout << "Top-" << k << " largest absolute errors:" << std::endl;
  for (int64_t i = 0; i < k; ++i) {
    const auto &item = sorted[static_cast<size_t>(i)];
    std::cout << std::setw(12) << item.index << std::setw(16) << item.actual
              << std::setw(16) << item.golden << std::setw(16) << item.absDiff << std::endl;
  }
}

VerificationSummary compareOutputs(const float *actual,
    const std::vector<float> &golden, int64_t count, float threshold = 0.1f) {
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

  printErrorDetails(errors, count);
  std::cout << "Max Absolute Error: " << maxAbsError << std::endl;
  std::cout << "Mean Squared Error: " << mse << std::endl;
  std::cout << "Error Count (错误点数量/总点数量): " << errorCount << "/" << count
            << std::endl;
  const bool passed = (errorCount == 0);
  std::cout << "Threshold Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return {maxAbsError, mse, errorCount, count, passed};
}

void printFinalSummary(const VerificationSummary &summary) {
  std::cout << "@@MODEL_TEST_RESULT@@ errors=" << summary.errorCount << "/"
            << summary.totalCount << " status="
            << (summary.passed ? "PASS" : "FAIL") << std::endl;
}

void verifyExpectedShape(const int64_t *shape, int64_t rank, const std::vector<int64_t> &expected) {
  bool ok = (rank == static_cast<int64_t>(expected.size()));
  if (ok) {
    for (int64_t i = 0; i < rank; ++i) {
      if (shape[i] != expected[static_cast<size_t>(i)]) {
        ok = false;
        break;
      }
    }
  }
  std::cout << "[VGG16 Layer Case Shape Check] " << (ok ? "PASS" : "WARNING") << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  const std::vector<int64_t> inputShape = {1, 3, 224, 224};
  const std::vector<int64_t> expectedOutputShape = {1, 64, 224, 224};

  int64_t inputElements = 1;
  for (int64_t d : inputShape)
    inputElements *= d;

  const fs::path exePath = (argc > 0) ? fs::path(argv[0]) : fs::path();
  const fs::path exeDir = exePath.has_parent_path() ? exePath.parent_path() : fs::path(".");
  const std::string modelPrefix = modelPrefixFromExecutable(exePath);

  const std::string inputFile = resolveDataFile(exeDir, modelPrefix, "input.bin");
  const std::string goldenFile = resolveDataFile(exeDir, modelPrefix, "output_golden.bin");

  std::vector<float> inputData = loadBinaryFloatFile(inputFile, inputElements);
  OMTensor *inputTensor = omTensorCreate(
      inputData.data(), const_cast<int64_t *>(inputShape.data()),
      static_cast<int64_t>(inputShape.size()), ONNX_TYPE_FLOAT);
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
  const int64_t *outputShape = omTensorGetShape(outputTensor);
  const int64_t outputRank = omTensorGetRank(outputTensor);

  int64_t outputElements = 1;
  for (int64_t i = 0; i < outputRank; ++i)
    outputElements *= outputShape[i];

  std::vector<float> golden = loadBinaryFloatFile(goldenFile, outputElements);
  const VerificationSummary summary = compareOutputs(outputData, golden, outputElements);
  verifyExpectedShape(outputShape, outputRank, expectedOutputShape);

  omTensorListDestroy(inputList);
  omTensorListDestroy(outputList);
  printFinalSummary(summary);
  return 0;
}
