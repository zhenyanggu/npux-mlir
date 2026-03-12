#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

constexpr int64_t kN = 1;
constexpr int64_t kC = 512;
constexpr int64_t kH = 7;
constexpr int64_t kW = 7;
constexpr int64_t kOutC = 64;

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

std::string resolveDataFile(
    const fs::path &exeDir, const std::string &prefix, const std::string &logical) {
  const fs::path prefixed = exeDir / (prefix + "_" + logical);
  if (fs::exists(prefixed))
    return prefixed.string();
  const fs::path legacy = exeDir / logical;
  if (fs::exists(legacy))
    return legacy.string();
  return prefixed.string();
}

std::vector<float> loadFloatBin(const std::string &path, int64_t expectedElements) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: cannot open file: " << path << std::endl;
    std::exit(1);
  }
  file.seekg(0, std::ios::end);
  const std::streamsize bytes = file.tellg();
  file.seekg(0, std::ios::beg);
  const auto expectedBytes = static_cast<std::streamsize>(expectedElements * sizeof(float));
  if (bytes != expectedBytes) {
    std::cerr << "Error: file size mismatch for " << path << ", expected "
              << expectedBytes << ", got " << bytes << std::endl;
    std::exit(1);
  }
  std::vector<float> data(expectedElements);
  if (!file.read(reinterpret_cast<char *>(data.data()), bytes)) {
    std::cerr << "Error: failed to read file: " << path << std::endl;
    std::exit(1);
  }
  return data;
}

float envToFloat(const char *name, float defaultValue) {
  const char *value = std::getenv(name);
  if (!value || *value == '\0')
    return defaultValue;
  try {
    return std::stof(value);
  } catch (...) {
    return defaultValue;
  }
}

VerificationSummary compareOutputs(
    const float *actual, const std::vector<float> &golden, int64_t count, float threshold) {
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

  const int64_t topK = std::min<int64_t>(10, count);
  std::partial_sort(errors.begin(), errors.begin() + topK, errors.end(),
      [](const ErrorItem &a, const ErrorItem &b) { return a.absDiff > b.absDiff; });

  std::cout << "\n=== Verification Report ===\n";
  std::cout << std::setw(12) << "index" << std::setw(16) << "actual" << std::setw(16)
            << "golden" << std::setw(16) << "abs diff" << std::endl;
  for (int64_t i = 0; i < topK; ++i) {
    const auto &item = errors[static_cast<size_t>(i)];
    std::cout << std::setw(12) << item.index << std::setw(16) << item.actual
              << std::setw(16) << item.golden << std::setw(16) << item.absDiff
              << std::endl;
  }

  const bool passed = (errorCount == 0);
  std::cout << "Max Absolute Error: " << maxAbsError << std::endl;
  std::cout << "Mean Squared Error: " << mse << std::endl;
  std::cout << "Error Count (错误点数量/总点数量): " << errorCount << "/" << count
            << std::endl;
  std::cout << "Threshold Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return {maxAbsError, mse, errorCount, count, passed};
}

void printFinalSummary(const VerificationSummary &summary) {
  std::cout << "@@MODEL_TEST_RESULT@@ errors=" << summary.errorCount << "/"
            << summary.totalCount << " status="
            << (summary.passed ? "PASS" : "FAIL") << std::endl;
}

void verifyShape(const int64_t *shape, int64_t rank) {
  const bool ok = rank == 4 && shape[0] == kN && shape[1] == kOutC && shape[2] == kH &&
      shape[3] == kW;
  std::cout << "[Shape Check] " << (ok ? "PASS" : "WARNING") << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  const fs::path exePath = (argc > 0) ? fs::path(argv[0]) : fs::path();
  const fs::path exeDir = exePath.has_parent_path() ? exePath.parent_path() : fs::path(".");
  const std::string prefix = modelPrefixFromExecutable(exePath);

  const std::string inputFile = resolveDataFile(exeDir, prefix, "input.bin");
  const std::string goldenFile = resolveDataFile(exeDir, prefix, "output_golden.bin");
  const auto input = loadFloatBin(inputFile, kN * kC * kH * kW);
  const auto golden = loadFloatBin(goldenFile, kN * kOutC * kH * kW);

  int64_t inputShape[] = {kN, kC, kH, kW};
  OMTensor *inputTensor = omTensorCreate(
      const_cast<float *>(input.data()), inputShape, 4, ONNX_TYPE_FLOAT);
  OMTensor *inputs[] = {inputTensor};
  OMTensorList *inputList = omTensorListCreate(inputs, 1);
  OMTensorList *outputList = run_main_graph(inputList);
  if (!outputList) {
    std::cerr << "Error: inference failed" << std::endl;
    omTensorListDestroy(inputList);
    return 1;
  }

  OMTensor *outputTensor = omTensorListGetOmtByIndex(outputList, 0);
  if (!outputTensor || !omTensorGetDataPtr(outputTensor)) {
    std::cerr << "Error: output tensor is invalid" << std::endl;
    omTensorListDestroy(inputList);
    omTensorListDestroy(outputList);
    return 1;
  }

  const int64_t *shape = omTensorGetShape(outputTensor);
  const int64_t rank = omTensorGetRank(outputTensor);
  float *output = reinterpret_cast<float *>(omTensorGetDataPtr(outputTensor));

  const auto summary = compareOutputs(
      output, golden, kN * kOutC * kH * kW, envToFloat("MODEL_TEST_THRESHOLD", 0.1f));
  verifyShape(shape, rank);

  omTensorListDestroy(inputList);
  omTensorListDestroy(outputList);
  printFinalSummary(summary);
  return summary.passed ? 0 : 1;
}
