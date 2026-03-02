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

struct DiffItem {
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
    std::cerr << "Error: file size mismatch for " << filename
              << ", expected: " << expectedBytes << ", got: " << bytes << std::endl;
    std::exit(1);
  }

  std::vector<float> data(expectedElements);
  if (!file.read(reinterpret_cast<char *>(data.data()), bytes)) {
    std::cerr << "Error: failed to read file: " << filename << std::endl;
    std::exit(1);
  }

  return data;
}

void printErrorDetails(const std::vector<DiffItem> &diffs, int64_t count) {
  std::cout << "\n=== Verification Report ===" << std::endl;
  std::cout << std::setw(8) << "index" << std::setw(16) << "actual"
            << std::setw(16) << "golden" << std::setw(16) << "abs diff"
            << std::endl;

  if (count <= 64) {
    for (const auto &item : diffs) {
      std::cout << std::setw(8) << item.index << std::setw(16) << item.actual
                << std::setw(16) << item.golden << std::setw(16)
                << item.absDiff << std::endl;
    }
    std::cout << "Displayed all " << count << " elements." << std::endl;
    return;
  }

  const size_t topK = std::min<size_t>(10, diffs.size());
  std::vector<DiffItem> top = diffs;
  std::partial_sort(top.begin(), top.begin() + topK, top.end(),
      [](const DiffItem &a, const DiffItem &b) {
        return a.absDiff > b.absDiff;
      });

  for (size_t i = 0; i < topK; ++i) {
    const auto &item = top[i];
    std::cout << std::setw(8) << item.index << std::setw(16) << item.actual
              << std::setw(16) << item.golden << std::setw(16)
              << item.absDiff << std::endl;
  }

  std::cout << "Displayed Top-" << topK
            << " largest-abs-diff elements out of " << count << "."
            << std::endl;
}

void verifyLayerNormStats(const float *data, int64_t dim0, int64_t dim1, int64_t dim2,
    const std::string &tag) {
  const double expectedMean = -1e-3;
  double maxMeanDeviation = 0.0;
  double maxVarDeviation = 0.0;

  for (int64_t n = 0; n < dim0; ++n) {
    for (int64_t s = 0; s < dim1; ++s) {
      const int64_t base = (n * dim1 + s) * dim2;
      double sum = 0.0;
      for (int64_t h = 0; h < dim2; ++h)
        sum += data[base + h];
      const double mean = sum / static_cast<double>(dim2);

      double varSum = 0.0;
      for (int64_t h = 0; h < dim2; ++h) {
        const double d = data[base + h] - mean;
        varSum += d * d;
      }
      const double var = varSum / static_cast<double>(dim2);

      maxMeanDeviation = std::max(maxMeanDeviation, std::abs(mean - expectedMean));
      maxVarDeviation = std::max(maxVarDeviation, std::abs(var - 1.0));
    }
  }

  std::cout << "[LayerNorm Stats - " << tag << "] "
            << "max|mean-(-1e-3)|=" << maxMeanDeviation
            << ", max|var-1|=" << maxVarDeviation << std::endl;
}

void compareOutputs(const float *actual, const std::vector<float> &golden,
    int64_t count, float threshold = 0.1f) {
  float maxAbsError = 0.0f;
  double mse = 0.0;
  int64_t errorCount = 0;
  std::vector<DiffItem> diffs;
  diffs.reserve(static_cast<size_t>(count));

  for (int64_t i = 0; i < count; ++i) {
    const float diff = std::abs(actual[i] - golden[static_cast<size_t>(i)]);
    maxAbsError = std::max(maxAbsError, diff);
    mse += static_cast<double>(diff) * static_cast<double>(diff);
    if (diff > threshold)
      ++errorCount;
    diffs.push_back({i, actual[i], golden[static_cast<size_t>(i)], diff});
  }

  mse /= static_cast<double>(count);

  printErrorDetails(diffs, count);

  std::cout << "Max Absolute Error: " << maxAbsError << std::endl;
  std::cout << "Mean Squared Error: " << mse << std::endl;
  std::cout << "Error Count (错误点数量/总点数量): " << errorCount << "/" << count
            << std::endl;

  if (maxAbsError < threshold) {
    std::cout << "RESULT: PASS" << std::endl;
  } else {
    std::cout << "RESULT: WARNING (max error exceeds threshold)" << std::endl;
  }
}

} // namespace

int main(int argc, char **argv) {
  constexpr int64_t dim0 = 1;
  constexpr int64_t dim1 = 128;
  constexpr int64_t dim2 = 768;
  constexpr int64_t inputElements = dim0 * dim1 * dim2;

  const fs::path exePath = (argc > 0) ? fs::path(argv[0]) : fs::path();
  const fs::path exeDir = exePath.has_parent_path() ? exePath.parent_path() : fs::path(".");
  const std::string modelPrefix = modelPrefixFromExecutable(exePath);

  const std::string inputFile = resolveDataFile(exeDir, modelPrefix, "input.bin");
  const std::string goldenFile = resolveDataFile(exeDir, modelPrefix, "output_golden.bin");

  std::cout << "Model prefix: " << modelPrefix << std::endl;
  std::cout << "Input file: " << inputFile << std::endl;
  std::cout << "Golden file: " << goldenFile << std::endl;

  std::vector<float> inputData = loadBinaryFloatFile(inputFile, inputElements);

  int64_t inputShape[] = {dim0, dim1, dim2};
  OMTensor *inputTensor = omTensorCreate(inputData.data(), inputShape, 3, ONNX_TYPE_FLOAT);
  if (!inputTensor) {
    std::cerr << "Error: failed to create input tensor" << std::endl;
    return 1;
  }

  OMTensor *inputs[] = {inputTensor};
  OMTensorList *inputList = omTensorListCreate(inputs, 1);
  if (!inputList) {
    std::cerr << "Error: failed to create input tensor list" << std::endl;
    return 1;
  }

  OMTensorList *outputList = run_main_graph(inputList);
  if (!outputList) {
    std::cerr << "Error: inference returned null output" << std::endl;
    omTensorListDestroy(inputList);
    return 1;
  }

  OMTensor *outputTensor = omTensorListGetOmtByIndex(outputList, 0);
  if (!outputTensor) {
    std::cerr << "Error: output tensor missing" << std::endl;
    omTensorListDestroy(inputList);
    omTensorListDestroy(outputList);
    return 1;
  }

  float *outputData = reinterpret_cast<float *>(omTensorGetDataPtr(outputTensor));
  const int64_t *outputShape = omTensorGetShape(outputTensor);
  const int64_t outputRank = omTensorGetRank(outputTensor);

  int64_t outputElements = 1;
  std::cout << "Output shape: [";
  for (int64_t i = 0; i < outputRank; ++i) {
    outputElements *= outputShape[i];
    std::cout << outputShape[i] << (i + 1 == outputRank ? "" : ", ");
  }
  std::cout << "]" << std::endl;

  std::vector<float> golden = loadBinaryFloatFile(goldenFile, outputElements);
  compareOutputs(outputData, golden, outputElements);

  if (outputRank == 3 && outputShape[0] == dim0 && outputShape[1] == dim1 && outputShape[2] == dim2) {
    verifyLayerNormStats(outputData, dim0, dim1, dim2, "actual");
    verifyLayerNormStats(golden.data(), dim0, dim1, dim2, "golden");
  }

  omTensorListDestroy(inputList);
  omTensorListDestroy(outputList);
  return 0;
}
