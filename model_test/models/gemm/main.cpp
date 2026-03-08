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

struct ErrorItem { int64_t index; float actual; float golden; float absDiff; };

int64_t envToInt64(const char *name, int64_t defaultValue) {
  const char *v = std::getenv(name);
  if (!v || *v == '\0') return defaultValue;
  try {
    return std::stoll(v);
  } catch (...) {
    return defaultValue;
  }
}

float envToFloat(const char *name, float defaultValue) {
  const char *v = std::getenv(name);
  if (!v || *v == '\0') return defaultValue;
  try {
    return std::stof(v);
  } catch (...) {
    return defaultValue;
  }
}

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

VerificationSummary compareOutputs(const float *actual,
  const std::vector<float> &golden, int64_t count,
  int64_t rows,
  int64_t cols,
  float threshold = 0.1f,
  int64_t topK = 10,
  bool printLayoutStats = true) {
  float maxAbsError = 0.0f;
  double mse = 0.0;
  int64_t errorCount = 0;
  std::vector<ErrorItem> errors;
  errors.reserve(static_cast<size_t>(count));

  std::vector<int64_t> rowErrorCounts;
  std::vector<int64_t> colBlockErrorCounts;
  std::vector<int64_t> laneErrorCounts;
  if (rows > 0 && cols > 0) {
    rowErrorCounts.assign(static_cast<size_t>(rows), 0);
    colBlockErrorCounts.assign(static_cast<size_t>((cols + 31) / 32), 0);
    laneErrorCounts.assign(32, 0);
  }

  for (int64_t i = 0; i < count; ++i) {
    const float diff = std::abs(actual[i] - golden[static_cast<size_t>(i)]);
    maxAbsError = std::max(maxAbsError, diff);
    mse += static_cast<double>(diff) * static_cast<double>(diff);
    if (diff > threshold) {
      ++errorCount;

      if (rows > 0 && cols > 0 && i < rows * cols) {
        const int64_t r = i / cols;
        const int64_t c = i % cols;
        rowErrorCounts[static_cast<size_t>(r)]++;
        colBlockErrorCounts[static_cast<size_t>(c / 32)]++;
        laneErrorCounts[static_cast<size_t>(c % 32)]++;
      }
    }
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

  if (printLayoutStats && errorCount > 0 && rows > 0 && cols > 0) {
    std::vector<std::pair<int64_t, int64_t>> rowRank;
    rowRank.reserve(static_cast<size_t>(rows));
    for (int64_t r = 0; r < rows; ++r)
      rowRank.push_back({r, rowErrorCounts[static_cast<size_t>(r)]});
    std::partial_sort(
        rowRank.begin(),
        rowRank.begin() + std::min<int64_t>(5, static_cast<int64_t>(rowRank.size())),
        rowRank.end(),
        [](const auto &a, const auto &b) { return a.second > b.second; });

    std::vector<std::pair<int64_t, int64_t>> blockRank;
    blockRank.reserve(colBlockErrorCounts.size());
    for (int64_t b = 0; b < static_cast<int64_t>(colBlockErrorCounts.size()); ++b)
      blockRank.push_back({b, colBlockErrorCounts[static_cast<size_t>(b)]});
    std::partial_sort(
        blockRank.begin(),
        blockRank.begin() + std::min<int64_t>(8, static_cast<int64_t>(blockRank.size())),
        blockRank.end(),
        [](const auto &a, const auto &b) { return a.second > b.second; });

    std::vector<std::pair<int64_t, int64_t>> laneRank;
    laneRank.reserve(32);
    for (int64_t lane = 0; lane < 32; ++lane)
      laneRank.push_back({lane, laneErrorCounts[static_cast<size_t>(lane)]});
    std::partial_sort(
        laneRank.begin(),
        laneRank.begin() + std::min<int64_t>(8, static_cast<int64_t>(laneRank.size())),
        laneRank.end(),
        [](const auto &a, const auto &b) { return a.second > b.second; });

    std::cout << "[Error Layout] Top rows by error count:";
    for (int64_t i = 0; i < std::min<int64_t>(5, static_cast<int64_t>(rowRank.size())); ++i) {
      if (rowRank[static_cast<size_t>(i)].second == 0) break;
      std::cout << " r" << rowRank[static_cast<size_t>(i)].first
                << ":" << rowRank[static_cast<size_t>(i)].second;
    }
    std::cout << std::endl;

    std::cout << "[Error Layout] Top col blocks (block=col/32):";
    for (int64_t i = 0; i < std::min<int64_t>(8, static_cast<int64_t>(blockRank.size())); ++i) {
      if (blockRank[static_cast<size_t>(i)].second == 0) break;
      std::cout << " b" << blockRank[static_cast<size_t>(i)].first
                << ":" << blockRank[static_cast<size_t>(i)].second;
    }
    std::cout << std::endl;

    std::cout << "[Error Layout] Top lanes (lane=col%32):";
    for (int64_t i = 0; i < std::min<int64_t>(8, static_cast<int64_t>(laneRank.size())); ++i) {
      if (laneRank[static_cast<size_t>(i)].second == 0) break;
      std::cout << " l" << laneRank[static_cast<size_t>(i)].first
                << ":" << laneRank[static_cast<size_t>(i)].second;
    }
    std::cout << std::endl;
  }

  const bool passed = (errorCount == 0);
  std::cout << "Threshold Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return {maxAbsError, mse, errorCount, count, passed};
}

void printFinalSummary(const VerificationSummary &summary) {
  std::cout << "@@MODEL_TEST_RESULT@@ errors=" << summary.errorCount << "/"
            << summary.totalCount << " status="
            << (summary.passed ? "PASS" : "FAIL") << std::endl;
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

    const float threshold = envToFloat("GEMM_THRESHOLD", 0.1f);
    const int64_t topK = std::max<int64_t>(1, envToInt64("GEMM_TOPK", 10));
    const bool printLayoutStats = envToInt64("GEMM_LAYOUT_STATS", 1) != 0;

    std::cout << "[Verify Config] threshold=" << threshold
        << ", topK=" << topK
        << ", layoutStats=" << (printLayoutStats ? "on" : "off")
        << std::endl;

  std::vector<float> golden = loadBinaryFloatFile(goldenFile, outputElements);
    const int64_t rows = (rank >= 2) ? shape[0] : 0;
    const int64_t cols = (rank >= 2) ? shape[1] : 0;
    const VerificationSummary summary = compareOutputs(
      outputData, golden, outputElements, rows, cols, threshold, topK, printLayoutStats);
  verifyShape(shape, rank);

  omTensorListDestroy(inputList);
  omTensorListDestroy(outputList);
  printFinalSummary(summary);
  return 0;
}
