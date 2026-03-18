#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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
  int64_t n;
  int64_t c;
  int64_t h;
  int64_t w;
};

bool hasValidNchwShape(const std::vector<int64_t> &shape) {
  if (shape.size() != 4)
    return false;
  return shape[0] > 0 && shape[1] > 0 && shape[2] > 0 && shape[3] > 0;
}

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
  std::cout << std::setw(12) << "index" << std::setw(8) << "n" << std::setw(8) << "c"
            << std::setw(8) << "h" << std::setw(8) << "w" << std::setw(16) << "actual"
            << std::setw(16) << "golden" << std::setw(16) << "abs diff" << std::endl;

  if (count <= 128) {
    for (const auto &item : errors) {
      std::cout << std::setw(12) << item.index << std::setw(8) << item.n << std::setw(8)
                << item.c << std::setw(8) << item.h << std::setw(8) << item.w
                << std::setw(16) << item.actual << std::setw(16) << item.golden
                << std::setw(16) << item.absDiff << std::endl;
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
    std::cout << std::setw(12) << item.index << std::setw(8) << item.n << std::setw(8)
              << item.c << std::setw(8) << item.h << std::setw(8) << item.w
              << std::setw(16) << item.actual << std::setw(16) << item.golden
              << std::setw(16) << item.absDiff << std::endl;
  }
}

void printNchwErrorBreakdown(const std::vector<ErrorItem> &errors,
    const std::vector<int64_t> &shape, float threshold, int64_t topK = 10) {
  if (!hasValidNchwShape(shape)) {
    std::cout << "\n[NCHW Error Breakdown] skipped (output rank != 4)." << std::endl;
    return;
  }

  const int64_t n = shape[0];
  const int64_t c = shape[1];
  const int64_t h = shape[2];
  const int64_t w = shape[3];
  const int64_t hwSize = h * w;

  std::vector<int64_t> channelErrCount(static_cast<size_t>(c), 0);
  std::vector<double> channelAbsSum(static_cast<size_t>(c), 0.0);
  std::vector<float> channelMaxAbs(static_cast<size_t>(c), 0.0f);
  std::vector<int64_t> spatialErrCount(static_cast<size_t>(hwSize), 0);
  std::vector<double> spatialAbsSum(static_cast<size_t>(hwSize), 0.0);

  int64_t thresholdErrCount = 0;
  for (const auto &item : errors) {
    if (item.absDiff <= threshold)
      continue;
    if (item.c < 0 || item.c >= c || item.h < 0 || item.h >= h || item.w < 0 || item.w >= w)
      continue;
    ++thresholdErrCount;
    const size_t cIdx = static_cast<size_t>(item.c);
    const size_t hwIdx = static_cast<size_t>(item.h * w + item.w);
    ++channelErrCount[cIdx];
    channelAbsSum[cIdx] += item.absDiff;
    channelMaxAbs[cIdx] = std::max(channelMaxAbs[cIdx], item.absDiff);
    ++spatialErrCount[hwIdx];
    spatialAbsSum[hwIdx] += item.absDiff;
  }

  std::cout << "\n=== NCHW Error Breakdown (abs diff > " << threshold << ") ==="
            << std::endl;
  if (thresholdErrCount == 0) {
    std::cout << "No error points above threshold." << std::endl;
    return;
  }

  std::vector<int64_t> channelOrder(static_cast<size_t>(c));
  std::iota(channelOrder.begin(), channelOrder.end(), 0);
  const int64_t topChannelK = std::min<int64_t>(topK, c);
  std::partial_sort(channelOrder.begin(), channelOrder.begin() + topChannelK,
      channelOrder.end(), [&](int64_t a, int64_t b) {
        const int64_t ca = channelErrCount[static_cast<size_t>(a)];
        const int64_t cb = channelErrCount[static_cast<size_t>(b)];
        if (ca == cb)
          return a < b;
        return ca > cb;
      });

  std::cout << "Top-" << topChannelK << " channels by error count:" << std::endl;
  std::cout << std::setw(10) << "channel" << std::setw(14) << "err_count"
            << std::setw(14) << "err_ratio%" << std::setw(16) << "mean_abs"
            << std::setw(16) << "max_abs" << std::endl;
  for (int64_t i = 0; i < topChannelK; ++i) {
    const int64_t ch = channelOrder[static_cast<size_t>(i)];
    const int64_t cnt = channelErrCount[static_cast<size_t>(ch)];
    const double ratio = (thresholdErrCount > 0)
                             ? (100.0 * static_cast<double>(cnt) /
                                   static_cast<double>(thresholdErrCount))
                             : 0.0;
    const double meanAbs = (cnt > 0)
                               ? (channelAbsSum[static_cast<size_t>(ch)] /
                                     static_cast<double>(cnt))
                               : 0.0;
    std::cout << std::setw(10) << ch << std::setw(14) << cnt << std::setw(14)
              << ratio << std::setw(16) << meanAbs << std::setw(16)
              << channelMaxAbs[static_cast<size_t>(ch)] << std::endl;
  }

  auto printModPattern = [&](int modBase) {
    if (c < modBase)
      return;
    std::vector<int64_t> buckets(static_cast<size_t>(modBase), 0);
    for (int64_t ch = 0; ch < c; ++ch)
      buckets[static_cast<size_t>(ch % modBase)] +=
          channelErrCount[static_cast<size_t>(ch)];
    std::cout << "Channel error modulo " << modBase << ":";
    for (int i = 0; i < modBase; ++i)
      std::cout << " [" << i << "]=" << buckets[static_cast<size_t>(i)];
    std::cout << std::endl;
  };
  printModPattern(2);
  printModPattern(3);
  printModPattern(4);
  printModPattern(8);

  std::vector<int64_t> spatialOrder(static_cast<size_t>(hwSize));
  std::iota(spatialOrder.begin(), spatialOrder.end(), 0);
  const int64_t topSpatialK = std::min<int64_t>(topK, hwSize);
  std::partial_sort(spatialOrder.begin(), spatialOrder.begin() + topSpatialK,
      spatialOrder.end(), [&](int64_t a, int64_t b) {
        const int64_t ca = spatialErrCount[static_cast<size_t>(a)];
        const int64_t cb = spatialErrCount[static_cast<size_t>(b)];
        if (ca == cb)
          return a < b;
        return ca > cb;
      });

  std::cout << "Top-" << topSpatialK << " spatial positions by error count:"
            << std::endl;
  std::cout << std::setw(8) << "h" << std::setw(8) << "w" << std::setw(14)
            << "err_count" << std::setw(16) << "mean_abs" << std::setw(16)
            << "affected_ch%" << std::endl;
  for (int64_t i = 0; i < topSpatialK; ++i) {
    const int64_t hw = spatialOrder[static_cast<size_t>(i)];
    const int64_t hh = hw / w;
    const int64_t ww = hw % w;
    const int64_t cnt = spatialErrCount[static_cast<size_t>(hw)];
    const double meanAbs = (cnt > 0)
                               ? (spatialAbsSum[static_cast<size_t>(hw)] /
                                     static_cast<double>(cnt))
                               : 0.0;
    const double chRatio = (n * c > 0)
                               ? (100.0 * static_cast<double>(cnt) /
                                     static_cast<double>(n * c))
                               : 0.0;
    std::cout << std::setw(8) << hh << std::setw(8) << ww << std::setw(14)
              << cnt << std::setw(16) << meanAbs << std::setw(16) << chRatio
              << std::endl;
  }
}

VerificationSummary compareOutputs(const float *actual,
    const std::vector<float> &golden, int64_t count,
    const std::vector<int64_t> &outputShape, float threshold = 0.1f) {
  float maxAbsError = 0.0f;
  double mse = 0.0;
  int64_t errorCount = 0;
  std::vector<ErrorItem> errors;
  errors.reserve(static_cast<size_t>(count));
  const bool hasNchw = hasValidNchwShape(outputShape);
  const int64_t cVal = hasNchw ? outputShape[1] : -1;
  const int64_t hVal = hasNchw ? outputShape[2] : -1;
  const int64_t wVal = hasNchw ? outputShape[3] : -1;
  const int64_t chw = hasNchw ? (cVal * hVal * wVal) : -1;
  const int64_t hw = hasNchw ? (hVal * wVal) : -1;

  for (int64_t i = 0; i < count; ++i) {
    const float diff = std::abs(actual[i] - golden[static_cast<size_t>(i)]);
    maxAbsError = std::max(maxAbsError, diff);
    mse += static_cast<double>(diff) * static_cast<double>(diff);
    if (diff > threshold)
      ++errorCount;
    int64_t nIdx = -1;
    int64_t cIdx = -1;
    int64_t hIdx = -1;
    int64_t wIdx = -1;
    if (hasNchw && chw > 0 && hw > 0) {
      nIdx = i / chw;
      const int64_t rem0 = i % chw;
      cIdx = rem0 / hw;
      const int64_t rem1 = rem0 % hw;
      hIdx = rem1 / wVal;
      wIdx = rem1 % wVal;
    }
    errors.push_back(
        {i, actual[i], golden[static_cast<size_t>(i)], diff, nIdx, cIdx, hIdx, wIdx});
  }
  mse /= static_cast<double>(count);

  printErrorDetails(errors, count);
  printNchwErrorBreakdown(errors, outputShape, threshold);
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
  std::cout << "[ResNet50 Layer Case Shape Check] " << (ok ? "PASS" : "WARNING") << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  const std::vector<int64_t> inputShape = {1, 512, 28, 28};
  const std::vector<int64_t> expectedOutputShape = {1, 512, 28, 28};

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
  std::vector<int64_t> outputShapeVec(
      outputShape, outputShape + static_cast<size_t>(outputRank));
  const VerificationSummary summary =
      compareOutputs(outputData, golden, outputElements, outputShapeVec);
  verifyExpectedShape(outputShape, outputRank, expectedOutputShape);

  omTensorListDestroy(inputList);
  omTensorListDestroy(outputList);
  printFinalSummary(summary);
  return 0;
}
