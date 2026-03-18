#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "OnnxMlirRuntime.h"

extern "C" OMTensorList *run_main_graph(OMTensorList *);

namespace {
namespace fs = std::filesystem;

constexpr int64_t kBatch = 8;
constexpr int64_t kRows = 128;
constexpr int64_t kCols = 256;
constexpr int64_t kPerSampleElements = kRows * kCols;
constexpr int64_t kAllElements = kBatch * kPerSampleElements;
constexpr float kThreshold = 0.1f;
constexpr int64_t kMaxErrorPointsToPrint = 16;
constexpr int64_t kTopAxisBinsToPrint = 8;

struct Options {
  std::string inputFile;
  std::string goldenFile;
  int64_t startIndex = 0;
  int64_t count = 1;
};

struct Summary {
  float maxAbsError = 0.0f;
  double mse = 0.0;
  int64_t errorCount = 0;
  int64_t totalCount = 0;
  bool passed = false;
};

struct ErrorPoint {
  int64_t sample = 0;
  int64_t row = 0;
  int64_t col = 0;
  float actual = 0.0f;
  float golden = 0.0f;
  float diff = 0.0f;
};

struct ParseResult {
  bool ok = false;
  bool showHelp = false;
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
    const fs::path &exeDir, const std::string &modelPrefix, const std::string &logicalName) {
  const fs::path prefixed = exeDir / (modelPrefix + "_" + logicalName);
  if (fs::exists(prefixed))
    return prefixed.string();
  const fs::path legacy = exeDir / logicalName;
  if (fs::exists(legacy))
    return legacy.string();
  return prefixed.string();
}

bool parseI64(const std::string &text, int64_t &value) {
  try {
    size_t pos = 0;
    const long long parsed = std::stoll(text, &pos, 10);
    if (pos != text.size())
      return false;
    value = static_cast<int64_t>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

void printUsage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " [--input <path>] [--golden <path>] [--index <n>] [--count <n>]\n"
            << "  --input  float32 input file, shape 8x128x256 (default: auto-detect)\n"
            << "  --golden float32 golden file, shape 8x128x256 (default: auto-detect)\n"
            << "  --index  start sample index in [0,7] (default: 0)\n"
            << "  --count  sample count in [1, 8-index] (default: 1)\n"
            << "  --help   print help\n";
}

ParseResult parseArgs(int argc, char **argv, Options &opts) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h")
      return {true, true};

    if (i + 1 >= argc) {
      std::cerr << "Error: missing value for argument: " << arg << std::endl;
      return {false, false};
    }
    const std::string value = argv[++i];
    if (arg == "--input") {
      opts.inputFile = value;
    } else if (arg == "--golden") {
      opts.goldenFile = value;
    } else if (arg == "--index") {
      if (!parseI64(value, opts.startIndex)) {
        std::cerr << "Error: invalid --index value: " << value << std::endl;
        return {false, false};
      }
    } else if (arg == "--count") {
      if (!parseI64(value, opts.count)) {
        std::cerr << "Error: invalid --count value: " << value << std::endl;
        return {false, false};
      }
    } else {
      std::cerr << "Error: unknown argument: " << arg << std::endl;
      return {false, false};
    }
  }
  return {true, false};
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
  const std::streamsize expectedBytes =
      static_cast<std::streamsize>(expectedElements * static_cast<int64_t>(sizeof(float)));
  if (bytes != expectedBytes) {
    std::cerr << "Error: file size mismatch for " << filename << ", expected " << expectedBytes
              << ", got " << bytes << std::endl;
    std::exit(1);
  }
  std::vector<float> data(static_cast<size_t>(expectedElements));
  if (!file.read(reinterpret_cast<char *>(data.data()), bytes)) {
    std::cerr << "Error: failed to read file: " << filename << std::endl;
    std::exit(1);
  }
  return data;
}

void checkOutputShape(const int64_t *shape, int64_t rank) {
  bool ok = (rank == 3 && shape[0] == kBatch && shape[1] == kRows && shape[2] == kCols);
  std::cout << "[MatrixAdd Shape Check] " << (ok ? "PASS" : "WARNING") << std::endl;
}

Summary compareRange(const float *actual, const std::vector<float> &golden, int64_t startIndex, int64_t count) {
  Summary summary;
  summary.totalCount = count * kPerSampleElements;
  std::vector<int64_t> rowErrorCounts(static_cast<size_t>(kRows), 0);
  std::vector<int64_t> colErrorCounts(static_cast<size_t>(kCols), 0);
  std::vector<ErrorPoint> firstErrorPoints;
  firstErrorPoints.reserve(static_cast<size_t>(kMaxErrorPointsToPrint));
  int64_t edgeErrorCount = 0;

  for (int64_t sample = startIndex; sample < startIndex + count; ++sample) {
    const int64_t sampleOffset = sample * kPerSampleElements;
    float sampleMaxAbs = 0.0f;
    double sampleMse = 0.0;
    int64_t sampleErrors = 0;

    for (int64_t i = 0; i < kPerSampleElements; ++i) {
      const int64_t index = sampleOffset + i;
      const float diff = std::abs(actual[index] - golden[static_cast<size_t>(index)]);
      summary.maxAbsError = std::max(summary.maxAbsError, diff);
      sampleMaxAbs = std::max(sampleMaxAbs, diff);
      summary.mse += static_cast<double>(diff) * static_cast<double>(diff);
      sampleMse += static_cast<double>(diff) * static_cast<double>(diff);
      if (diff > kThreshold) {
        ++summary.errorCount;
        ++sampleErrors;
        const int64_t row = i / kCols;
        const int64_t col = i % kCols;
        ++rowErrorCounts[static_cast<size_t>(row)];
        ++colErrorCounts[static_cast<size_t>(col)];
        if (row == 0 || row == (kRows - 1) || col == 0 || col == (kCols - 1))
          ++edgeErrorCount;
        if (static_cast<int64_t>(firstErrorPoints.size()) < kMaxErrorPointsToPrint) {
          firstErrorPoints.push_back(
              ErrorPoint{sample, row, col, actual[index], golden[static_cast<size_t>(index)], diff});
        }
      }
    }

    sampleMse /= static_cast<double>(kPerSampleElements);
    std::cout << "[sample " << sample << "] max_abs_error=" << sampleMaxAbs
              << " mse=" << sampleMse
              << " errors=" << sampleErrors << "/" << kPerSampleElements << std::endl;
  }

  summary.mse /= static_cast<double>(summary.totalCount);
  summary.passed = (summary.errorCount == 0);
  std::cout << "Max Absolute Error: " << summary.maxAbsError << std::endl;
  std::cout << "Mean Squared Error: " << summary.mse << std::endl;
  std::cout << "Error Count (错误点数量/总点数量): " << summary.errorCount << "/"
            << summary.totalCount << std::endl;
  std::cout << "Threshold Result: " << (summary.passed ? "PASS" : "FAIL") << std::endl;

  if (!summary.passed) {
    std::cout << "[Error Analysis] first " << firstErrorPoints.size()
              << " error points (sample,row,col,actual,golden,diff):" << std::endl;
    for (size_t i = 0; i < firstErrorPoints.size(); ++i) {
      const auto &p = firstErrorPoints[i];
      std::cout << "  #" << (i + 1) << " (" << p.sample << "," << p.row << "," << p.col
                << ") a=" << p.actual << " g=" << p.golden << " d=" << p.diff << std::endl;
    }

    auto printTopBins = [](const std::vector<int64_t> &counts, const char *axisName) {
      std::vector<std::pair<int64_t, int64_t>> nonZero; // (count, index)
      nonZero.reserve(counts.size());
      int64_t totalAxisErrors = 0;
      for (size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] <= 0)
          continue;
        totalAxisErrors += counts[i];
        nonZero.emplace_back(counts[i], static_cast<int64_t>(i));
      }
      std::sort(
          nonZero.begin(), nonZero.end(),
          [](const std::pair<int64_t, int64_t> &a,
             const std::pair<int64_t, int64_t> &b) {
            if (a.first != b.first)
              return a.first > b.first;
            return a.second < b.second;
          });

      std::cout << "[Error Analysis] " << axisName << " non-zero bins: "
                << nonZero.size() << "/" << counts.size()
                << ", total errors on axis=" << totalAxisErrors << std::endl;
      const int64_t topK =
          std::min<int64_t>(kTopAxisBinsToPrint, static_cast<int64_t>(nonZero.size()));
      for (int64_t k = 0; k < topK; ++k) {
        const std::pair<int64_t, int64_t> &item = nonZero[static_cast<size_t>(k)];
        const int64_t cnt = item.first;
        const int64_t idx = item.second;
        std::cout << "  top" << (k + 1) << " " << axisName << "[" << idx << "]=" << cnt
                  << std::endl;
      }
    };

    printTopBins(rowErrorCounts, "row");
    printTopBins(colErrorCounts, "col");
    std::cout << "[Error Analysis] edge errors (row in {0," << (kRows - 1)
              << "} or col in {0," << (kCols - 1) << "}): " << edgeErrorCount << "/"
              << summary.errorCount << std::endl;
  }
  return summary;
}

void printFinalSummary(const Summary &summary) {
  std::cout << "@@MODEL_TEST_RESULT@@ errors=" << summary.errorCount << "/"
            << summary.totalCount << " status="
            << (summary.passed ? "PASS" : "FAIL") << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  const fs::path exePath = (argc > 0) ? fs::path(argv[0]) : fs::path();
  const fs::path exeDir = exePath.has_parent_path() ? exePath.parent_path() : fs::path(".");
  const std::string modelPrefix = modelPrefixFromExecutable(exePath);

  Options opts;
  opts.inputFile = resolveDataFile(exeDir, modelPrefix, "input.bin");
  opts.goldenFile = resolveDataFile(exeDir, modelPrefix, "output_golden.bin");

  const ParseResult parse = parseArgs(argc, argv, opts);
  if (!parse.ok) {
    printUsage(argv[0]);
    return 1;
  }
  if (parse.showHelp) {
    printUsage(argv[0]);
    return 0;
  }

  if (opts.startIndex < 0 || opts.startIndex >= kBatch) {
    std::cerr << "Error: --index out of range, expected [0, " << (kBatch - 1) << "]" << std::endl;
    return 1;
  }
  if (opts.count <= 0 || opts.startIndex + opts.count > kBatch) {
    std::cerr << "Error: --count out of range, expected [1, " << (kBatch - opts.startIndex)
              << "] for index=" << opts.startIndex << std::endl;
    return 1;
  }

  std::vector<float> inputData = loadBinaryFloatFile(opts.inputFile, kAllElements);
  std::vector<float> goldenData = loadBinaryFloatFile(opts.goldenFile, kAllElements);

  int64_t inputShape[] = {kBatch, kRows, kCols};
  OMTensor *inputTensor = omTensorCreate(inputData.data(), inputShape, 3, ONNX_TYPE_FLOAT);
  OMTensor *inputs[] = {inputTensor};
  OMTensorList *inputList = omTensorListCreate(inputs, 1);
  OMTensorList *outputList = run_main_graph(inputList);

  if (!outputList) {
    std::cerr << "Error: inference returned null output list" << std::endl;
    omTensorListDestroy(inputList);
    return 1;
  }

  OMTensor *outputTensor = omTensorListGetOmtByIndex(outputList, 0);
  if (!outputTensor) {
    std::cerr << "Error: output tensor[0] is null" << std::endl;
    omTensorListDestroy(inputList);
    omTensorListDestroy(outputList);
    return 1;
  }

  float *outputData = reinterpret_cast<float *>(omTensorGetDataPtr(outputTensor));
  if (!outputData) {
    std::cerr << "Error: output tensor data pointer is null" << std::endl;
    omTensorListDestroy(inputList);
    omTensorListDestroy(outputList);
    return 1;
  }

  const int64_t *shape = omTensorGetShape(outputTensor);
  const int64_t rank = omTensorGetRank(outputTensor);
  int64_t outputElements = 1;
  for (int64_t i = 0; i < rank; ++i)
    outputElements *= shape[i];
  if (outputElements != kAllElements) {
    std::cerr << "Error: unexpected output element count, expected " << kAllElements
              << ", got " << outputElements << std::endl;
    omTensorListDestroy(inputList);
    omTensorListDestroy(outputList);
    return 1;
  }

  std::cout << "[Verify Config] index=" << opts.startIndex << ", count=" << opts.count
            << ", threshold=" << kThreshold << std::endl;
  const Summary summary = compareRange(outputData, goldenData, opts.startIndex, opts.count);
  checkOutputShape(shape, rank);

  omTensorListDestroy(inputList);
  omTensorListDestroy(outputList);
  printFinalSummary(summary);
  return summary.passed ? 0 : 1;
}
