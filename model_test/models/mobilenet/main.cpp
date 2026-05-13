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

constexpr int64_t kChannels = 3;
constexpr int64_t kHeight = 224;
constexpr int64_t kWidth = 224;
constexpr int64_t kImageElements = kChannels * kHeight * kWidth;
constexpr int64_t kNumClasses = 1000;

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

struct Options {
  std::string imagesFile;
  std::string labelsFile;
  std::string goldenFile;
  int64_t startIndex = 0;
  int64_t count = 1;
  float diffThreshold = 0.25f;
};

enum class ParseStatus {
  kOk,
  kHelp,
  kError,
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
  return name.empty() ? "mobilenet" : name;
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

std::vector<uint8_t> loadU8Bin(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: cannot open file: " << path << std::endl;
    std::exit(1);
  }

  file.seekg(0, std::ios::end);
  const std::streamsize bytes = file.tellg();
  file.seekg(0, std::ios::beg);

  if (bytes <= 0) {
    std::cerr << "Error: invalid uint8 binary size for " << path << ": " << bytes
              << std::endl;
    std::exit(1);
  }

  std::vector<uint8_t> data(static_cast<size_t>(bytes));
  if (!file.read(reinterpret_cast<char *>(data.data()), bytes)) {
    std::cerr << "Error: failed to read file: " << path << std::endl;
    std::exit(1);
  }
  return data;
}

std::vector<int64_t> loadI64Bin(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: cannot open file: " << path << std::endl;
    std::exit(1);
  }

  file.seekg(0, std::ios::end);
  const std::streamsize bytes = file.tellg();
  file.seekg(0, std::ios::beg);

  if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(int64_t)) != 0) {
    std::cerr << "Error: invalid int64 binary size for " << path << ": " << bytes
              << std::endl;
    std::exit(1);
  }

  std::vector<int64_t> data(static_cast<size_t>(bytes / sizeof(int64_t)));
  if (!file.read(reinterpret_cast<char *>(data.data()), bytes)) {
    std::cerr << "Error: failed to read file: " << path << std::endl;
    std::exit(1);
  }
  return data;
}

std::vector<float> loadF32Bin(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: cannot open file: " << path << std::endl;
    std::exit(1);
  }

  file.seekg(0, std::ios::end);
  const std::streamsize bytes = file.tellg();
  file.seekg(0, std::ios::beg);

  if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
    std::cerr << "Error: invalid float32 binary size for " << path << ": " << bytes
              << std::endl;
    std::exit(1);
  }

  std::vector<float> data(static_cast<size_t>(bytes / sizeof(float)));
  if (!file.read(reinterpret_cast<char *>(data.data()), bytes)) {
    std::cerr << "Error: failed to read file: " << path << std::endl;
    std::exit(1);
  }
  return data;
}

int64_t argmax(const float *row, int64_t size) {
  int64_t idx = 0;
  float best = row[0];
  for (int64_t i = 1; i < size; ++i) {
    if (row[i] > best) {
      best = row[i];
      idx = i;
    }
  }
  return idx;
}

bool parseI64(const std::string &s, int64_t &value) {
  try {
    size_t pos = 0;
    const long long v = std::stoll(s, &pos, 10);
    if (pos != s.size())
      return false;
    value = static_cast<int64_t>(v);
    return true;
  } catch (...) {
    return false;
  }
}

bool parseFloat(const std::string &s, float &value) {
  try {
    size_t pos = 0;
    const float v = std::stof(s, &pos);
    if (pos != s.size())
      return false;
    value = v;
    return true;
  } catch (...) {
    return false;
  }
}

void printUsage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n"
            << "Options:\n"
            << "  --images <path>          uint8 image file, N x 3 x 224 x 224 (default: auto-detect)\n"
            << "  --labels <path>          int64 labels, N entries (default: auto-detect)\n"
            << "  --golden <path>          float32 logits, N x 1000 (default: auto-detect)\n"
            << "  --index <int>            start index (default: 0)\n"
            << "  --count <int>            number of samples to run (default: 1)\n"
            << "  --diff-threshold <float> per-logit abs diff threshold (default: 0.25)\n"
            << "  --help                   show this help\n";
}

ParseStatus parseArgs(int argc, char **argv, Options &opts) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h")
      return ParseStatus::kHelp;

    if (i + 1 >= argc) {
      std::cerr << "Error: missing value for argument: " << arg << std::endl;
      return ParseStatus::kError;
    }
    const std::string value = argv[++i];

    if (arg == "--images") {
      opts.imagesFile = value;
    } else if (arg == "--labels") {
      opts.labelsFile = value;
    } else if (arg == "--golden") {
      opts.goldenFile = value;
    } else if (arg == "--index") {
      if (!parseI64(value, opts.startIndex)) {
        std::cerr << "Error: invalid --index value: " << value << std::endl;
        return ParseStatus::kError;
      }
    } else if (arg == "--count") {
      if (!parseI64(value, opts.count)) {
        std::cerr << "Error: invalid --count value: " << value << std::endl;
        return ParseStatus::kError;
      }
    } else if (arg == "--diff-threshold") {
      if (!parseFloat(value, opts.diffThreshold)) {
        std::cerr << "Error: invalid --diff-threshold value: " << value << std::endl;
        return ParseStatus::kError;
      }
    } else {
      std::cerr << "Error: unknown argument: " << arg << std::endl;
      return ParseStatus::kError;
    }
  }

  if (opts.startIndex < 0) {
    std::cerr << "Error: --index must be >= 0" << std::endl;
    return ParseStatus::kError;
  }
  if (opts.count <= 0) {
    std::cerr << "Error: --count must be > 0" << std::endl;
    return ParseStatus::kError;
  }
  if (opts.diffThreshold < 0.0f) {
    std::cerr << "Error: --diff-threshold must be >= 0" << std::endl;
    return ParseStatus::kError;
  }

  return ParseStatus::kOk;
}

} // namespace

int main(int argc, char **argv) {
  const fs::path exePath = (argc > 0) ? fs::path(argv[0]) : fs::path();
  const fs::path exeDir = exePath.has_parent_path() ? exePath.parent_path() : fs::path(".");
  const std::string modelPrefix = modelPrefixFromExecutable(exePath);

  Options opts;
  opts.imagesFile = resolveDataFile(exeDir, modelPrefix, "images_u8.bin");
  opts.labelsFile = resolveDataFile(exeDir, modelPrefix, "labels.bin");
  opts.goldenFile = resolveDataFile(exeDir, modelPrefix, "output_golden.bin");

  const ParseStatus parseStatus = parseArgs(argc, argv, opts);
  if (parseStatus == ParseStatus::kHelp) {
    printUsage(argv[0]);
    return 0;
  }
  if (parseStatus == ParseStatus::kError) {
    printUsage(argv[0]);
    return 1;
  }

  std::cout << "Model prefix: " << modelPrefix << std::endl;
  std::cout << "Images file: " << opts.imagesFile << std::endl;
  std::cout << "Labels file: " << opts.labelsFile << std::endl;
  std::cout << "Golden file: " << opts.goldenFile << std::endl;

  if (!fs::exists(opts.imagesFile)) {
    std::cerr << "Error: images file not found: " << opts.imagesFile << std::endl;
    return 1;
  }
  if (!fs::exists(opts.labelsFile)) {
    std::cerr << "Error: labels file not found: " << opts.labelsFile << std::endl;
    return 1;
  }
  if (!fs::exists(opts.goldenFile)) {
    std::cerr << "Error: golden file not found: " << opts.goldenFile << std::endl;
    return 1;
  }

  const std::vector<uint8_t> rawImages = loadU8Bin(opts.imagesFile);
  if (rawImages.size() % static_cast<size_t>(kImageElements) != 0) {
    std::cerr << "Error: images file size is not multiple of sample size(" << kImageElements
              << "): " << rawImages.size() << std::endl;
    return 1;
  }

  const int64_t numImages =
      static_cast<int64_t>(rawImages.size() / static_cast<size_t>(kImageElements));
  const std::vector<int64_t> labels = loadI64Bin(opts.labelsFile);
  if (static_cast<int64_t>(labels.size()) != numImages) {
    std::cerr << "Error: labels count mismatch, images=" << numImages
              << ", labels=" << labels.size() << std::endl;
    return 1;
  }

  const std::vector<float> golden = loadF32Bin(opts.goldenFile);
  if (golden.size() % static_cast<size_t>(kNumClasses) != 0) {
    std::cerr << "Error: golden file size is not multiple of class count(" << kNumClasses
              << ")" << std::endl;
    return 1;
  }

  const int64_t goldenRows =
      static_cast<int64_t>(golden.size() / static_cast<size_t>(kNumClasses));
  if (goldenRows != numImages) {
    std::cerr << "Error: golden rows mismatch, images=" << numImages
              << ", golden_rows=" << goldenRows << std::endl;
    return 1;
  }

  if (opts.startIndex >= numImages) {
    std::cerr << "Error: --index " << opts.startIndex
              << " out of range, total images=" << numImages << std::endl;
    return 1;
  }

  const int64_t runCount = std::min<int64_t>(opts.count, numImages - opts.startIndex);
  std::cout << "Running range: index=" << opts.startIndex << ", count=" << runCount
            << ", total_images=" << numImages << std::endl;

  int64_t labelCorrect = 0;
  int64_t semanticErrors = 0;
  int64_t numericErrors = 0;
  int64_t classificationErrors = 0;

  for (int64_t sample = 0; sample < runCount; ++sample) {
    const int64_t imageIndex = opts.startIndex + sample;
    const uint8_t *imagePtr =
        rawImages.data() + static_cast<size_t>(imageIndex * kImageElements);

    std::vector<float> inputData(static_cast<size_t>(kImageElements));
    for (int64_t c = 0; c < kChannels; ++c) {
      const int64_t cBase = c * kHeight * kWidth;
      for (int64_t hw = 0; hw < kHeight * kWidth; ++hw) {
        const float x = static_cast<float>(imagePtr[cBase + hw]) / 255.0f;
        inputData[static_cast<size_t>(cBase + hw)] = (x - kMean[c]) / kStd[c];
      }
    }

    int64_t inShape[4] = {1, kChannels, kHeight, kWidth};
    OMTensor *inputTensor = omTensorCreate(inputData.data(), inShape, 4, ONNX_TYPE_FLOAT);
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

    OMTensor *outTensor = omTensorListGetOmtByIndex(outputList, 0);
    if (!outTensor) {
      std::cerr << "Error: missing output tensor" << std::endl;
      omTensorListDestroy(inputList);
      omTensorListDestroy(outputList);
      return 1;
    }

    float *out = reinterpret_cast<float *>(omTensorGetDataPtr(outTensor));
    if (!out) {
      std::cerr << "Error: output data pointer is null" << std::endl;
      omTensorListDestroy(inputList);
      omTensorListDestroy(outputList);
      return 1;
    }

    const int64_t *outShape = omTensorGetShape(outTensor);
    const int64_t outRank = omTensorGetRank(outTensor);
    if (outRank != 2 || outShape[0] != 1 || outShape[1] != kNumClasses) {
      std::cerr << "Error: unexpected output shape, rank=" << outRank << ", shape=[";
      for (int64_t i = 0; i < outRank; ++i)
        std::cerr << outShape[i] << (i + 1 == outRank ? "" : ",");
      std::cerr << "]" << std::endl;
      omTensorListDestroy(inputList);
      omTensorListDestroy(outputList);
      return 1;
    }

    const size_t goldenBase = static_cast<size_t>(imageIndex * kNumClasses);
    const float *goldenRow = golden.data() + goldenBase;

    const int64_t pred = argmax(out, kNumClasses);
    const int64_t goldenPred = argmax(goldenRow, kNumClasses);
    const int64_t label = labels[static_cast<size_t>(imageIndex)];

    const bool semanticOk = (pred == goldenPred);
    if (!semanticOk)
      ++semanticErrors;

    const bool labelMatch = (pred == label);
    if (labelMatch)
      ++labelCorrect;

    float maxAbs = 0.0f;
    double mse = 0.0;
    int64_t diffErrors = 0;
    for (int64_t c = 0; c < kNumClasses; ++c) {
      const float diff = std::fabs(out[c] - goldenRow[c]);
      maxAbs = std::max(maxAbs, diff);
      mse += static_cast<double>(diff) * static_cast<double>(diff);
      if (diff > opts.diffThreshold)
        ++diffErrors;
    }
    mse /= static_cast<double>(kNumClasses);

    if (diffErrors > 0)
      ++numericErrors;

    if (!labelMatch)
      ++classificationErrors;

    std::cout << "sample=" << imageIndex << " pred=" << pred << " golden_pred=" << goldenPred
              << " label=" << label << " label_match=" << (labelMatch ? "YES" : "NO")
              << " max_abs=" << maxAbs << " mse=" << mse
              << " diff_errors=" << diffErrors << "/" << kNumClasses
              << " status=" << (labelMatch ? "PASS" : "FAIL")
              << std::endl;

    omTensorListDestroy(inputList);
    omTensorListDestroy(outputList);
  }

  const double labelAcc =
      static_cast<double>(labelCorrect) / static_cast<double>(runCount);
  std::cout << "Classification accuracy: " << labelCorrect << "/" << runCount << " ("
            << std::fixed << std::setprecision(4) << labelAcc * 100.0 << "%)" << std::endl;
  std::cout << "Semantic check (pred==golden_pred): " << (runCount - semanticErrors)
            << "/" << runCount << std::endl;
  std::cout << "Numeric check (all logits <= threshold): " << (runCount - numericErrors)
            << "/" << runCount << std::endl;

  const bool pass = (classificationErrors == 0);
  std::cout << "@@MODEL_TEST_RESULT@@ errors=" << classificationErrors << "/" << runCount
            << " status=" << (pass ? "PASS" : "FAIL") << std::endl;
  return pass ? 0 : 1;
}
