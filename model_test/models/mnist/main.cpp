#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "OnnxMlirRuntime.h"

extern "C" OMTensorList *run_main_graph(OMTensorList *);

namespace {

namespace fs = std::filesystem;

constexpr int64_t kChannels = 1;
constexpr int64_t kHeight = 28;
constexpr int64_t kWidth = 28;
constexpr int64_t kImageElements = kChannels * kHeight * kWidth;
constexpr int64_t kNumClasses = 10;
constexpr float kGoldenDiffThreshold = 0.1f;
constexpr const char *kAsciiRamp = " .:-=+*#%@";

struct Options {
  std::string imagesFile;
  std::string labelsFile;
  std::string goldenFile;
  int64_t startIndex = 0;
  int64_t count = 1;
  bool printAscii = false;
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

std::vector<float> loadFloatBin(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: cannot open file: " << path << std::endl;
    std::exit(1);
  }

  file.seekg(0, std::ios::end);
  const std::streamsize bytes = file.tellg();
  file.seekg(0, std::ios::beg);

  if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
    std::cerr << "Error: invalid float binary size for " << path << ": " << bytes
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

std::string renderAscii(const uint8_t *image) {
  constexpr int kRampSize = 10;
  std::ostringstream oss;
  for (int64_t r = 0; r < kHeight; ++r) {
    for (int64_t c = 0; c < kWidth; ++c) {
      const int64_t idx = r * kWidth + c;
      const int level = static_cast<int>(
          std::round((static_cast<float>(image[idx]) / 255.0f) * (kRampSize - 1)));
      oss << kAsciiRamp[level];
    }
    oss << '\n';
  }
  return oss.str();
}

void printUsage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n"
            << "Options:\n"
            << "  --images <path>    uint8 image file, N x 28 x 28 (default: auto-detect)\n"
            << "  --labels <path>    int64 label file with N entries (default: auto-detect)\n"
            << "  --golden <path>    float32 logits file, N x 10 (default: auto-detect)\n"
            << "  --index <int>      start image index (default: 0)\n"
            << "  --count <int>      number of images to run (default: 1)\n"
            << "  --ascii            print image in ASCII before inference\n"
            << "  --help             show this help\n";
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

ParseStatus parseArgs(int argc, char **argv, Options &opts) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h")
      return ParseStatus::kHelp;
    if (arg == "--ascii") {
      opts.printAscii = true;
      continue;
    }

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

  const std::vector<uint8_t> rawImages = loadU8Bin(opts.imagesFile);
  if (rawImages.size() % static_cast<size_t>(kImageElements) != 0) {
    std::cerr << "Error: images file size is not multiple of 28x28: "
              << rawImages.size() << std::endl;
    return 1;
  }

  const int64_t numImages =
      static_cast<int64_t>(rawImages.size() / static_cast<size_t>(kImageElements));
  if (opts.startIndex >= numImages) {
    std::cerr << "Error: --index " << opts.startIndex
              << " out of range, total images=" << numImages << std::endl;
    return 1;
  }
  const int64_t runCount = std::min<int64_t>(opts.count, numImages - opts.startIndex);

  std::vector<int64_t> labels;
  bool hasLabels = false;
  if (fs::exists(opts.labelsFile)) {
    labels = loadI64Bin(opts.labelsFile);
    if (static_cast<int64_t>(labels.size()) < opts.startIndex + runCount) {
      std::cerr << "Error: labels count(" << labels.size()
                << ") is smaller than requested range end(" << opts.startIndex + runCount
                << ")" << std::endl;
      return 1;
    }
    hasLabels = true;
  } else {
    std::cout << "Warning: labels file not found, classification accuracy will be skipped"
              << std::endl;
  }

  std::vector<float> golden;
  bool hasGolden = false;
  if (fs::exists(opts.goldenFile)) {
    golden = loadFloatBin(opts.goldenFile);
    if (golden.size() % static_cast<size_t>(kNumClasses) != 0) {
      std::cerr << "Error: golden file size is not multiple of class count(" << kNumClasses
                << ")" << std::endl;
      return 1;
    }
    const int64_t goldenRows =
        static_cast<int64_t>(golden.size() / static_cast<size_t>(kNumClasses));
    if (goldenRows < opts.startIndex + runCount) {
      std::cerr << "Error: golden rows(" << goldenRows
                << ") is smaller than requested range end(" << opts.startIndex + runCount
                << ")" << std::endl;
      return 1;
    }
    hasGolden = true;
  } else {
    std::cout << "Warning: golden file not found, output diff check will be skipped"
              << std::endl;
  }

  std::cout << "Running range: index=" << opts.startIndex << ", count=" << runCount
            << ", total_images=" << numImages << std::endl;

  int64_t classifyCorrect = 0;
  int64_t classifyTotal = 0;
  int64_t goldenPassCount = 0;
  int64_t goldenCheckedCount = 0;
  int64_t inferenceTotalUs = 0;

  for (int64_t sample = 0; sample < runCount; ++sample) {
    const int64_t imageIndex = opts.startIndex + sample;
    const uint8_t *imagePtr =
        rawImages.data() + static_cast<size_t>(imageIndex * kImageElements);

    if (opts.printAscii) {
      std::cout << "\n=== Sample " << imageIndex << " ASCII ===" << std::endl;
      std::cout << renderAscii(imagePtr);
    }

    std::vector<float> preprocessedInput(static_cast<size_t>(kImageElements));
    for (int64_t i = 0; i < kImageElements; ++i) {
      preprocessedInput[static_cast<size_t>(i)] =
          static_cast<float>(imagePtr[static_cast<size_t>(i)]) / 255.0f;
    }

    int64_t shape[4] = {1, kChannels, kHeight, kWidth};
    OMTensor *inputTensor = omTensorCreate(preprocessedInput.data(), shape, 4, ONNX_TYPE_FLOAT);
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

    const auto inferStart = std::chrono::steady_clock::now();
    OMTensorList *outputList = run_main_graph(inputList);
    const auto inferEnd = std::chrono::steady_clock::now();
    const int64_t inferUs =
        std::chrono::duration_cast<std::chrono::microseconds>(inferEnd - inferStart)
            .count();
    inferenceTotalUs += inferUs;
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
      std::cerr << "Error: null output data pointer" << std::endl;
      omTensorListDestroy(inputList);
      omTensorListDestroy(outputList);
      return 1;
    }

    const int64_t *outShape = omTensorGetShape(outTensor);
    const int64_t outRank = omTensorGetRank(outTensor);
    int64_t outElems = 1;
    for (int64_t i = 0; i < outRank; ++i)
      outElems *= outShape[i];
    if (outElems != kNumClasses) {
      std::cerr << "Error: expected " << kNumClasses << " classes, got " << outElems
                << std::endl;
      omTensorListDestroy(inputList);
      omTensorListDestroy(outputList);
      return 1;
    }

    std::cout << "\n=== Sample " << imageIndex << " Result ===" << std::endl;
    std::cout << "Inference time: " << inferUs << " us" << std::endl;
    const int64_t pred = argmax(out, kNumClasses);
    std::cout << "Predicted label: " << pred;
    if (hasLabels) {
      const int64_t label = labels[static_cast<size_t>(imageIndex)];
      const bool matched = (pred == label);
      ++classifyTotal;
      if (matched)
        ++classifyCorrect;
      std::cout << ", Golden label: " << label
                << ", Match: " << (matched ? "YES" : "NO");
    }
    std::cout << std::endl;

    if (hasGolden) {
      const size_t goldenBase = static_cast<size_t>(imageIndex * kNumClasses);
      float maxAbsError = 0.0f;
      double mse = 0.0;
      int64_t diffErrors = 0;

      std::cout << std::setw(8) << "class" << std::setw(16) << "actual"
                << std::setw(16) << "golden" << std::setw(16) << "abs diff"
                << std::endl;
      for (int64_t c = 0; c < kNumClasses; ++c) {
        const float g = golden[goldenBase + static_cast<size_t>(c)];
        const float diff = std::fabs(out[c] - g);
        maxAbsError = std::max(maxAbsError, diff);
        mse += static_cast<double>(diff) * static_cast<double>(diff);
        if (diff > kGoldenDiffThreshold)
          ++diffErrors;

        std::cout << std::setw(8) << c << std::setw(16) << out[c] << std::setw(16)
                  << g << std::setw(16) << diff << std::endl;
      }

      mse /= static_cast<double>(kNumClasses);
      const bool goldenPass = (diffErrors == 0);
      ++goldenCheckedCount;
      if (goldenPass)
        ++goldenPassCount;
      std::cout << "Golden check: max_abs=" << maxAbsError << ", mse=" << mse
                << ", diff_errors=" << diffErrors << "/" << kNumClasses
                << ", status=" << (goldenPass ? "PASS" : "FAIL") << std::endl;
    }

    omTensorListDestroy(inputList);
    omTensorListDestroy(outputList);
  }

  std::cout << "\n=== Summary ===" << std::endl;
  if (classifyTotal > 0) {
    const double acc =
        static_cast<double>(classifyCorrect) / static_cast<double>(classifyTotal);
    std::cout << "Classification accuracy: " << classifyCorrect << "/" << classifyTotal
              << " (" << std::fixed << std::setprecision(4) << acc * 100.0 << "%)"
              << std::endl;
  } else {
    std::cout << "Classification accuracy: skipped (no labels)" << std::endl;
  }

  if (goldenCheckedCount > 0) {
    std::cout << "Golden diff pass: " << goldenPassCount << "/" << goldenCheckedCount
              << std::endl;
  } else {
    std::cout << "Golden diff check: skipped (no golden file)" << std::endl;
  }

  if (runCount > 0) {
    const double inferenceAvgUs =
        static_cast<double>(inferenceTotalUs) / static_cast<double>(runCount);
    std::cout << "Inference time total: " << inferenceTotalUs << " us" << std::endl;
    std::cout << "Inference time avg: " << inferenceAvgUs << " us/sample" << std::endl;
  }

  const bool pass = (classifyTotal == 0) ? true : (classifyCorrect == classifyTotal);
  const int64_t errors = pass ? 0 : (classifyTotal - classifyCorrect);
  const int64_t total = (classifyTotal == 0) ? runCount : classifyTotal;
  std::cout << "@@MODEL_TEST_RESULT@@ errors=" << errors << "/" << total
            << " status=" << (pass ? "PASS" : "FAIL") << std::endl;
  return pass ? 0 : 1;
}
