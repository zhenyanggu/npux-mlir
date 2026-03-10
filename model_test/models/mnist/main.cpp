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

constexpr int64_t kChannels = 1;
constexpr int64_t kHeight = 28;
constexpr int64_t kWidth = 28;
constexpr int64_t kImageElements = kChannels * kHeight * kWidth;
constexpr int64_t kNumClasses = 10;
constexpr float kAbsDiffThreshold = 0.1f;

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

} // namespace

int main(int argc, char **argv) {
  const fs::path exePath = (argc > 0) ? fs::path(argv[0]) : fs::path();
  const fs::path exeDir = exePath.has_parent_path() ? exePath.parent_path() : fs::path(".");
  const std::string modelPrefix = modelPrefixFromExecutable(exePath);

  const std::string imagesFile = resolveDataFile(exeDir, modelPrefix, "images_u8.bin");
  const std::string inputFile = resolveDataFile(exeDir, modelPrefix, "input.bin");
  const std::string goldenFile =
      resolveDataFile(exeDir, modelPrefix, "output_golden.bin");
  const std::string labelsFile = resolveDataFile(exeDir, modelPrefix, "labels.bin");

  std::cout << "Model prefix: " << modelPrefix << std::endl;
  std::cout << "Images file (raw uint8): " << imagesFile << std::endl;
  std::cout << "Input file (legacy float): " << inputFile << std::endl;
  std::cout << "Golden file: " << goldenFile << std::endl;
  std::cout << "Labels file: " << labelsFile << std::endl;

  bool useRawImages = fs::exists(imagesFile);
  std::vector<uint8_t> rawImageData;
  std::vector<float> inputData;

  if (useRawImages) {
    rawImageData = loadU8Bin(imagesFile);
  } else {
    std::cout << "Warning: raw image file not found, fallback to float input: "
              << inputFile << std::endl;
    inputData = loadFloatBin(inputFile);
  }

  const std::vector<float> goldenData = loadFloatBin(goldenFile);
  const std::vector<int64_t> labels = loadI64Bin(labelsFile);

  if (useRawImages) {
    if (rawImageData.size() != static_cast<size_t>(kImageElements)) {
      std::cerr << "Error: expected one raw image with " << kImageElements
                << " bytes, got " << rawImageData.size()
                << std::endl;
      return 1;
    }
  } else {
    if (inputData.size() != static_cast<size_t>(kImageElements)) {
      std::cerr << "Error: expected one float input image with " << kImageElements
                << " elements, got " << inputData.size()
                << std::endl;
      return 1;
    }
  }

  if (labels.size() != 1) {
    std::cerr << "Error: expected exactly one label, got " << labels.size()
              << std::endl;
    return 1;
  }
  if (goldenData.size() != static_cast<size_t>(kNumClasses)) {
    std::cerr << "Error: expected exactly one golden row with " << kNumClasses
              << " values, got " << goldenData.size()
              << std::endl;
    return 1;
  }

  std::vector<float> preprocessedInput(static_cast<size_t>(kImageElements));

  const float *sampleInput = nullptr;
  if (useRawImages) {
    for (int64_t i = 0; i < kImageElements; ++i) {
      preprocessedInput[static_cast<size_t>(i)] =
          static_cast<float>(rawImageData[static_cast<size_t>(i)]) / 255.0f;
    }
    sampleInput = preprocessedInput.data();
  } else {
    sampleInput = inputData.data();
  }

  int64_t shape[4] = {1, kChannels, kHeight, kWidth};
  OMTensor *inputTensor =
      omTensorCreate(const_cast<float *>(sampleInput), shape, 4, ONNX_TYPE_FLOAT);
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

  float maxAbsError = 0.0f;
  double mse = 0.0;
  int64_t outputErrorCount = 0;

  std::cout << "\n=== Per-Class Output Diff ===" << std::endl;
  std::cout << std::setw(8) << "class" << std::setw(16) << "actual"
            << std::setw(16) << "golden" << std::setw(16) << "abs diff"
            << std::endl;

  for (int64_t c = 0; c < kNumClasses; ++c) {
    const float diff = std::fabs(out[c] - goldenData[static_cast<size_t>(c)]);
    maxAbsError = std::max(maxAbsError, diff);
    mse += static_cast<double>(diff) * static_cast<double>(diff);
    if (diff > kAbsDiffThreshold)
      ++outputErrorCount;

    std::cout << std::setw(8) << c << std::setw(16) << out[c] << std::setw(16)
              << goldenData[static_cast<size_t>(c)] << std::setw(16) << diff
              << std::endl;
  }

  mse /= static_cast<double>(kNumClasses);

  const int64_t pred = argmax(out, kNumClasses);
  const int64_t label = labels[0];
  const bool classMatched = (pred == label);

  std::cout << "\n=== Single Image Summary ===" << std::endl;
  std::cout << "Predicted label: " << pred << std::endl;
  std::cout << "Golden label: " << label << std::endl;
  std::cout << "Classification match: " << (classMatched ? "YES" : "NO")
            << std::endl;
  std::cout << "Max Absolute Error: " << maxAbsError << std::endl;
  std::cout << "Mean Squared Error: " << mse << std::endl;
  std::cout << "Output diff errors (>|" << kAbsDiffThreshold << "|): "
            << outputErrorCount << "/" << kNumClasses << std::endl;

  omTensorListDestroy(inputList);
  omTensorListDestroy(outputList);

  const bool pass = classMatched;
  const int64_t errors = pass ? 0 : 1;
  std::cout << "@@MODEL_TEST_RESULT@@ errors=" << errors
            << "/1 status=" << (pass ? "PASS" : "FAIL") << std::endl;
  return pass ? 0 : 1;
}
