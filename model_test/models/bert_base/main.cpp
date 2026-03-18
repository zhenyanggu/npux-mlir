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

constexpr int64_t kFixedSeqLen = 128;

struct Options {
  std::string inputIdsFile;
  std::string attentionMaskFile;
  std::string tokenTypeIdsFile;
  std::string goldenFile;
  std::string labelsFile;
  int64_t startIndex = 0;
  int64_t count = 1;
  float diffThreshold = 0.2f;
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
  return name.empty() ? "bert_base" : name;
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
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "Options:\n"
      << "  --input-ids <path>       int64 input_ids, N x seq_len (default: auto-detect)\n"
      << "  --attention-mask <path>  int64 attention_mask, N x seq_len (default: auto-detect)\n"
      << "  --token-type-ids <path>  int64 token_type_ids, N x seq_len (default: auto-detect, optional)\n"
      << "  --golden <path>          float32 golden logits, N x (...) (default: auto-detect)\n"
      << "  --labels <path>          int64 labels (optional, default: auto-detect if exists)\n"
      << "  NOTE: sequence length is fixed to 128 (no dynamic seq-len)\n"
      << "  --index <int>            start sample index (default: 0)\n"
      << "  --count <int>            number of samples to run (default: 1)\n"
      << "  --diff-threshold <float> per-logit abs diff threshold (default: 0.2)\n"
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

    if (arg == "--input-ids") {
      opts.inputIdsFile = value;
    } else if (arg == "--attention-mask") {
      opts.attentionMaskFile = value;
    } else if (arg == "--token-type-ids") {
      opts.tokenTypeIdsFile = value;
    } else if (arg == "--golden") {
      opts.goldenFile = value;
    } else if (arg == "--labels") {
      opts.labelsFile = value;
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

int64_t tensorElementCount(const OMTensor *tensor) {
  if (!tensor)
    return 0;
  const int64_t rank = omTensorGetRank(tensor);
  if (rank <= 0)
    return 0;
  const int64_t *shape = omTensorGetShape(const_cast<OMTensor *>(tensor));
  if (!shape)
    return 0;

  int64_t count = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (shape[i] <= 0)
      return 0;
    count *= shape[i];
  }
  return count;
}

} // namespace

int main(int argc, char **argv) {
  const fs::path exePath = (argc > 0) ? fs::path(argv[0]) : fs::path();
  const fs::path exeDir = exePath.has_parent_path() ? exePath.parent_path() : fs::path(".");
  const std::string modelPrefix = modelPrefixFromExecutable(exePath);

  Options opts;
  opts.inputIdsFile = resolveDataFile(exeDir, modelPrefix, "input_ids.bin");
  opts.attentionMaskFile = resolveDataFile(exeDir, modelPrefix, "attention_mask.bin");
  opts.tokenTypeIdsFile = resolveDataFile(exeDir, modelPrefix, "token_type_ids.bin");
  opts.goldenFile = resolveDataFile(exeDir, modelPrefix, "output_golden.bin");
  opts.labelsFile = resolveDataFile(exeDir, modelPrefix, "labels.bin");

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
  std::cout << "input_ids file: " << opts.inputIdsFile << std::endl;
  std::cout << "attention_mask file: " << opts.attentionMaskFile << std::endl;
  std::cout << "token_type_ids file: " << opts.tokenTypeIdsFile << std::endl;
  std::cout << "golden file: " << opts.goldenFile << std::endl;
  std::cout << "labels file: " << opts.labelsFile << std::endl;
  std::cout << "fixed seq_len: " << kFixedSeqLen << std::endl;

  if (!fs::exists(opts.inputIdsFile)) {
    std::cerr << "Error: input_ids file not found: " << opts.inputIdsFile << std::endl;
    return 1;
  }
  if (!fs::exists(opts.attentionMaskFile)) {
    std::cerr << "Error: attention_mask file not found: " << opts.attentionMaskFile
              << std::endl;
    return 1;
  }
  if (!fs::exists(opts.goldenFile)) {
    std::cerr << "Error: golden file not found: " << opts.goldenFile << std::endl;
    return 1;
  }

  const std::vector<int64_t> inputIds = loadI64Bin(opts.inputIdsFile);
  if (inputIds.empty()) {
    std::cerr << "Error: input_ids is empty" << std::endl;
    return 1;
  }
  if (static_cast<int64_t>(inputIds.size()) % kFixedSeqLen != 0) {
    std::cerr << "Error: input_ids element count " << inputIds.size()
              << " is not divisible by fixed seq_len=" << kFixedSeqLen << std::endl;
    return 1;
  }
  const int64_t numSamples =
      static_cast<int64_t>(inputIds.size()) / kFixedSeqLen;

  const std::vector<int64_t> attentionMask = loadI64Bin(opts.attentionMaskFile);
  if (attentionMask.size() != inputIds.size()) {
    std::cerr << "Error: attention_mask size mismatch, input_ids=" << inputIds.size()
              << ", attention_mask=" << attentionMask.size() << std::endl;
    return 1;
  }

  std::vector<int64_t> tokenTypeIds;
  bool hasTokenTypeIds = fs::exists(opts.tokenTypeIdsFile);
  if (hasTokenTypeIds) {
    tokenTypeIds = loadI64Bin(opts.tokenTypeIdsFile);
    if (tokenTypeIds.size() != inputIds.size()) {
      std::cerr << "Error: token_type_ids size mismatch, input_ids=" << inputIds.size()
                << ", token_type_ids=" << tokenTypeIds.size() << std::endl;
      return 1;
    }
  } else {
    tokenTypeIds.assign(inputIds.size(), 0);
    std::cout << "Info: token_type_ids file not found, fallback to all-zero segment ids."
              << std::endl;
  }

  const std::vector<float> golden = loadF32Bin(opts.goldenFile);
  if (static_cast<int64_t>(golden.size()) % numSamples != 0) {
    std::cerr << "Error: golden element count " << golden.size()
              << " is not divisible by numSamples=" << numSamples << std::endl;
    return 1;
  }
  const int64_t goldenPerSample =
      static_cast<int64_t>(golden.size()) / numSamples;

  std::vector<int64_t> labels;
  const bool hasLabels = fs::exists(opts.labelsFile);
  if (hasLabels) {
    labels = loadI64Bin(opts.labelsFile);
    if (static_cast<int64_t>(labels.size()) != numSamples) {
      std::cerr << "Error: labels count mismatch, samples=" << numSamples
                << ", labels=" << labels.size() << std::endl;
      return 1;
    }
  } else {
    std::cout << "Info: labels file not found, skip label accuracy." << std::endl;
  }

  if (opts.startIndex >= numSamples) {
    std::cerr << "Error: --index " << opts.startIndex
              << " out of range, total samples=" << numSamples << std::endl;
    return 1;
  }
  const int64_t runCount = std::min<int64_t>(opts.count, numSamples - opts.startIndex);
  std::cout << "Running range: index=" << opts.startIndex << ", count=" << runCount
            << ", total_samples=" << numSamples << std::endl;

  int64_t semanticErrors = 0;
  int64_t numericErrors = 0;
  int64_t labelCorrect = 0;

  for (int64_t sample = 0; sample < runCount; ++sample) {
    const int64_t sampleIndex = opts.startIndex + sample;
    const int64_t tokenBase = sampleIndex * kFixedSeqLen;

    int64_t inputShape[2] = {1, kFixedSeqLen};
    int64_t *idsPtr =
        const_cast<int64_t *>(inputIds.data() + static_cast<size_t>(tokenBase));
    int64_t *maskPtr =
        const_cast<int64_t *>(attentionMask.data() + static_cast<size_t>(tokenBase));
    int64_t *typePtr =
        const_cast<int64_t *>(tokenTypeIds.data() + static_cast<size_t>(tokenBase));

    OMTensor *idsTensor = omTensorCreate(idsPtr, inputShape, 2, ONNX_TYPE_INT64);
    OMTensor *maskTensor = omTensorCreate(maskPtr, inputShape, 2, ONNX_TYPE_INT64);
    OMTensor *typeTensor = omTensorCreate(typePtr, inputShape, 2, ONNX_TYPE_INT64);
    if (!idsTensor || !maskTensor || !typeTensor) {
      std::cerr << "Error: failed to create input tensors" << std::endl;
      return 1;
    }

    OMTensor *inputs[] = {idsTensor, maskTensor, typeTensor};
    OMTensorList *inputList = omTensorListCreate(inputs, 3);
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

    const int64_t outRank = omTensorGetRank(outTensor);
    const int64_t *outShape = omTensorGetShape(outTensor);
    if (!outShape || outRank < 2 || outShape[0] != 1) {
      std::cerr << "Error: unexpected output shape. rank=" << outRank << std::endl;
      omTensorListDestroy(inputList);
      omTensorListDestroy(outputList);
      return 1;
    }
    if (outShape[1] != kFixedSeqLen) {
      std::cerr << "Error: output seq_len mismatch. expect " << kFixedSeqLen
                << ", got " << outShape[1] << std::endl;
      omTensorListDestroy(inputList);
      omTensorListDestroy(outputList);
      return 1;
    }

    const int64_t outputElements = tensorElementCount(outTensor);
    if (outputElements <= 0) {
      std::cerr << "Error: invalid output element count: " << outputElements << std::endl;
      omTensorListDestroy(inputList);
      omTensorListDestroy(outputList);
      return 1;
    }
    if (outputElements != goldenPerSample) {
      std::cerr << "Error: output element count mismatch. model=" << outputElements
                << ", golden_per_sample=" << goldenPerSample << std::endl;
      omTensorListDestroy(inputList);
      omTensorListDestroy(outputList);
      return 1;
    }

    const size_t goldenBase = static_cast<size_t>(sampleIndex * goldenPerSample);
    const float *goldenRow = golden.data() + goldenBase;

    const int64_t pred = argmax(out, outputElements);
    const int64_t goldenPred = argmax(goldenRow, outputElements);
    const bool semanticOk = (pred == goldenPred);
    if (!semanticOk)
      ++semanticErrors;

    bool labelMatch = false;
    int64_t label = -1;
    if (hasLabels) {
      label = labels[static_cast<size_t>(sampleIndex)];
      labelMatch = (pred == label);
      if (labelMatch)
        ++labelCorrect;
    }

    float maxAbs = 0.0f;
    double mse = 0.0;
    int64_t diffErrors = 0;
    for (int64_t i = 0; i < outputElements; ++i) {
      const float diff = std::fabs(out[i] - goldenRow[i]);
      maxAbs = std::max(maxAbs, diff);
      mse += static_cast<double>(diff) * static_cast<double>(diff);
      if (diff > opts.diffThreshold)
        ++diffErrors;
    }
    mse /= static_cast<double>(outputElements);

    if (diffErrors > 0)
      ++numericErrors;

    std::cout << "sample=" << sampleIndex << " pred=" << pred
              << " golden_pred=" << goldenPred;
    if (hasLabels) {
      std::cout << " label=" << label
                << " label_match=" << (labelMatch ? "YES" : "NO");
    }
    std::cout << " max_abs=" << maxAbs << " mse=" << mse
              << " diff_errors=" << diffErrors << "/" << outputElements
              << " status=" << (semanticOk ? "PASS" : "FAIL") << std::endl;

    omTensorListDestroy(inputList);
    omTensorListDestroy(outputList);
  }

  if (hasLabels) {
    const double labelAcc = static_cast<double>(labelCorrect) / static_cast<double>(runCount);
    std::cout << "Label consistency: " << labelCorrect << "/" << runCount << " ("
              << std::fixed << std::setprecision(4) << labelAcc * 100.0 << "%)"
              << std::endl;
  }
  std::cout << "Semantic check (pred==golden_pred): " << (runCount - semanticErrors)
            << "/" << runCount << std::endl;
  std::cout << "Numeric check (all logits <= threshold): " << (runCount - numericErrors)
            << "/" << runCount << std::endl;

  const bool pass = (semanticErrors == 0);
  std::cout << "@@MODEL_TEST_RESULT@@ errors=" << semanticErrors << "/" << runCount
            << " status=" << (pass ? "PASS" : "FAIL") << std::endl;
  return pass ? 0 : 1;
}
