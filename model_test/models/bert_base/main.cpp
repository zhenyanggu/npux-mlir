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
constexpr float kDefaultDiffThreshold = 0.5f;
constexpr float kDefaultTokenAccThreshold = 0.99f;
constexpr int64_t kDefaultMaxTokenMismatches = 1;
constexpr size_t kTopDiffCount = 5;

struct Options {
  std::string inputIdsFile;
  std::string attentionMaskFile;
  std::string tokenTypeIdsFile;
  std::string goldenFile;
  std::string labelsFile;
  int64_t startIndex = 0;
  int64_t count = 1;
  float diffThreshold = kDefaultDiffThreshold;
  float tokenAccThreshold = kDefaultTokenAccThreshold;
  int64_t maxTokenMismatches = kDefaultMaxTokenMismatches;
};

struct DiffItem {
  int64_t flatIndex;
  int64_t tokenIndex;
  int64_t vocabIndex;
  float actual;
  float golden;
  float absDiff;
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

void maybeInsertTopDiff(std::vector<DiffItem> &topDiffs, const DiffItem &item) {
  if (topDiffs.size() < kTopDiffCount) {
    topDiffs.push_back(item);
  } else {
    auto smallest = std::min_element(topDiffs.begin(), topDiffs.end(),
        [](const DiffItem &lhs, const DiffItem &rhs) { return lhs.absDiff < rhs.absDiff; });
    if (smallest == topDiffs.end() || item.absDiff <= smallest->absDiff)
      return;
    *smallest = item;
  }

  std::sort(topDiffs.begin(), topDiffs.end(),
      [](const DiffItem &lhs, const DiffItem &rhs) { return lhs.absDiff > rhs.absDiff; });
}

void printUsage(const char *argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "Options:\n"
      << "  --input-ids <path>       int64 input_ids, N x seq_len (default: auto-detect)\n"
      << "  --attention-mask <path>  int64 attention_mask, N x seq_len (default: auto-detect)\n"
      << "  --token-type-ids <path>  int64 token_type_ids, N x seq_len (default: auto-detect, optional)\n"
      << "  --golden <path>          float32 golden logits, N x (...) (default: auto-detect)\n"
      << "  --labels <path>          legacy option, ignored for bert token-level check\n"
      << "  NOTE: sequence length is fixed to 128 (no dynamic seq-len)\n"
      << "  --index <int>            start sample index (default: 0)\n"
      << "  --count <int>            number of samples to run (default: 1)\n"
      << "  --diff-threshold <float> per-logit abs diff threshold (default: "
      << kDefaultDiffThreshold << ")\n"
      << "  --token-acc-threshold <float> minimum per-sample token accuracy in [0, 1]"
      << " (default: " << kDefaultTokenAccThreshold << ")\n"
      << "  --max-token-mismatches <int> allowed token mismatches per sample"
      << " (default: " << kDefaultMaxTokenMismatches << ")\n"
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
    } else if (arg == "--token-acc-threshold") {
      if (!parseFloat(value, opts.tokenAccThreshold)) {
        std::cerr << "Error: invalid --token-acc-threshold value: " << value
                  << std::endl;
        return ParseStatus::kError;
      }
    } else if (arg == "--max-token-mismatches") {
      if (!parseI64(value, opts.maxTokenMismatches)) {
        std::cerr << "Error: invalid --max-token-mismatches value: " << value
                  << std::endl;
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
  if (opts.tokenAccThreshold < 0.0f || opts.tokenAccThreshold > 1.0f) {
    std::cerr << "Error: --token-acc-threshold must be in [0, 1]" << std::endl;
    return ParseStatus::kError;
  }
  if (opts.maxTokenMismatches < 0) {
    std::cerr << "Error: --max-token-mismatches must be >= 0" << std::endl;
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
  std::cout << "labels file: " << opts.labelsFile << " (ignored)" << std::endl;
  std::cout << "fixed seq_len: " << kFixedSeqLen << std::endl;
  std::cout << "verification config: diff_threshold=" << opts.diffThreshold
            << ", token_acc_threshold=" << opts.tokenAccThreshold
            << ", max_token_mismatches=" << opts.maxTokenMismatches << std::endl;

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

  if (fs::exists(opts.labelsFile)) {
    std::cout << "Info: labels file is ignored for bert token-level validation."
              << std::endl;
  }

  if (opts.startIndex >= numSamples) {
    std::cerr << "Error: --index " << opts.startIndex
              << " out of range, total samples=" << numSamples << std::endl;
    return 1;
  }
  const int64_t runCount = std::min<int64_t>(opts.count, numSamples - opts.startIndex);
  std::cout << "Running range: index=" << opts.startIndex << ", count=" << runCount
            << ", total_samples=" << numSamples << std::endl;

  int64_t exactSemanticErrors = 0;
  int64_t relaxedSemanticErrors = 0;
  int64_t numericErrors = 0;
  int64_t totalValidTokens = 0;
  int64_t totalMatchedTokens = 0;
  int64_t totalDiffErrors = 0;
  int64_t totalOutputElements = 0;
  double totalAbsError = 0.0;
  double totalSquaredError = 0.0;
  float globalMaxAbs = 0.0f;

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
    if (!outShape || outRank != 3 || outShape[0] != 1) {
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
    if (outShape[2] <= 0) {
      std::cerr << "Error: invalid vocab dimension: " << outShape[2] << std::endl;
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

    const int64_t vocabSize = outShape[2];
    int64_t validTokens = 0;
    int64_t matchedTokens = 0;
    int64_t firstMismatchPos = -1;
    int64_t firstPredToken = -1;
    int64_t firstGoldenToken = -1;
    for (int64_t token = 0; token < kFixedSeqLen; ++token) {
      if (maskPtr[token] == 0)
        continue;
      ++validTokens;
      const int64_t tokenOffset = token * vocabSize;
      const int64_t predToken = argmax(out + tokenOffset, vocabSize);
      const int64_t goldenToken = argmax(goldenRow + tokenOffset, vocabSize);
      if (predToken == goldenToken) {
        ++matchedTokens;
      } else if (firstMismatchPos < 0) {
        firstMismatchPos = token;
        firstPredToken = predToken;
        firstGoldenToken = goldenToken;
      }
    }
    totalValidTokens += validTokens;
    totalMatchedTokens += matchedTokens;

    const int64_t tokenMismatches = validTokens - matchedTokens;
    const double sampleTokenAcc = validTokens > 0
                                      ? static_cast<double>(matchedTokens) /
                                            static_cast<double>(validTokens)
                                      : 0.0;
    const bool exactSemanticOk = (validTokens > 0) && (matchedTokens == validTokens);
    const bool relaxedSemanticOk = (validTokens > 0) &&
                                   (sampleTokenAcc >= opts.tokenAccThreshold) &&
                                   (tokenMismatches <= opts.maxTokenMismatches);
    if (!exactSemanticOk)
      ++exactSemanticErrors;
    if (!relaxedSemanticOk)
      ++relaxedSemanticErrors;

    float maxAbs = 0.0f;
    double meanAbs = 0.0;
    double mse = 0.0;
    int64_t diffErrors = 0;
    std::vector<DiffItem> topDiffs;
    topDiffs.reserve(kTopDiffCount);
    for (int64_t i = 0; i < outputElements; ++i) {
      const float diff = std::fabs(out[i] - goldenRow[i]);
      maxAbs = std::max(maxAbs, diff);
      meanAbs += static_cast<double>(diff);
      mse += static_cast<double>(diff) * static_cast<double>(diff);
      if (diff > opts.diffThreshold)
        ++diffErrors;
      maybeInsertTopDiff(topDiffs,
          {i, i / vocabSize, i % vocabSize, out[i], goldenRow[i], diff});
    }
    meanAbs /= static_cast<double>(outputElements);
    mse /= static_cast<double>(outputElements);
    totalDiffErrors += diffErrors;
    totalOutputElements += outputElements;
    totalAbsError += meanAbs * static_cast<double>(outputElements);
    totalSquaredError += mse * static_cast<double>(outputElements);
    globalMaxAbs = std::max(globalMaxAbs, maxAbs);

    if (diffErrors > 0)
      ++numericErrors;

    std::cout << "sample=" << sampleIndex << " valid_tokens=" << validTokens
              << " token_match=" << matchedTokens << "/" << validTokens
              << " token_acc=" << std::fixed << std::setprecision(4)
              << sampleTokenAcc * 100.0 << "%"
              << " token_mismatches=" << tokenMismatches;
    if (firstMismatchPos >= 0) {
      std::cout << " first_mismatch_pos=" << firstMismatchPos
                << " pred_token=" << firstPredToken
                << " golden_token=" << firstGoldenToken;
    }
    std::cout << " semantic_exact=" << (exactSemanticOk ? "PASS" : "FAIL")
              << " semantic_relaxed=" << (relaxedSemanticOk ? "PASS" : "FAIL")
              << " max_abs=" << maxAbs << " mean_abs=" << meanAbs
              << " mse=" << mse
              << " diff_errors=" << diffErrors << "/" << outputElements
              << " diff_error_rate=" << std::setprecision(4)
              << (outputElements > 0
                         ? 100.0 * static_cast<double>(diffErrors) /
                               static_cast<double>(outputElements)
                         : 0.0)
              << "% threshold=" << opts.diffThreshold
              << " status=" << (relaxedSemanticOk ? "PASS" : "FAIL") << std::endl;
    if (!topDiffs.empty()) {
      std::cout << "  top_logit_diffs:" << std::endl;
      for (const DiffItem &item : topDiffs) {
        std::cout << "    token=" << item.tokenIndex << " vocab=" << item.vocabIndex
                  << " flat_index=" << item.flatIndex
                  << " actual=" << item.actual
                  << " golden=" << item.golden
                  << " abs_diff=" << item.absDiff << std::endl;
      }
    }

    omTensorListDestroy(inputList);
    omTensorListDestroy(outputList);
  }

  const double tokenAcc = totalValidTokens > 0
                              ? static_cast<double>(totalMatchedTokens) /
                                    static_cast<double>(totalValidTokens)
                              : 0.0;
  const double overallMae = totalOutputElements > 0
                                ? totalAbsError / static_cast<double>(totalOutputElements)
                                : 0.0;
  const double overallMse = totalOutputElements > 0
                                ? totalSquaredError / static_cast<double>(totalOutputElements)
                                : 0.0;
  const double overallDiffErrorRate = totalOutputElements > 0
                                          ? static_cast<double>(totalDiffErrors) /
                                                static_cast<double>(totalOutputElements)
                                          : 0.0;
  std::cout << "Token match rate: " << totalMatchedTokens << "/" << totalValidTokens
            << " (" << std::fixed << std::setprecision(4) << tokenAcc * 100.0 << "%)"
            << std::endl;
  std::cout << "Exact semantic check (all valid tokens match golden): "
            << (runCount - exactSemanticErrors)
            << "/" << runCount << std::endl;
  std::cout << "Relaxed semantic check (token_acc >= " << opts.tokenAccThreshold
            << ", mismatches <= " << opts.maxTokenMismatches << "): "
            << (runCount - relaxedSemanticErrors)
            << "/" << runCount << std::endl;
  std::cout << "Numeric check (all logits <= " << opts.diffThreshold << "): "
            << (runCount - numericErrors)
            << "/" << runCount << std::endl;
  std::cout << "Overall numeric stats: max_abs=" << globalMaxAbs
            << " mean_abs=" << overallMae
            << " mse=" << overallMse
            << " diff_error_rate=" << overallDiffErrorRate * 100.0 << "%" << std::endl;

  const bool pass = (relaxedSemanticErrors == 0);
  std::cout << "@@MODEL_TEST_RESULT@@ errors=" << relaxedSemanticErrors << "/" << runCount
            << " status=" << (pass ? "PASS" : "FAIL") << std::endl;
  return pass ? 0 : 1;
}
