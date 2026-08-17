#include "src/Compiler/NpuConfig.hpp"
#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp" // 根据你的项目路径调整
#include "src/Compiler/CompilerOptions.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/WithColor.h"
#include <cstdint>

using namespace llvm;

namespace {
const std::vector<int64_t> DEFAULT_GELU_TILE = {0, 0, 32, 32};
const std::vector<int64_t> DEFAULT_CONV_TILE = {0, 0, 32, 32};
const std::vector<int64_t> DEFAULT_MATMUL_TILE = {32, 32};

// 默认 SRAM 总大小 (128kB)
const std::string DEFAULT_SRAM_SIZE = "128kB";
} // namespace

namespace npux {

// ========================================================
// 1. 注册命令行参数
// ========================================================

cl::opt<std::string> npuConfigFile("npu-config",
    cl::desc("Path to Hardware Config JSON (defines physical sram_size etc.)"), 
    cl::value_desc("filename"),
    cl::cat(onnx_mlir::OnnxMlirCommonOptions));

cl::opt<std::string> npuTilingConfigFile("npu-tiling-config",
    cl::desc("Path to DSE Tiling Strategy JSON (defines spm_bytes, acc_bytes, layer params)"),
    cl::value_desc("filename"), 
    cl::cat(onnx_mlir::OnnxMlirCommonOptions));

cl::opt<std::string> cmdGeluTileSize("gelu-tile-size",
    cl::desc("Override Gelu tile size, e.g., [0,0,32,32]"),
    cl::cat(onnx_mlir::OnnxMlirCommonOptions));

cl::opt<std::string> cmdConvTileSize("conv-tile-size",
    cl::desc("Override conv tile size, e.g.,[0,0,32,32]"),
    cl::cat(onnx_mlir::OnnxMlirCommonOptions));

cl::opt<std::string> cmdMatMulTileSize("matmul-tile-size",
    cl::desc("Override matmul tile size, e.g.,[0,0,32,32]"),
    cl::cat(onnx_mlir::OnnxMlirCommonOptions));

// --- 内存覆盖参数 ---
cl::opt<std::string> cmdSramSize("npu-sram-size",
    cl::desc("Override TOTAL NPU SRAM size (fallback if spm/acc not set)"),
    cl::init(""), 
    cl::cat(onnx_mlir::OnnxMlirCommonOptions));

cl::opt<std::string> cmdSpmSize("npu-spm-size",
    cl::desc("Override SPM size (e.g. 64KB). Higest Priority."),
    cl::init(""), 
    cl::cat(onnx_mlir::OnnxMlirCommonOptions));

cl::opt<std::string> cmdAccSize("npu-acc-size",
    cl::desc("Override Accumulator size (e.g. 64KB). Higest Priority."),
    cl::init(""), 
    cl::cat(onnx_mlir::OnnxMlirCommonOptions));

cl::opt<NpuHardwareAbi> npuHardwareAbi("npu-hardware-abi",
    cl::desc("Select the NPU hardware ABI"),
    cl::values(
        clEnumValN(NpuHardwareAbi::Legacy, "legacy",
            "Use the existing npu_*_run runtime ABI"),
        clEnumValN(NpuHardwareAbi::VersaPDescriptorV1, "versa-p-v1",
            "Use the Versa-P 64-bit descriptor ABI")),
    cl::init(NpuHardwareAbi::Legacy),
    cl::cat(onnx_mlir::OnnxMlirCommonOptions));

cl::opt<std::string> emitConfigFile("emit-npu-config",
    cl::desc("Generate a default NPU configuration JSON file"),
    cl::value_desc("filename"), 
    cl::cat(onnx_mlir::OnnxMlirCommonOptions));


// ========================================================
// 2. 核心逻辑实现
// ========================================================

NPUConfig &NPUConfig::getInstance() {
  static NPUConfig instance;
  return instance;
}

bool useVersaPDescriptorAbi() {
  return npuHardwareAbi == NpuHardwareAbi::VersaPDescriptorV1;
}

NPUConfig::NPUConfig() {
  if (!emitConfigFile.empty()) {
    emitDefaultConfig(emitConfigFile);
    llvm::outs() << "NPU configuration template generated at: " << emitConfigFile << "\n";
    exit(0);
  }
}

void NPUConfig::emitDefaultConfig(const std::string &filename) {
  json::Object root;
  json::Object tileSizes;
  tileSizes["gelu"] = json::Array(DEFAULT_GELU_TILE);
  tileSizes["conv"] = json::Array(DEFAULT_CONV_TILE);
  tileSizes["matmul"] = json::Array(DEFAULT_MATMUL_TILE);
  root["tile_sizes"] = std::move(tileSizes);
  root["sram_size"] = DEFAULT_SRAM_SIZE;

  json::Value rootVal(std::move(root));
  std::error_code EC;
  raw_fd_ostream os(filename, EC, sys::fs::OF_Text);
  if (EC) {
    errs() << "Error opening file: " << EC.message() << "\n";
    return;
  }
  os << formatv("{0:2}", rootVal);
  os.close();
}

// --------------------------------------------------------
// JSON Parsing Helpers
// --------------------------------------------------------

void NPUConfig::parseLayerStrategy(const json::Object &root) {
  auto *layersArr = root.getArray("layers");
  if (!layersArr) return;

  for (const auto &val : *layersArr) {
    auto *layerObj = val.getAsObject();
    if (!layerObj) continue;

    auto name = layerObj->getString("layer_name");
    if (!name) continue;

    auto *params = layerObj->getObject("tile_params");
    if (!params) continue;

    LayerTileConfig config;
    if (auto v = params->getInteger("t_oh")) config.t_oh = *v; else continue;
    if (auto v = params->getInteger("t_ow")) config.t_ow = *v; else continue;
    if (auto v = params->getInteger("t_ic")) config.t_ic = *v; else continue;
    if (auto v = params->getInteger("t_oc")) config.t_oc = *v; else continue;

    layerStrategyMap[name.value()] = config;
  }
}

// [新增] 解析 hw_config: spm_bytes, acc_bytes
void NPUConfig::parseDseHwConfig(const json::Object &root) {
    auto *hwObj = root.getObject("hw_config");
    if (!hwObj) return;

    if (auto v = hwObj->getInteger("spm_bytes")) {
        dseSpmBytes = *v;
    }
    if (auto v = hwObj->getInteger("acc_bytes")) {
        dseAccBytes = *v;
    }
}

// --------------------------------------------------------
// Loaders
// --------------------------------------------------------

void NPUConfig::loadHardwareConfigIfNeeded() {
  std::lock_guard<std::mutex> lock(configMutex);
  if (isHwLoaded) return;

  if (npuConfigFile.empty()) {
    isHwLoaded = true;
    return;
  }

  auto bufferOrError = MemoryBuffer::getFile(npuConfigFile);
  if (!bufferOrError) {
    WithColor::error() << "Could not open NPU config file: " << npuConfigFile << "\n";
    exit(1);
  }

  auto parsed = json::parse(bufferOrError.get()->getBuffer());
  if (!parsed) {
    handleAllErrors(parsed.takeError(), [&](const ErrorInfoBase &E) {
      WithColor::error() << "Failed to parse JSON: " << E.message() << "\n";
    });
    exit(1);
  }

  if (auto *obj = parsed->getAsObject()) {
    jsonConfig = *obj;
    
    // 顺便解析一下 sram_size，缓存起来
    if (auto val = obj->get("sram_size")) {
        if (auto i = val->getAsInteger()) {
            hwSramBytes = *i;
        } else if (auto s = val->getAsString()) {
            hwSramBytes = parseMemorySize(s.value().str());
        }
    }
  }
  isHwLoaded = true;
}

void NPUConfig::loadTilingConfigIfNeeded() {
  std::lock_guard<std::mutex> lock(configMutex);
  if (isTilingLoaded) return;

  if (npuTilingConfigFile.empty()) {
    isTilingLoaded = true;
    return;
  }

  auto bufferOrError = MemoryBuffer::getFile(npuTilingConfigFile);
  if (!bufferOrError) {
    WithColor::error() << "Could not open Tiling Config file: " << npuTilingConfigFile << "\n";
    return;
  }

  auto parsed = json::parse(bufferOrError.get()->getBuffer());
  if (!parsed) {
    WithColor::error() << "Failed to parse Tiling JSON\n";
    return;
  }

  if (auto *obj = parsed->getAsObject()) {
    parseLayerStrategy(*obj);
    parseDseHwConfig(*obj); // [新增] 调用解析
  }
  isTilingLoaded = true;
}

// --------------------------------------------------------
// Memory Size Getters (Simplified & Robust)
// --------------------------------------------------------

// 私有辅助：获取物理总大小 (仅用于 Fallback)
int64_t NPUConfig::getRawHardwareTotalSram() {
    // 1. CLI Total Override
    if (cmdSramSize.getNumOccurrences() > 0 && !cmdSramSize.empty()) {
        return parseMemorySize(cmdSramSize);
    }
    // 2. HW Config JSON
    loadHardwareConfigIfNeeded();
    if (hwSramBytes.has_value()) {
        return hwSramBytes.value();
    }
    // 3. Default
    return parseMemorySize(DEFAULT_SRAM_SIZE);
}

int64_t NPUConfig::getSpmSize() {
    // 1. [Highest] Command Line Override
    if (cmdSpmSize.getNumOccurrences() > 0 && !cmdSpmSize.empty()) {
        return parseMemorySize(cmdSpmSize);
    }

    // 2. [Medium] Tiling JSON Strategy
    loadTilingConfigIfNeeded();
    if (dseSpmBytes.has_value()) {
        return dseSpmBytes.value();
    }

    // 3. [Fallback] Half of Total Hardware SRAM
    return getRawHardwareTotalSram() / 2;
}

int64_t NPUConfig::getAccSize() {
    // 1. [Highest] Command Line Override
    if (cmdAccSize.getNumOccurrences() > 0 && !cmdAccSize.empty()) {
        return parseMemorySize(cmdAccSize);
    }

    // 2. [Medium] Tiling JSON Strategy
    loadTilingConfigIfNeeded();
    if (dseAccBytes.has_value()) {
        return dseAccBytes.value();
    }

    // 3. [Fallback] Half of Total Hardware SRAM
    return getRawHardwareTotalSram() / 2;
}

int64_t NPUConfig::getSramSize() {
    // 直接返回两者之和，确保无论通过 CLI 还是 JSON 覆盖，总量都是自洽的
    return getSpmSize() + getAccSize();
}


// --------------------------------------------------------
// Other Utilities
// --------------------------------------------------------

std::optional<LayerTileConfig> NPUConfig::getCustomLayerConfig(StringRef layerName) {
  loadTilingConfigIfNeeded();
  auto it = layerStrategyMap.find(layerName);
  if (it != layerStrategyMap.end()) {
    return it->second;
  }
  return std::nullopt;
}

int64_t NPUConfig::parseMemorySize(const std::string &str) {
  StringRef s(str);
  s = s.trim();
  if (s.empty()) return -1;

  size_t numEnd = 0;
  while (numEnd < s.size() && (isdigit(s[numEnd]) || s[numEnd] == '.')) {
    numEnd++;
  }

  double val = 0.0;
  try {
    val = std::stod(std::string(s.substr(0, numEnd)));
  } catch (...) {
    WithColor::error() << "Invalid memory size format: " << str << "\n";
    exit(1);
  }

  StringRef suffix = s.substr(numEnd).trim();
  int64_t multiplier = 1;
  if (suffix.empty()) multiplier = 1;
  else if (suffix.equals_insensitive("k") || suffix.equals_insensitive("kb")) multiplier = 1024;
  else if (suffix.equals_insensitive("m") || suffix.equals_insensitive("mb")) multiplier = 1024 * 1024;
  else if (suffix.equals_insensitive("g") || suffix.equals_insensitive("gb")) multiplier = 1024 * 1024 * 1024;
  else {
    WithColor::error() << "Unknown memory unit in: " << str << ". Use KB, MB, or GB.\n";
    exit(1);
  }
  return static_cast<int64_t>(val * multiplier);
}

std::vector<int64_t> NPUConfig::parseVectorString(const std::string &str) {
  std::vector<int64_t> result;
  StringRef s(str);
  s = s.trim(" []");
  SmallVector<StringRef, 4> parts;
  s.split(parts, ',');
  for (auto p : parts) {
    long long val;
    if (!p.trim().getAsInteger(10, val)) {
      result.push_back(val);
    }
  }
  return result;
}

// Tile Getters
std::vector<int64_t> NPUConfig::getGeluTileSize() {
  if (cmdGeluTileSize.getNumOccurrences() > 0) return parseVectorString(cmdGeluTileSize);
  loadHardwareConfigIfNeeded();
  if (auto *tileMap = jsonConfig.getObject("tile_sizes")) {
    if (auto *arr = tileMap->getArray("gelu")) {
      std::vector<int64_t> vec;
      for (auto &v : *arr) if (auto i = v.getAsInteger()) vec.push_back(*i);
      if (!vec.empty()) return vec;
    }
  }
  return {};
}

std::vector<int64_t> NPUConfig::getMatMulTileSize() {
  if (cmdMatMulTileSize.getNumOccurrences() > 0) return parseVectorString(cmdMatMulTileSize);
  loadHardwareConfigIfNeeded();
  if (auto *tileMap = jsonConfig.getObject("tile_sizes")) {
    if (auto *arr = tileMap->getArray("matmul")) {
      std::vector<int64_t> vec;
      for (auto &v : *arr) if (auto i = v.getAsInteger()) vec.push_back(*i);
      if (!vec.empty()) return vec;
    }
  }
  return {};
}

std::vector<int64_t> NPUConfig::getConvTileSize() {
  if (cmdConvTileSize.getNumOccurrences() > 0) return parseVectorString(cmdConvTileSize);
  loadHardwareConfigIfNeeded();
  if (auto *tileMap = jsonConfig.getObject("tile_sizes")) {
    if (auto *arr = tileMap->getArray("conv")) {
      std::vector<int64_t> vec;
      for (auto &v : *arr) if (auto i = v.getAsInteger()) vec.push_back(*i);
      if (!vec.empty()) return vec;
    }
  }
  return {};
}

} // namespace npux
