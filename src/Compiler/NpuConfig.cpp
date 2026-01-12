//=============================================
// src/Compiler/NpuConfig.cpp
// this file implements the NPU configuration management
//=============================================

#include "src/Compiler/NpuConfig.hpp"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/WithColor.h"
#include "src/Compiler/CompilerOptions.hpp"

using namespace llvm;


namespace {
    const std::vector<int64_t> DEFAULT_GELU_TILE = {0, 0, 32, 32};
    const std::vector<int64_t> DEFAULT_CONV_TILE = {1, 64, 32, 32};
    const std::vector<int64_t> DEFAULT_MATMUL_TILE = {32, 32};
    
    // 默认 SRAM 大小 (128kB)

    const std::string DEFAULT_SRAM_SIZE = "128kB"; 
}


namespace npux {

// 1. 注册命令行参数
// 这些参数会对 onnx-mlir 和 onnx-mlir-opt 全局可见
cl::opt<std::string> npuConfigFile("npu-config", 
    cl::desc("Path to NPU configuration JSON file"), 
    cl::value_desc("filename"),
    cl::cat(onnx_mlir::OnnxMlirCommonOptions)); 

cl::opt<std::string> cmdGeluTileSize("gelu-tile-size", 
    cl::desc("Override Gelu tile size, e.g., [0,0,32,32]"), 
    cl::cat(onnx_mlir::OnnxMlirCommonOptions)); 

cl::opt<std::string> cmdSramSize("npu-sram-size", 
    cl::desc("Override NPU SRAM size (e.g. 1024, 1KB, 1MB)"), 
    cl::init(""), // 默认为空字符串，表示未设置
    cl::cat(onnx_mlir::OnnxMlirCommonOptions));

cl::opt<std::string> emitConfigFile("emit-npu-config",
    cl::desc("Generate a default NPU configuration JSON file to the specified path"),
    cl::value_desc("filename"),
    cl::cat(onnx_mlir::OnnxMlirCommonOptions)); 

// 2. 单例实现
NPUConfig& NPUConfig::getInstance() {
    static NPUConfig instance;
    return instance;
}

NPUConfig::NPUConfig() {

    if (!emitConfigFile.empty()) {
        emitDefaultConfig(emitConfigFile);
        
        
        llvm::outs() << "NPU configuration template generated at: " << emitConfigFile << "\n";
        exit(0); 
    }
}

void NPUConfig::emitDefaultConfig(const std::string& filename) {
    json::Object root;
    
    json::Object tileSizes;
    tileSizes["gelu"] = json::Array(DEFAULT_GELU_TILE);     // 使用常量
    tileSizes["conv"] = json::Array(DEFAULT_CONV_TILE);     // 使用常量
    tileSizes["matmul"] = json::Array(DEFAULT_MATMUL_TILE); // 使用常量
    
    root["tile_sizes"] = std::move(tileSizes);
    
    // 使用易读的字符串形式写入默认值
    root["sram_size"] = DEFAULT_SRAM_SIZE; 

    json::Value rootVal(std::move(root));

    std::error_code EC;
    raw_fd_ostream os(filename, EC, sys::fs::OF_Text);
    if (EC) {
        errs() << "Error opening file for writing: " << EC.message() << "\n";
        return;
    }
    os << formatv("{0:2}", rootVal);
    os.close();
}

void NPUConfig::loadConfigIfNeeded() {
    std::lock_guard<std::mutex> lock(loadMutex);
    if (isLoaded) return;

    // 如果没有传入 config file，则跳过加载，视为全部使用默认或 CLI
    if (npuConfigFile.empty()) {
        isLoaded = true;
        return;
    }

    // 读取文件
    auto bufferOrError = MemoryBuffer::getFile(npuConfigFile);
    if (!bufferOrError) {
        WithColor::error() << "Could not open NPU config file: " << npuConfigFile << "\n";
        exit(1);
    }

    // 解析 JSON
    auto parsed = json::parse(bufferOrError.get()->getBuffer());
    if (!parsed) {
        handleAllErrors(parsed.takeError(), [&](const ErrorInfoBase &E) {
            WithColor::error() << "Failed to parse JSON: " << E.message() << "\n";
        });
        exit(1);
    }

    if (auto *obj = parsed->getAsObject()) {
        jsonConfig = *obj;
    } else {
        WithColor::error() << "JSON config must be an object\n";
        exit(1);
    }

    isLoaded = true;
}

int64_t NPUConfig::parseMemorySize(const std::string& str) {
    StringRef s(str);
    s = s.trim();
    if (s.empty()) return -1;

    // 1. 找到数字部分的结尾
    size_t numEnd = 0;
    while (numEnd < s.size() && (isdigit(s[numEnd]) || s[numEnd] == '.')) {
        numEnd++;
    }

    // 2. 解析数值
    double val = 0.0;
    StringRef numPart = s.substr(0, numEnd);
    // 使用 std::stod 解析浮点数 (比如 1.5MB)
    try {
        val = std::stod(std::string(numPart));
    } catch (...) {
        WithColor::error() << "Invalid memory size format: " << str << "\n";
        exit(1);
    }

    // 3. 解析后缀
    StringRef suffix = s.substr(numEnd).trim();
    int64_t multiplier = 1;

    if (suffix.empty()) {
        multiplier = 1; // 默认单位 bytes
    } else if (suffix.equals_insensitive("k") || suffix.equals_insensitive("kb")) {
        multiplier = 1024;
    } else if (suffix.equals_insensitive("m") || suffix.equals_insensitive("mb")) {
        multiplier = 1024 * 1024;
    } else if (suffix.equals_insensitive("g") || suffix.equals_insensitive("gb")) {
        multiplier = 1024 * 1024 * 1024;
    } else {
        WithColor::error() << "Unknown memory unit in: " << str << ". Use KB, MB, or GB.\n";
        exit(1);
    }

    return static_cast<int64_t>(val * multiplier);
}


// 辅助：解析字符串 "[0,0,32,32]" 到 vector
std::vector<int64_t> NPUConfig::parseVectorString(const std::string& str) {
    std::vector<int64_t> result;
    StringRef s(str);
    s = s.trim(" []"); // 去掉首尾括号
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

// =======================
// 优先级逻辑的核心实现
// =======================

std::vector<int64_t> NPUConfig::getGeluTileSize() {
    // 1. 检查命令行 Override (优先级最高)
    if (cmdGeluTileSize.getNumOccurrences() > 0) {
        return parseVectorString(cmdGeluTileSize);
    }

    // 2. 确保 JSON 已加载
    loadConfigIfNeeded();

    // 3. 检查 JSON 配置
    if (auto *tileMap = jsonConfig.getObject("tile_sizes")) {
        if (auto *arr = tileMap->getArray("gelu")) {
            std::vector<int64_t> vec;
            for (auto &v : *arr) {
                if (auto i = v.getAsInteger()) vec.push_back(*i);
            }
            if (!vec.empty()) return vec;
        }
    }

    // 4. 返回默认值 (Code Default)
    return DEFAULT_GELU_TILE; 
}

int64_t NPUConfig::getSramSize() {
    // 1. CLI Override (优先级最高)
    // 此时 cmdSramSize 是 string，需要解析
    if (cmdSramSize.getNumOccurrences() > 0 && !cmdSramSize.empty()) {
        return parseMemorySize(cmdSramSize);
    }

    loadConfigIfNeeded();

    // 2. JSON
    // JSON 里可能是整数 (1048576) 也可能是字符串 ("1MB")，需要双重判断
    if (auto val = jsonConfig.get("sram_size")) {
        if (auto i = val->getAsInteger()) {
            return *i; // 这是一个纯数字
        } else if (auto s = val->getAsString()) {
            return parseMemorySize(s.value().str()); // 这是一个带单位的字符串
        }
    }

    // 3. Default
    return parseMemorySize(DEFAULT_SRAM_SIZE); // 使用常量
}



} // namespace npux
