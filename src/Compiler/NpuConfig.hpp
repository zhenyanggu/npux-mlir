//=============================================
// src/Compiler/NpuConfig.hpp
//=============================================
#pragma once
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/ADT/StringMap.h"
#include <mutex>
#include <vector>
#include <string>


namespace npux {

// 1. 定义全局命令行选项引用 (在 .cpp 中具体定义)
extern llvm::cl::opt<std::string> npuConfigFile;
extern llvm::cl::opt<std::string> cmdGeluTileSize; // 接收 "[0,0,32,32]" 字符串
extern llvm::cl::opt<std::string> cmdSramSize;

class NPUConfig {
public:
    static NPUConfig& getInstance();

    // 获取 Gelu Tile Size
    // 逻辑：CLI > JSON > Default
    std::vector<int64_t> getGeluTileSize();
    
    // 获取 Conv Tile Size
    std::vector<int64_t> getConvTileSize();

    // 获取 SRAM Size
    int64_t getSramSize();

    void loadConfigIfNeeded();

private:
    NPUConfig(); // 私有构造
    
    // 加载 JSON 的核心函数
    
    
    // 辅助函数：解析 "[1,2,3]" 字符串
    std::vector<int64_t> parseVectorString(const std::string& str);
    int64_t parseMemorySize(const std::string& str);
    void emitDefaultConfig(const std::string& filename);

    bool isLoaded = false;
    std::mutex loadMutex;
    
    // 存储解析后的 JSON 数据
    // 这里简单起见用了 LLVM 的 json::Value，你也可以解析成 struct
    llvm::json::Object jsonConfig;
};

} // namespace npux
