#pragma once
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/ADT/StringMap.h"
#include <mutex>
#include <vector>
#include <string>
#include <optional>

namespace npux {

// 1. 定义全局命令行选项引用
extern llvm::cl::opt<std::string> npuConfigFile;
extern llvm::cl::opt<std::string> npuTilingConfigFile;
extern llvm::cl::opt<std::string> cmdGeluTileSize;
extern llvm::cl::opt<std::string> cmdConvTileSize;
extern llvm::cl::opt<std::string> cmdMatMulTileSize;

// [修改] SRAM 相关的命令行现在有三个
extern llvm::cl::opt<std::string> cmdSramSize; // 总大小 (备用/Fallback)
extern llvm::cl::opt<std::string> cmdSpmSize;  // [新增] 强制指定 SPM
extern llvm::cl::opt<std::string> cmdAccSize;  // [新增] 强制指定 ACC

struct LayerTileConfig {
    int64_t t_oh;
    int64_t t_ow;
    int64_t t_ic;
    int64_t t_oc;
};

class NPUConfig {
public:
    static NPUConfig& getInstance();

    // 内存获取逻辑 (已简化)
    int64_t getSpmSize();  // 优先级: CLI > DSE JSON > Default/2
    int64_t getAccSize();  // 优先级: CLI > DSE JSON > Default/2
    int64_t getSramSize(); // 返回 getSpmSize() + getAccSize()

    // Tile Size 获取
    std::vector<int64_t> getGeluTileSize();
    std::vector<int64_t> getConvTileSize();
    std::vector<int64_t> getMatMulTileSize();

    // Layer Config 获取
    std::optional<LayerTileConfig> getCustomLayerConfig(llvm::StringRef layerName);
    
    // 加载逻辑
    void loadHardwareConfigIfNeeded();
    void loadTilingConfigIfNeeded();

private:
    NPUConfig();

    // 辅助函数
    std::vector<int64_t> parseVectorString(const std::string& str);
    int64_t parseMemorySize(const std::string& str);
    void emitDefaultConfig(const std::string& filename);
    
    // 解析 JSON 片段
    void parseLayerStrategy(const llvm::json::Object& root);
    void parseDseHwConfig(const llvm::json::Object& root); // [新增] 解析 hw_config

    // 内部 Fallback 逻辑：获取单纯的硬件总大小 (不涉及 SPM/ACC 划分)
    int64_t getRawHardwareTotalSram(); 

    bool isHwLoaded = false;
    bool isTilingLoaded = false;
    std::mutex configMutex;

    // 存储解析值
    std::optional<int64_t> dseSpmBytes; // From Tiling JSON "spm_bytes"
    std::optional<int64_t> dseAccBytes; // From Tiling JSON "acc_bytes"
    std::optional<int64_t> hwSramBytes; // From Hardware JSON "sram_size"
    
    llvm::json::Object jsonConfig;
    llvm::StringMap<LayerTileConfig> layerStrategyMap;
};

} // namespace npux