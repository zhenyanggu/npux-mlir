#pragma once
#include "mlir/IR/PatternMatch.h"
#include <functional>
#include <map>
#include <string>

using namespace mlir;

namespace npux {

// 定义两个回调函数的类型
using CheckFn = std::function<bool(Operation*)>;
using AddPatternFn = std::function<void(RewritePatternSet&)>;

class NPUConversionRegistry {
public:
    // 注册 Op 的支持逻辑
    static void registerOp(StringRef opName, CheckFn check, AddPatternFn addPattern);

    // 供 Partition Pass 使用：检查 Op 是否支持
    static bool isSupported(Operation* op);

    // 供 Lowering Pass 使用：加载所有转换 Pattern
    static void populatePatterns(RewritePatternSet& patterns);

private:
    struct Entry {
        CheckFn check;
        AddPatternFn addPattern;
    };
    static std::map<std::string, Entry>& getRegistry();
};

// --- 自动化注册器 ---
// 这里是模板类，放在头文件里方便各个 .cpp 使用
template <typename ConverterT, typename OpT>
struct NPUOpRegistration {
    NPUOpRegistration() {
        // 1. 构造检查函数
        CheckFn check = [](Operation* op) {
            // 这里假设 ConverterT 里有一个静态方法 isHardwareSupported
            // 如果你不想每个类都写这个方法，可以用 SFINAE 技术做个默认返回 true 的模板元编程，但这里简单起见先硬性要求
            return ConverterT::isHardwareSupported(cast<OpT>(op));
        };

        // 2. 构造添加 Pattern 函数
        AddPatternFn addPattern = [](RewritePatternSet& patterns) {
            patterns.add<ConverterT>(patterns.getContext());
        };

        // 3. 注册到单例 Map 中
        NPUConversionRegistry::registerOp(OpT::getOperationName(), check, addPattern);
    }
};

} // namespace npux
