//=============================================================================
// /src/Support/NPUConversionRegistry.cpp
// this file implements the NPUConversionRegistry for registering
// NPU supported operations and their conversion patterns.
//=============================================================================


#include "src/Support/NPUConversionRegistry.hpp"

using namespace mlir;
using namespace npux;

std::map<std::string, NPUConversionRegistry::Entry>& NPUConversionRegistry::getRegistry() {
    static std::map<std::string, Entry> registry;
    return registry;
}

void NPUConversionRegistry::registerOp(StringRef opName, CheckFn check, AddPatternFn addPattern) {
    getRegistry()[opName.str()] = {check, addPattern};
}

// [新增] 实现动态添加 Illegal Ops
void NPUConversionRegistry::setIllegalOps(ConversionTarget& target, MLIRContext* context) {
    auto& registry = getRegistry();
    for (auto& [opName, entry] : registry) {
        // 使用 opName 字符串和 context 构造 OperationName，动态添加到 target
        target.addIllegalOp(OperationName(opName, context));
    }
}

void NPUConversionRegistry::populatePatterns(RewritePatternSet& patterns) {
    for (auto& [name, entry] : getRegistry()) {
        entry.addPattern(patterns);
    }
}
