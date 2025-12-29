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

bool NPUConversionRegistry::isSupported(Operation* op) {
    auto& registry = getRegistry();
    auto it = registry.find(op->getName().getStringRef().str());
    if (it == registry.end()) return false;
    return it->second.check(op);
}

void NPUConversionRegistry::populatePatterns(RewritePatternSet& patterns) {
    for (auto& [name, entry] : getRegistry()) {
        entry.addPattern(patterns);
    }
}