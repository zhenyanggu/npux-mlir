#pragma once
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h" // 需要引入这个以使用 ConversionTarget
#include <functional>
#include <map>
#include <string>

using namespace mlir;

namespace npux {

using CheckFn = std::function<bool(Operation*)>;
using AddPatternFn = std::function<void(RewritePatternSet&)>;

class NPUConversionRegistry {
public:
    static void registerOp(StringRef opName, CheckFn check, AddPatternFn addPattern);

    static void setIllegalOps(ConversionTarget& target, MLIRContext* context);

    static void populatePatterns(RewritePatternSet& patterns);

private:
    struct Entry {
        CheckFn check;
        AddPatternFn addPattern;
    };
    static std::map<std::string, Entry>& getRegistry();
};


template <typename ConverterT, typename OpT>
struct NPUOpRegistration {
    NPUOpRegistration() {
        CheckFn check = [](Operation* op) { return true; }; // 默认返回 true
        AddPatternFn addPattern = [](RewritePatternSet& patterns) {
            patterns.add<ConverterT>(patterns.getContext());
        };
        NPUConversionRegistry::registerOp(OpT::getOperationName(), check, addPattern);
    }
};

} // namespace npux