#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
class DialectRegistry;
}

namespace npux {
void registerONNXToNPUXPasses();

std::unique_ptr<mlir::Pass> createConvertONNXToNPUXPass();
} // namespace npux
