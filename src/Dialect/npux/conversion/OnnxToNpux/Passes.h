#pragma once
// Minimal ONNX -> NPUX conversion pass declarations
#include "mlir/Pass/Pass.h"

namespace mlir {
class DialectRegistry;
}

namespace npux {
void registerONNXToNPUXPasses();

std::unique_ptr<mlir::Pass> createConvertONNXToNPUXPass();
} // namespace npux
