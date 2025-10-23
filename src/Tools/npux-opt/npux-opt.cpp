#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/InitLLVM.h"
//func方言
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/npux/ir/npuxDialect.h"
#include "src/Dialect/npux/conversion/OnnxToNpux/Passes.h"
#define NPUX_OPT
using namespace mlir;
using namespace llvm;

int main(int argc, char **argv) {


  DialectRegistry registry;
  registry.insert<ONNXDialect, npux::npuxDialect, func::FuncDialect>();
  npux::registerONNXToNPUXPasses();
  return asMainReturnCode(MlirOptMain(argc, argv, "npux-opt", registry));
}