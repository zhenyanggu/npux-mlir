#ifndef NPUX_TO_NPUMIDDLE_UTILS_H
#define NPUX_TO_NPUMIDDLE_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"

namespace npu_middle {
class NpuMiddleTypeConverter : public mlir::TypeConverter {
public:
  NpuMiddleTypeConverter();
  bool isSignatureLegal(mlir::FunctionType funcType) {
    return llvm::all_of(llvm::concat<const mlir::Type>(
                            funcType.getInputs(), funcType.getResults()),
        [this](mlir::Type type) { return isLegal(type); });
  }


  bool isSignatureLegal(mlir::func::CallOp call) {
    auto f = [this](mlir::Type type) { return isLegal(type); };
    return llvm::all_of(call.getOperandTypes(), f) &&
           llvm::all_of(call.getResultTypes(), f);
  }
};

void populateLoweringNpuxToNpuMiddlePatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);


    
// this function is here just to test if we're writing the whole project as MLIR
// is intended to
void populateLoweringNpuxTestPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);

} // namespace npu_middle

#endif