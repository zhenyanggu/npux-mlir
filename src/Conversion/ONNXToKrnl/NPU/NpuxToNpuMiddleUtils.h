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
#include "src/Dialect/npux/ir/npuxOps.h"
#include "src/Dialect/NpuMiddle/NpuMiddleOps.hpp"
#include "src/Dialect/NpuMiddle/DialectBuilder.hpp"

namespace npu_middle {

void populateLoweringNpuxToNpuMiddlePatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);


    
// this function is here just to test if we're writing the whole project as MLIR
// is intended to
void populateLoweringNpuxTestPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);

} // namespace npu_middle

#endif