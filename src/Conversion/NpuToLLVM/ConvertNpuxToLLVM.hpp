//================================================
// src/Conversion/NpuToLLVM/ConvertNpuxToLLVM.hpp
// this file declares npu to llvm conversion patterns
// which will be used in convert-krnl-to-llvm pass
//================================================

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "src/Dialect/Npux/NpuxOps.hpp"

#pragma once

namespace npux {
void populateNpuxToLLVMConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter);
}
