//======================================================
// src/Conversion/NpuBufferization/NpuBufferizationHelper.hpp
//======================================================

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"

// 声明全局函数
mlir::LogicalResult RunNpuBufferization(mlir::ModuleOp module);

// 声明 Pattern 注册辅助函数
void populateBufferizationCleanUpHelperPatterns(mlir::RewritePatternSet &patterns);
