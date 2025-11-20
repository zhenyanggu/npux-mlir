
#include "src/Dialect/NpuMiddle/NpuMiddleOps.hpp"

void npu_middle::NpuMiddleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/Dialect/NpuMiddle/NpuMiddleOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "src/Dialect/NpuMiddle/NpuMiddleOps.cpp.inc"

#include "src/Dialect/NpuMiddle/NpuMiddleDialect.cpp.inc"