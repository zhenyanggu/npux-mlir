#ifndef NPU_MIDDLE_OPS_H
#define NPU_MIDDLE_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"


#include "src/Dialect/NpuMiddle/NpuMiddleDialect.hpp.inc"

#define GET_OP_CLASSES
#include "src/Dialect/NpuMiddle/NpuMiddleOps.hpp.inc"






#endif