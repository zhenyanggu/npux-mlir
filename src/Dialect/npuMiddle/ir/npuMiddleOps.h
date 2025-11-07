// Defines npux operations.

#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
// #include "mlir/IR/AttrTypeBase.h"
// #include "mlir/IR/OpBase.h"
// #include "dialect/npu_common/npu_attrs.h"

#include "npuxDialect.h"
// #include "dialect/npux/ir/npuxOps.h.inc"
#define GET_OP_CLASSES
#include "src/Dialect/npux/ir/npux.h.inc"