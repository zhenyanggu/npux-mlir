// src/Dialect/Npux/NpuxOps.hpp
#ifndef NPUXOPS_HPP
#define NPUXOPS_HPP

#include <optional>
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// 1. Dialect 声明
#include "src/Dialect/Npux/NpuxDialect.hpp.inc"

#include "src/Dialect/Npux/NpuxEnums.h.inc"


#define GET_ATTRDEF_CLASSES
#include "src/Dialect/Npux/NpuxAttributes.hpp.inc"

// 4. Op 声明
#define GET_OP_CLASSES
#include "src/Dialect/Npux/NpuxOps.hpp.inc"

#endif // NPUXOPS_HPP