//===============================================
// src/Dialect/Npucore/NpucoreOps.hpp
// this file declares npucore dialect ops.
//===============================================

#ifndef NPUCOREOPS_HPP
#define NPUCOREOPS_HPP

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "src/Interface/NpucoreStructuredOpInterface.hpp"

#include "src/Dialect/Npucore/NpucoreDialect.hpp.inc"

#define GET_OP_CLASSES
#include "src/Dialect/Npucore/NpucoreOps.hpp.inc"

#endif
