//==========================================
//src/Dialect/Npux/NpuxOps.cpp
//this file defines npux Dialect Operations
//==========================================

#include "src/Dialect/Npux/NpuxOps.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace npux;

// ==========================================================
// Dialect 初始化
// ==========================================================
#include "src/Dialect/Npux/NpuxDialect.cpp.inc"

void NpuxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/Dialect/Npux/NpuxOps.cpp.inc"
  >();
  
}


#include "src/Dialect/Npux/NpuxEnums.cpp.inc"

// Attributes 实现
#define GET_ATTRDEF_CLASSES
#include "src/Dialect/Npux/NpuxAttributes.cpp.inc"

// Op 实现
#define GET_OP_CLASSES
#include "src/Dialect/Npux/NpuxOps.cpp.inc"