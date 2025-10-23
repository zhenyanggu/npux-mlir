#include "npuxDialect.h"
// #include "dialect/npu_common/npu_attrs.h"
#include "npuxOps.h"
#include "src/Dialect/npux/ir/npuxDialect.cpp.inc"


using namespace npux;

void npuxDialect::initialize() {
  // 下面的代码会生成 Op 的列表，专门用来初始化
  addOperations<
#define GET_OP_LIST
#include "src/Dialect/npux/ir/npux.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// NPUX Operator Definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Dialect/npux/ir/npux.cpp.inc"
