#include "npuMiddleDialect.h"
// #include "dialect/npu_common/npu_attrs.h"
#include "npuMiddleOps.h"
#include "src/Dialect/npuMiddle/ir/npuMiddleDialect.cpp.inc"


using namespace npu_middle;

void npuMiddleDialect::initialize() {
  // 下面的代码会生成 Op 的列表，专门用来初始化
  addOperations<
#define GET_OP_LIST
#include "src/Dialect/npuMiddle/ir/npuMiddle.cpp.inc"
  >();
}


#define GET_OP_CLASSES
#include "src/Dialect/npu_middle/ir/npu_middle.cpp.inc"
