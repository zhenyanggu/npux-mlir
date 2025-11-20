

#include "src/Dialect/NpuMiddle/DialectBuilder.hpp"
#include "mlir/IR/Value.h"
#include "src/Dialect/NpuMiddle/NpuMiddleOps.hpp"

using namespace mlir;
using namespace npu_middle;

Value NpuMiddleBuilder::mvin(Value input, StringRef  memspace) {
  auto resultType = input.getType();
  return b()
      .create<NpuMiddleMvinOp>(loc(), resultType, input, memspace)
      .getResult();
}

Value NpuMiddleBuilder::mvout(Value input, StringRef  memspace) {
  auto resultType = input.getType();
  return b()
      .create<NpuMiddleMvoutOp>(loc(), resultType, input, memspace)
      .getResult();
}

Value NpuMiddleBuilder::gelu(Value input) {
  auto resultType = input.getType();
  return b()
      .create<NpuMiddleGeluOp>(loc(), resultType, input)
      .getResult();
}
