

#ifndef NPU_MIDDLE_BUILDER_H
#define NPU_MIDDLE_BUILDER_H

#include "mlir/IR/Value.h"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/NpuMiddle/NpuMiddleOps.hpp"

namespace npu_middle {

// NpuMiddleBuilder 负责封装 NpuMiddle 方言操作的创建逻辑。
struct NpuMiddleBuilder : public onnx_mlir::DialectBuilder {
  // 构造函数：继承自基类 DialectBuilder
  NpuMiddleBuilder(mlir::Location loc) : onnx_mlir::DialectBuilder(loc) {}
  NpuMiddleBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : onnx_mlir::DialectBuilder(b, loc) {}
  NpuMiddleBuilder(const onnx_mlir::DialectBuilder &db)
      : onnx_mlir::DialectBuilder(db) {}
  virtual ~NpuMiddleBuilder() {}

  // 1. NpuMiddleMvinOp: 内存传输 (Memory In)
  // 参数:
  // - input: 要传输的数据 (mlir::Value, 对应 AnyMemRef:$input)
  // - memspace: 目标地址空间属性 (mlir::Attribute, 对应 AddressSpaceAttr:$memspace)
  mlir::Value mvin(mlir::Value input, mlir::StringRef  memspace);

  // 2. NpuMiddleMvoutOp: 内存传输 (Memory Out)
  // 参数:
  // - input: 要传输的数据 (mlir::Value, 对应 AnyMemRef:$input)
  // - memspace: 目标地址空间属性 (mlir::Attribute, 对应 AddressSpaceAttr:$memspace)
  mlir::Value mvout(mlir::Value input, mlir::StringRef  memspace);

  // 3. NpuMiddleGeluOp: GELU 激活函数
  // 参数:
  // - input: 激活函数的输入 (mlir::Value, 对应 AnyMemRef:$input)
  mlir::Value gelu(mlir::Value input);
};

} // namespace npu_middle

#endif // NPU_MIDDLE_BUILDER_H