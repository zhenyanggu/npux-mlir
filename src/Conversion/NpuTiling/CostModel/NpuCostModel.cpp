
//======================================================
// src/Conversion/NpuTiling/CostModel/NpuCostModel.cpp
// This file implements the NPUCostModel class, which provides methods to
// calculate optimal tile sizes for different types of operations (Matmul,
// Conv2D, Elementwise based on the NPU's hardware constraints like SRAM size
// and number of MACs).
//======================================================
#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h" // 引入 report_fatal_error

using namespace mlir;

namespace npux {

llvm::SmallVector<int64_t> NPUCostModel::getOptimalTileSizes(
    mlir::linalg::LinalgOp op) {
  return llvm::TypeSwitch<mlir::Operation *, llvm::SmallVector<int64_t>>(
      op.getOperation())
      .Case<mlir::linalg::GenericOp>([&](mlir::linalg::GenericOp genericOp) {
        auto libCallAttr =
            genericOp->getAttrOfType<mlir::StringAttr>("library_call");
        if (!libCallAttr) {
          genericOp.emitOpError("missing 'library_call' attribute.");
          llvm::report_fatal_error("NPU CostModel: Unidentifiable GenericOp.");
        }
        llvm::StringRef libCall = libCallAttr.getValue();

        // 1. 定义成员函数指针类型
        using CostModelFunc = llvm::SmallVector<int64_t> (NPUCostModel::*)(mlir::linalg::GenericOp);

        // 2. 构建静态路由表
        static const llvm::StringMap<CostModelFunc> dispatchTable = {
            {"npu_conv",                   &NPUCostModel::getConv2DTileSizes},
            {"npu_matmul",                 &NPUCostModel::getGemmTileSizes},
            {"npu_gemm",                   &NPUCostModel::getGemmTileSizes},
            {"npu_gelu",                   &NPUCostModel::getGeluTileSizes},
            {"npu_matadd",                 &NPUCostModel::getMatAddTileSizes},
            {"npu_layout_nchw_to_nchwc32", &NPUCostModel::getLayoutTileSizes},
            {"npu_layout_nchwc32_to_nchw", &NPUCostModel::getLayoutTileSizes},
            {"npu_transpose",              &NPUCostModel::getLayoutTileSizes},
            {"npu_maxpool",                &NPUCostModel::getMaxPoolTileSizes}
        };

        // 3. 查表并调用
        auto it = dispatchTable.find(libCall);
        if (it != dispatchTable.end()) {
          CostModelFunc func = it->second;
          return (this->*func)(genericOp); // 成员函数指针调用
        }

        // 4. Fallback 错误处理
        genericOp.emitOpError("has unsupported library_call: ") << libCall;
        llvm::report_fatal_error("NPU CostModel: Unsupported library_call.");
        return llvm::SmallVector<int64_t>{};
      })
      .Default([&](mlir::Operation *unknownOp) -> llvm::SmallVector<int64_t> { // 明确指定返回类型或在末尾添加 return
        // 处理所有其他未定义的 LinalgOp 类型
        unknownOp->emitOpError(
            "is an unsupported operation type for NPU tiling.");
        llvm::report_fatal_error("NPU CostModel: Unknown operation type.");
        return llvm::SmallVector<int64_t>{};
      });
}

} // namespace npux
