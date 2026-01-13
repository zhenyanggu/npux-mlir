#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "src/Pass/Passes.hpp"


void npux::registerBufferDeallocTest() {
  mlir::PassPipelineRegistration<>(
      "buffer-dealloc-test",                               // 命令行 Flag 名字
      "Only runs buildBufferDeallocationPipeline for testing", // 描述
      [](mlir::OpPassManager &pm) {
        // 这里完全只包含你想隔离测试的那一段 C++ 逻辑
        mlir::bufferization::BufferDeallocationPipelineOptions bufferDeallocOptions;
        mlir::bufferization::buildBufferDeallocationPipeline(pm, bufferDeallocOptions);
      });
}