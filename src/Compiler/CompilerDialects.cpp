/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ CompilerDialects.cpp ------------------------===//

#include "CompilerDialects.hpp"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"

#include "mlir/InitAllDialects.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"

#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h" // Affine 也常用于边界计算
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"    // 报错的主角
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"

#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"

#include "src/Dialect/Npux/NpuxOps.hpp"
using namespace mlir;

namespace onnx_mlir {

DialectRegistry registerDialects(ArrayRef<accel::Accelerator::Kind> accels) {
  DialectRegistry registry;

  // Note that we cannot consult command line options because they have not yet
  // been parsed when registerDialects() is called.

  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<affine::AffineDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<vector::VectorDialect>();
  registry.insert<shape::ShapeDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<ONNXDialect>();
  registry.insert<KrnlDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<bufferization::BufferizationDialect>();
  registry.insert<npucore::NpucoreDialect>();
  registry.insert<npux::NpuxDialect>();

  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);

  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);

  arith::registerValueBoundsOpInterfaceExternalModels(registry);
  affine::registerValueBoundsOpInterfaceExternalModels(registry);
  scf::registerValueBoundsOpInterfaceExternalModels(registry);    // Fix: dialect 'scf'
  tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  vector::registerValueBoundsOpInterfaceExternalModels(registry);
  linalg::registerSubsetOpInterfaceExternalModels(registry);
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);

  

  registerOpenMPDialectTranslation(registry);
  mlir::memref::registerRuntimeVerifiableOpInterfaceExternalModels(registry);

  // Initialize accelerator(s) if required.
  accel::initAccelerators(accels);

  // Register dialects for accelerators.
  for (auto *accel : accel::Accelerator::getAccelerators())
    accel->registerDialects(registry);

  // Register interface needed by both old and new buffer deallocation pass.
  memref::registerAllocationOpInterfaceExternalModels(registry);
  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);

  return registry;
}

} // namespace onnx_mlir
