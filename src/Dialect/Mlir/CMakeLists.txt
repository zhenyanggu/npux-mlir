# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMMlirDialects
  IndexExpr.cpp
  IndexExprDetail.cpp
  IndexExprBuilder.cpp
  DialectBuilder.cpp
  VectorMachineSupport.cpp

  DEPENDS
  OMKrnlIncGen
  OMSpecializedKernelOpInterface

  LINK_LIBS PUBLIC
  OMCompilerOptions
  MLIRMathDialect
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRLLVMIRTransforms
  MLIRShapeDialect
  )

add_onnx_mlir_library(OMMlirDialects_clean
  IndexExpr.cpp
  IndexExprDetail.cpp
  IndexExprBuilder.cpp
  DialectBuilder.cpp
  VectorMachineSupport.cpp
  
  EXCLUDE_FROM_OM_LIBS
  DEPENDS
  OMKrnlIncGen
  OMSpecializedKernelOpInterface

  LINK_LIBS PUBLIC
  OMCompilerOptions_clean
  MLIRMathDialect
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRLLVMIRTransforms
  MLIRShapeDialect
  )
