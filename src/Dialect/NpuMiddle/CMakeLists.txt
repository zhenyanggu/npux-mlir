

add_onnx_mlir_dialect(NpuMiddle npu_middle)
add_onnx_mlir_dialect_doc(npu_middle NpuMiddle.td)

add_onnx_mlir_library(NpuMiddleOps
  DialectBuilder.cpp
  NpuMiddleOps.cpp

  DEPENDS
  OMNpuMiddleIncGen

  LINK_LIBS PUBLIC
  OMCompilerOptions
  OMONNXOps
  MLIRLLVMCommonConversion
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  )