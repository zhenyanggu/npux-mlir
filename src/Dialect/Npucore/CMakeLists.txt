set(LLVM_TARGET_DEFINITIONS Npucore.td)

add_onnx_mlir_dialect(Npucore npucore)
add_onnx_mlir_dialect_doc(npucore Npucore.td)

add_onnx_mlir_library(NpucoreOps
  NpucoreOps.cpp

  DEPENDS
  OMNpucoreIncGen
  OMNpucoreStructuredOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRTilingInterface
  OMNpucoreStructuredOpInterface
  )
