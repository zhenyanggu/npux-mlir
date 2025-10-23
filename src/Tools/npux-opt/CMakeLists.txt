# SPDX-License-Identifier: Apache-2.0

get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

add_onnx_mlir_executable(npux-opt
  npux-opt.cpp

  LINK_LIBS PRIVATE
  ${OMLibs}
  # OMCompilerDialects
  # onnx
  OMONNXDialectCore
  # OMCompilerOptions
  # OMCompilerPasses
  # OMAccelerator
  # OMVersion
  # MLIRAffineTransforms
  # MLIRLinalgTransforms
  # MLIRMemRefTransforms
  # TOSA方言
  MLIRFuncDialect
  MLIROptLib
  npux
  ONNXToNPUX
  )
  target_compile_definitions(npux-opt PRIVATE NPU)
