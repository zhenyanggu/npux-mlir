# SPDX-License-Identifier: Apache-2.0



add_onnx_mlir_executable(npu-middle-opt
  npu-middle-opt.cpp

  LINK_LIBS PRIVATE

  OMONNXOps_clean

  MLIRIR
  MLIROpenMPToLLVMIRTranslation

  MLIRFuncDialect
  MLIROptLib
  npuMiddle
  ONNXToNPUX
  npuMiddleTransforms
  )

target_include_directories(npu-middle-opt PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}
)
  target_compile_definitions(npu-middle-opt PRIVATE NPU)