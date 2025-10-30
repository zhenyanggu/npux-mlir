# SPDX-License-Identifier: Apache-2.0

#get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)


add_onnx_mlir_executable(npux-opt
  npux-opt.cpp

  LINK_LIBS PRIVATE
#  ${OMLibs}


#  OMAccelerator
#  OMInitAccelerators
#  OMKrnlOps
  OMONNXOps_clean




  MLIRIR
  MLIROpenMPToLLVMIRTranslation

  MLIRFuncDialect
  MLIROptLib
  npux
  ONNXToNPUX
  npuxTransforms
  )

target_include_directories(npux-opt PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}
)
  target_compile_definitions(npux-opt PRIVATE NPU)