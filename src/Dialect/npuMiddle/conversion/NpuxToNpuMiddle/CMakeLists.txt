# Header-only, but ensure build tree has path
add_mlir_library(NpuxToNpuMiddle
  ConvertONNXToNPUX.cpp
  LINK_LIBS PRIVATE
    MLIRIR MLIRPass MLIRTransforms
    MLIRFuncDialect
    # ONNX op definitions (TypeIDs for ONNX ops like ONNXGeluOp)
    # ONNXMLIR::OMONNXOps
    # Our dialect
    NpuMiddle
)
target_include_directories(ONNXToNPUX PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(ONNXToNPUX PRIVATE ${ONNX_MLIR_SRC_ROOT}/build)