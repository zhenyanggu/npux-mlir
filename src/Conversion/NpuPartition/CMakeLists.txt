add_onnx_mlir_library(NpuPatition
    ConvertONNXToLinalgNpu.cpp
    ONNXOpLabel.cpp
    NpuOutline.cpp
    ONNXOpConversion.cpp

    LINK_LIBS PUBLIC
    OMONNXOps
    NpuSupport
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRFuncDialect
)
