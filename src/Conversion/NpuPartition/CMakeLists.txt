add_onnx_mlir_library(NpuPatition
    ConvertONNXToLinalgNpu.cpp
    NpuOutline.cpp
    ONNXOpConversion.cpp
    NpuQuantHelper.cpp
    NpuMerge.cpp

    LINK_LIBS PUBLIC
    OMONNXOps
    NpuSupport
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTransformUtils
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRBufferizationDialect
    MLIRSCFDialect
    OMONNXDialectCore
)
