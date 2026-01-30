add_onnx_mlir_library(NpuPatition
    ConvertONNXToLinalgNpu.cpp
    NpuOutline.cpp
    LinalgConversionHelper.cpp
    NpuMerge.cpp
    NpuCleanPack.cpp
    NpuRegionExtension.cpp

    Linalg/LayerNorm.cpp
    Linalg/Conv.cpp
    Linalg/Unary.cpp
    Linalg/Gemm.cpp

    LINK_LIBS PUBLIC
    OMONNXOps
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
