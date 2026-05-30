add_onnx_mlir_library(NpuPatition
    ConvertONNXToNpucore.cpp
    NpuCoreConversionHelper.cpp
    
    NpuCore/Gelu.cpp
    NpuCore/LayerNorm.cpp
    NpuCore/MatAdd.cpp
    NpuCore/MatMul.cpp
    NpuCore/MaxPool.cpp
    NpuCore/Softmax.cpp
    NpuCore/Conv.cpp

    LINK_LIBS PUBLIC
    OMONNXOps
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTransformUtils
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRBufferizationDialect
    MLIRSCFDialect
    NpucoreOps
    OMONNXDialectCore
)
