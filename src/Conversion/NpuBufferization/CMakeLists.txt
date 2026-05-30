add_onnx_mlir_library(NpuBufferization
    NpuBufferizationHelper.cpp
    
    LINK_LIBS PUBLIC
    OMONNXOps
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTransformUtils
    MLIRFuncDialect
    MLIRBufferizationTransforms
    MLIRBufferizationDialect
)
