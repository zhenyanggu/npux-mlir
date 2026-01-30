add_onnx_mlir_library(NpuBufferization
    NpuLowerPack.cpp
    
    LINK_LIBS PUBLIC
    OMONNXOps
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTransformUtils
    MLIRFuncDialect
    MLIRLinalgDialect
)
