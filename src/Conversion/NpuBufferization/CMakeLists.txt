add_onnx_mlir_library(NpuBufferization
    OneShotBufferize.cpp
    SignatureRewrite.cpp

    LINK_LIBS PUBLIC
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect
)