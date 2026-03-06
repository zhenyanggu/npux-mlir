add_onnx_mlir_library(NpuToLLVM
    NpuInline.cpp
    ConvertLinalgToNpux.cpp
    NpuxConversionHelper.cpp
    Npux/SfuOpConvert.cpp
    Npux/ComputeOpConvert.cpp
    Npux/NpuMemoryManagement.cpp
    NpuMemPlan.cpp
    Npux/InsertNpuLifecycle.cpp
    BufferDeallocTest.cpp
    NpuLowerSubview.cpp
    NpuxComputeFusion.cpp
    CustomBufferLoopHoist.cpp

    LINK_LIBS PUBLIC
    # --- 基础库 ---
    MLIRIR
    MLIRSupport
    MLIRPass

    # --- 转换工具 ---
    MLIRTransformUtils
    MLIRTransforms

    # --- Dialect 定义 ---
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRLLVMDialect
    MLIRBufferizationTransforms

    # --- 转换 Pass 依赖 ---
    MLIRMemRefTransforms


    MLIRLLVMCommonConversion

    OMCompilerOptions

)
