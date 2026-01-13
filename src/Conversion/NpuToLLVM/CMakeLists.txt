add_onnx_mlir_library(NpuToLLVM
    NpuInline.cpp
    ConvertLinalgToNpux.cpp
    NpuxConversionHelper.cpp
    SfuOpConvert.cpp
    NpuMemoryManagement.cpp
    NpuMemPlan.cpp
    InsertNpuLifecycle.cpp
    NpuSramPromotion.cpp
    BufferDeallocTest.cpp

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

    # --- 转换 Pass 依赖 ---
    MLIRMemRefTransforms


    MLIRLLVMCommonConversion

    OMCompilerOptions

)
