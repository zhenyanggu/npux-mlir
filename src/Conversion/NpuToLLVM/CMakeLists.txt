add_onnx_mlir_library(NpuToLLVM
    NpuInstructionLowering.cpp
    NpuMemoryPlacement.cpp
    NpuInline.cpp
    NpuAlloc.cpp
    NpuSramPromotion.cpp
    NpuFinalizeLLVM.cpp

    ConvertLinalgToNpux.cpp
    NpuxConversionHelper.cpp
    SfuOpConvert.cpp
    NpuMemoryManagement.cpp
    NpuMemPlan.cpp
    InsertNpuLifecycle.cpp

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
