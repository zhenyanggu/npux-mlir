add_onnx_mlir_library(NpuToLLVM
    NpuInstructionLowering.cpp
    NpuMemoryPlacement.cpp
    NpuInline.cpp
    NpuAlloc.cpp
    NpuSramPromotion.cpp
    NpuFinalizeLLVM.cpp # 确保加上了新文件

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
    
    # 【新增】必须添加这个，否则找不到 LLVMTypeConverter
    MLIRLLVMCommonConversion 
)