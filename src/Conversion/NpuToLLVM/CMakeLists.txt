add_onnx_mlir_library(NpuToLLVM
    NpuInline.cpp
    ConvertLinalgToNpux.cpp
    NpuxConversionHelper.cpp
    VersaPDescriptor.cpp
    VersaPBankScheduler.cpp
    VersaPStaticEligibility.cpp
    VersaPDmaRegion.cpp
    VersaPRegionMark.cpp
    Npux/SfuOpConvert.cpp
    Npux/ComputeOpConvert.cpp
    Npux/NpuMemoryManagement.cpp
    NpuMemPlan.cpp
    Npux/InsertNpuLifecycle.cpp
    BufferDeallocTest.cpp
    NpuLowerSubview.cpp
    NpuxComputeFusion.cpp
    CustomBufferLoopHoist.cpp
    RemoveDuplicateMvinBias.cpp
    EraseNpuMemorySpace.cpp

    DEPENDS
    OMNpuxIncGen
    OMNpuxEnums

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
    NpuxOps

)

if(BUILD_TESTING)
  add_onnx_mlir_executable(VersaPDescriptorTest
    VersaPDescriptorTest.cpp
    NO_INSTALL
    LINK_LIBS PRIVATE
    NpuToLLVM
    LLVMSupport
    )
  add_test(NAME VersaPDescriptorTest COMMAND VersaPDescriptorTest)
  add_onnx_mlir_executable(VersaPBankSchedulerTest
    VersaPBankSchedulerTest.cpp
    NO_INSTALL
    LINK_LIBS PRIVATE
    NpuToLLVM
    LLVMSupport
    )
  add_test(NAME VersaPBankSchedulerTest COMMAND VersaPBankSchedulerTest)
endif()
