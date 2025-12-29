add_onnx_mlir_library(NpuToLLVM
    # 确保包含了你刚刚写的源文件
    NpuInstructionLowering.cpp
    NpuMemoryPlacement.cpp
    NpuInline.cpp
    NpuAlloc.cpp

    LINK_LIBS PUBLIC
    # --- 1. 基础核心库 ---
    MLIRIR
    MLIRSupport
    MLIRPass
    
    # --- 2. 转换工具库 (重要) ---
    # applyPatternsAndFoldGreedily (GreedyPatternRewriteDriver) 需要这个库
    MLIRTransformUtils 

    # --- 3. Dialect 库 ---
    # FuncOp, CallOp
    MLIRFuncDialect
    
    # AllocOp, CopyOp, SubViewOp
    MLIRMemRefDialect
    
    # [新增] GenericOp (你的代码里解析了 linalg.generic)
    MLIRLinalgDialect
    
    # [新增] ConstantOp (你的代码里用了 arith::ConstantOp 创建常量参数)
    MLIRArithDialect
    
    # [新增] ForOp (如果你处理了 scf.for，或者 bufferization 需要)
    MLIRSCFDialect

    MLIRTransforms
)