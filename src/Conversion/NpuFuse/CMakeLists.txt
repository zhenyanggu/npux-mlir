add_onnx_mlir_library(NpuFuse
    FuseHelper.cpp
    NpuFuse.cpp

    LINK_LIBS PUBLIC
    # 核心 MLIR 基础库
    MLIRIR              # 对应: mlir/IR/PatternMatch.h, mlir/IR/MLIRContext.h (Value, Type, Attribute等)
    MLIRPass            # 对应: mlir/Pass/Pass.h (PassWrapper, OperationPass)
    MLIRSupport         # 对应: llvm::errs(), 通用支持工具
    MLIRTransforms      # 对应: mlir/Transforms/GreedyPatternRewriteDriver.h (applyPatternsGreedily)

    # 具体 Dialect 库
    MLIRFuncDialect     # 对应: mlir/Dialect/Func/IR/FuncOps.h (func::FuncOp)
    MLIRLinalgDialect   # 对应: mlir/Dialect/Linalg/IR/Linalg.h (linalg::GenericOp)

    # ONNX-MLIR 项目特定库
    OMCompilerOptions   # 项目配置选项 (通常用于获取编译flag或项目全局设置)
)