add_onnx_mlir_library(NpuTiling
    ElemWise.cpp
    NpuTiling.cpp
    Conv.cpp
    Gemm.cpp
    Layout.cpp
    MaxPool.cpp
    NpuTilingHelper.cpp
    NpuInnerTiling.cpp
    ElemWiseInnerTiling.cpp
    ConvInnerTiling.cpp
    NpuInsertDma.cpp
    NpuOpSplitting.cpp
    ConvSpatialTiling.cpp
    RestoreAllocSpace.cpp
    RemoveRedundantDma.cpp


    LINK_LIBS PUBLIC
    # === 基础 MLIR 库 ===
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms          # 包含 GreedyPatternRewriteDriver

    # === Dialect 定义库 (对应你的 #include "mlir/Dialect/xxx") ===
    MLIRFuncDialect         # func::FuncOp
    MLIRLinalgDialect       # linalg::GenericOp
    MLIRSCFDialect          # scf::ForOp (Tiling 后生成的循环)
    MLIRTensorDialect       # tensor::ExtractSliceOp (Tiling 后生成的切片)

    # === 接口与工具库 (解决 undefined reference 报错的关键) ===
    MLIRTilingInterface     # fix: cast<TilingInterface>
    MLIRSCFTransforms       # fix: scf::tileUsingSCF
    MLIRDialectUtils        # fix: getAsOpFoldResult
    #MLIRLinalgTransforms
    OMCompilerOptions
)