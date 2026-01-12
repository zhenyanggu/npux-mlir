
set(LLVM_TARGET_DEFINITIONS Npux.td)

# 2. 生成 Enum 的声明 (.h.inc) 和定义 (.cpp.inc)
mlir_tablegen(NpuxEnums.h.inc -gen-enum-decls)
mlir_tablegen(NpuxEnums.cpp.inc -gen-enum-defs)

# 3. 添加生成目标，方便其他库依赖
add_public_tablegen_target(OMNpuxEnums)
# 2. 原有的 Dialect 定义
add_onnx_mlir_dialect(Npux npux)
add_onnx_mlir_dialect_doc(npux Npux.td)

# 3. 修改库定义，添加对 OMNpuxEnums 的依赖
add_onnx_mlir_library(NpuxOps
  NpuxOps.cpp

  DEPENDS
  OMNpuxIncGen
  OMNpuxEnums  # [新增] 必须依赖 Enum 生成目标

  LINK_LIBS PUBLIC
  OMCompilerOptions
  MLIRLLVMCommonConversion
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  )