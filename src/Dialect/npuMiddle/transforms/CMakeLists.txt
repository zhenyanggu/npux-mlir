# Generate pass declarations for npux transforms.
set(LLVM_TARGET_DEFINITIONS npuxPasses.td)
mlir_tablegen(npuxPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRnpuxTransformsIncGen)

# Build the transforms library implementing NPUX passes.
add_mlir_library(npuxTransforms
	OutlineNpuxOps.cpp

	DEPENDS
		MLIRnpuxTransformsIncGen

	LINK_LIBS PRIVATE
		MLIRIR
		MLIRPass
		MLIRFuncDialect
)
target_include_directories(npuxTransforms PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(npuxTransforms PRIVATE ${ONNX_MLIR_SRC_ROOT}/build)
target_include_directories(npuxTransforms PRIVATE ${CMAKE_CURRENT_BINARY_DIR})