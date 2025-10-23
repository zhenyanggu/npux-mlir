# Generate TableGen'd dialect and op declarations/definitions for npux.
# This creates target MLIRnpuxIncGen and produces files like:
#   - dialect/npux/IR/npuxDialect.h.inc
#   - dialect/npux/IR/npuxDialect.cpp.inc
#   - dialect/npux/IR/npuxOps.h.inc
#   - dialect/npux/IR/npuxOps.cpp.inc
# include_directories(${CMAKE_BINARY_DIR}/src)
add_mlir_dialect(npux npux)
add_custom_target(npux_inc_gen DEPENDS MLIRnpuxIncGen)
add_mlir_library(npux
	npux.cpp
	DEPENDS
		MLIRnpuxIncGen
	# LINK_LIBS PUBLIC
	# 	npu_common
)
target_include_directories(npux PRIVATE ${ONNX_MLIR_SRC_ROOT}/build)
# message("include dir: ${ONNX_MLIR_SRC_ROOT}/build/src")

