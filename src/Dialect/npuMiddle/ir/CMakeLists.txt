add_mlir_dialect(npuMiddle npu_middle)
add_custom_target(npu_middle_inc_gen DEPENDS MLIRnpu_middleIncGen)
add_mlir_library(npu_middle
	npuMiddle.cpp
	DEPENDS
		MLIRnpu_middleIncGen
	# LINK_LIBS PUBLIC
	# 	npu_common
)
target_include_directories(npu_middle PRIVATE ${ONNX_MLIR_SRC_ROOT}/build)


