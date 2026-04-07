# SPDX-License-Identifier: Apache-2.0

set(NPUX_MLIR_DRIVER_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/npux-mlir-driver")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/npux-mlir-driver.py.in"
  "${NPUX_MLIR_DRIVER_SCRIPT}"
  @ONLY
)

file(CHMOD
  "${NPUX_MLIR_DRIVER_SCRIPT}"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

add_custom_target(npux-mlir-driver ALL DEPENDS "${NPUX_MLIR_DRIVER_SCRIPT}" onnx-mlir)
