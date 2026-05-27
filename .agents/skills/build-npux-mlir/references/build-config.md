# Build Config

## Source Of Truth

This skill mirrors the native build settings from `.vscode/settings.json` in the repository root.

## Native Configure Settings

- Source directory: `${workspaceFolder}`
- Build directory: `${workspaceFolder}/build`
- Generator: `Ninja`
- `MLIR_DIR`: `/home/pc/toolchain/llvm-project/build/lib/cmake/mlir`
- `LLVM_DIR`: `/home/pc/toolchain/llvm-project/build/lib/cmake/llvm`
- `CMAKE_PREFIX_PATH`: `/home/pc/miniconda3/envs/mlir`
- `Python3_EXECUTABLE`: `/home/pc/miniconda3/envs/mlir/bin/python3`
- `CMAKE_BUILD_TYPE`: `Debug`
- `LLVM_ENABLE_ASSERTIONS`: `ON`
- `PATH` prefix: `~/tools/ccache`
- `CCACHE_DIR`: `~/.cache/ccache`
- Build tool args: `-j 8`

## Related Repository Entrypoints

- `cmake_arm.sh`: older ARM cross-compilation helper
- `cmake_arm64.sh`: ZCU102 or ARM64-oriented cross-compilation helper
- `交叉编译.md`: short note pointing to `cmake_arm64.sh`

Use the bundled scripts for the default native build. Use the repository scripts above only when the request is explicitly about cross compilation.
