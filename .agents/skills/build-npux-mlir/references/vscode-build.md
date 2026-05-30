# VS Code Build Reference

This file mirrors the build-related entries from `.vscode/settings.json` in the repository root.

## Configure settings

- `cmake.sourceDirectory=${workspaceFolder}`
- `cmake.buildDirectory=${workspaceFolder}/build`
- `MLIR_DIR=/home/pc/toolchain/llvm-project/build/lib/cmake/mlir`
- `LLVM_DIR=/home/pc/toolchain/llvm-project/build/lib/cmake/llvm`
- `CMAKE_PREFIX_PATH=/home/pc/miniconda3/envs/mlir`
- `Python3_EXECUTABLE=/home/pc/miniconda3/envs/mlir/bin/python3`
- `CMAKE_BUILD_TYPE=Debug`
- `LLVM_ENABLE_ASSERTIONS=ON`

## Build arguments

- `cmake.buildToolArgs=["-j", "8"]`

## Configure environment

- `PATH=~/tools/ccache:${env:PATH}`
- `CCACHE_DIR=~/.cache/ccache`
