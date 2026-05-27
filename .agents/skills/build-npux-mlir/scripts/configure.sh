#!/usr/bin/env bash

set -euo pipefail

# Return the repository root for this skill invocation.
repo_root() {
    git rev-parse --show-toplevel
}

# Configure the native build directory with the workspace's VSCode CMake settings.
run_configure() {
    local root
    root="$(repo_root)"

    export PATH="$HOME/tools/ccache:$PATH"
    export CCACHE_DIR="$HOME/.cache/ccache"

    cmake -S "$root" -B "$root/build" -G Ninja \
        -DMLIR_DIR=/home/pc/toolchain/llvm-project/build/lib/cmake/mlir \
        -DLLVM_DIR=/home/pc/toolchain/llvm-project/build/lib/cmake/llvm \
        -DCMAKE_PREFIX_PATH=/home/pc/miniconda3/envs/mlir \
        -DPython3_EXECUTABLE=/home/pc/miniconda3/envs/mlir/bin/python3 \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLLVM_ENABLE_ASSERTIONS=ON
}

# Execute the default configure workflow for this repository.
main() {
    run_configure
}

main "$@"
