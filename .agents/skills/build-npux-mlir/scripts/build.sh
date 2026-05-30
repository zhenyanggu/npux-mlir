#!/usr/bin/env bash

#=============================================================================
# /.agents/skills/build-npux-mlir/scripts/build.sh
# this script configures and builds the local npux-mlir workspace using
# the repository's checked-in VS Code CMake settings.
#=============================================================================

set -euo pipefail

# Build the project from the workspace root so relative paths stay stable.
function main() {
  local workspace_root
  workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

  export PATH="${HOME}/tools/ccache:${PATH}"
  export CCACHE_DIR="${HOME}/.cache/ccache"

  cmake -S "${workspace_root}" -B "${workspace_root}/build" \
    -DMLIR_DIR=/home/pc/toolchain/llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=/home/pc/toolchain/llvm-project/build/lib/cmake/llvm \
    -DCMAKE_PREFIX_PATH=/home/pc/miniconda3/envs/mlir \
    -DPython3_EXECUTABLE=/home/pc/miniconda3/envs/mlir/bin/python3 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_ASSERTIONS=ON

  cmake --build "${workspace_root}/build" -- -j 8
}

main "$@"
