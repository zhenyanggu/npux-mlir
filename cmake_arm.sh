#!/usr/bin/env bash

set -euo pipefail

NPUX_ENV_PREFIX="${NPUX_ENV_PREFIX:-${HOME}/.local/npux-env}"
LLVM_BUILD_ROOT_DEFAULT="${NPUX_ENV_PREFIX}/build/llvm-project"
BUILD_DIR="${BUILD_DIR:-build-arm}"
CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
ONNX_MLIR_CCACHE_BUILD="${ONNX_MLIR_CCACHE_BUILD:-OFF}"
HOST_ABSL_DIR="${absl_DIR:-${HOST_ABSL_DIR:-${NPUX_ENV_PREFIX}/lib/cmake/absl}}"
HOST_MLIR_DIR="${MLIR_DIR:-${HOST_MLIR_DIR:-${LLVM_BUILD_ROOT_DEFAULT}/lib/cmake/mlir}}"
HOST_LLVM_DIR="${LLVM_DIR:-${HOST_LLVM_DIR:-${LLVM_BUILD_ROOT_DEFAULT}/lib/cmake/llvm}}"
HOST_PY_INC="${HOST_PY_INC:-}"
HOST_PY_LIB="${HOST_PY_LIB:-}"

if [[ -z "${HOST_PY_INC}" || -z "${HOST_PY_LIB}" ]]; then
  mapfile -t python_paths < <(python3 - <<'PY'
import os
import glob
import sysconfig
include_dir = sysconfig.get_paths().get("include", "")
version = sysconfig.get_python_version()
major_minor = version.split(".")
lib_dirs = []
for key in ("LIBDIR", "LIBPL"):
    value = sysconfig.get_config_var(key) or ""
    if value and value not in lib_dirs:
        lib_dirs.append(value)

candidates = []
for key in ("LDLIBRARY", "LIBRARY"):
    lib_name = sysconfig.get_config_var(key) or ""
    for lib_dir in lib_dirs:
        if lib_name:
            candidates.append(os.path.join(lib_dir, lib_name))

shared_patterns = [
    f"libpython{version}.so",
    f"libpython{version}.so.*",
]
if len(major_minor) == 2:
    shared_patterns.extend([
        f"libpython{major_minor[0]}.so",
        f"libpython{major_minor[0]}.{major_minor[1]}.dylib",
    ])

for lib_dir in lib_dirs:
    for pattern in shared_patterns:
        candidates.extend(sorted(glob.glob(os.path.join(lib_dir, pattern))))

library = ""
for candidate in candidates:
    if candidate.endswith(".so") or ".so." in candidate or candidate.endswith(".dylib"):
        library = candidate
        break
if not library:
    for candidate in candidates:
        if os.path.exists(candidate):
            library = candidate
            break

print(include_dir)
print(library)
PY
)
  HOST_PY_INC="${HOST_PY_INC:-${python_paths[0]}}"
  HOST_PY_LIB="${HOST_PY_LIB:-${python_paths[1]}}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [[ "${CMAKE_GENERATOR}" == "Ninja" ]] && ! command -v ninja >/dev/null 2>&1; then
  CMAKE_GENERATOR="Unix Makefiles"
fi

cmake -G "${CMAKE_GENERATOR}" .. \
  -DCMAKE_TOOLCHAIN_FILE=../arm_toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNX_MLIR_ENABLE_PYRUNTIME_LIGHT=ON \
  -DONNX_MLIR_ENABLE_JNI=OFF \
  -DONNX_MLIR_CCACHE_BUILD="${ONNX_MLIR_CCACHE_BUILD}" \
  -Dabsl_DIR="${HOST_ABSL_DIR}" \
  -DMLIR_DIR="${HOST_MLIR_DIR}" \
  -DLLVM_DIR="${HOST_LLVM_DIR}" \
  -DPython3_INCLUDE_DIR="${HOST_PY_INC}" \
  -DPython3_LIBRARY="${HOST_PY_LIB}"
