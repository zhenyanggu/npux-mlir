#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}" && pwd)

NPUX_ENV_PREFIX="${NPUX_ENV_PREFIX:-${HOME}/.local/npux-env}"
LLVM_BUILD_ROOT_DEFAULT="${NPUX_ENV_PREFIX}/build/llvm-project"
BUILD_DIR="${BUILD_DIR:-build-zcu102}"
TOOLCHAIN_FILE="${TOOLCHAIN_FILE:-zcu102_toolchain.cmake}"
CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
ONNX_MLIR_CCACHE_BUILD="${ONNX_MLIR_CCACHE_BUILD:-OFF}"
HOST_ABSL_DIR="${absl_DIR:-${HOST_ABSL_DIR:-${NPUX_ENV_PREFIX}/lib/cmake/absl}}"
HOST_MLIR_DIR="${MLIR_DIR:-${HOST_MLIR_DIR:-${LLVM_BUILD_ROOT_DEFAULT}/lib/cmake/mlir}}"
HOST_LLVM_DIR="${LLVM_DIR:-${HOST_LLVM_DIR:-${LLVM_BUILD_ROOT_DEFAULT}/lib/cmake/llvm}}"
JOBS="${NPUX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

die() {
  echo -e "${RED}[Error] $*${NC}" >&2
  exit 1
}

find_first_executable() {
  local candidate
  for candidate in "$@"; do
    [[ -z "${candidate}" ]] && continue
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

detect_python() {
  python3 - <<'PY'
import os
import glob
import sysconfig
include_dir = sysconfig.get_paths().get("include", "")
version = f"{sysconfig.get_python_version()}"
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
}

echo ">>> Starting ZCU102 Cross-Compilation Setup"

CXX_COMPILER=$(find_first_executable \
  "${CROSS_CXX:-}" \
  "${NPUX_ENV_PREFIX}/toolchains/aarch64/bin/aarch64-none-linux-gnu-g++" \
  "${NPUX_ENV_PREFIX}/toolchains/aarch64/bin/aarch64-linux-gnu-g++" \
  "/usr/bin/aarch64-linux-gnu-g++" \
  "aarch64-none-linux-gnu-g++" \
  "aarch64-linux-gnu-g++") || die "cross C++ compiler not found; activate the local environment or install a toolchain"

C_COMPILER=$(find_first_executable \
  "${CROSS_CC:-}" \
  "${NPUX_ENV_PREFIX}/toolchains/aarch64/bin/aarch64-none-linux-gnu-gcc" \
  "${NPUX_ENV_PREFIX}/toolchains/aarch64/bin/aarch64-linux-gnu-gcc" \
  "/usr/bin/aarch64-linux-gnu-gcc" \
  "aarch64-none-linux-gnu-gcc" \
  "aarch64-linux-gnu-gcc") || die "cross C compiler not found; activate the local environment or install a toolchain"

mapfile -t python_paths < <(detect_python)
HOST_PY_INC="${HOST_PY_INC:-${python_paths[0]}}"
HOST_PY_LIB="${HOST_PY_LIB:-${python_paths[1]}}"

[[ -d "${HOST_PY_INC}" ]] || die "Python include dir not found: ${HOST_PY_INC}"
[[ -f "${HOST_PY_LIB}" ]] || die "Python shared library not found: ${HOST_PY_LIB}"
[[ -d "${HOST_ABSL_DIR}" ]] || die "absl cmake dir not found: ${HOST_ABSL_DIR}"
[[ -d "${HOST_MLIR_DIR}" ]] || die "MLIR cmake dir not found: ${HOST_MLIR_DIR}"
[[ -d "${HOST_LLVM_DIR}" ]] || die "LLVM cmake dir not found: ${HOST_LLVM_DIR}"

echo -e " -> Cross C++: ${GREEN}${CXX_COMPILER}${NC}"
echo -e " -> Cross C  : ${GREEN}${C_COMPILER}${NC}"
echo -e " -> Python   : ${GREEN}${HOST_PY_INC}${NC}"

cat > "${TOOLCHAIN_FILE}" <<EOF
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER "${C_COMPILER}")
set(CMAKE_CXX_COMPILER "${CXX_COMPILER}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
EOF

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [[ "${CMAKE_GENERATOR}" == "Ninja" ]] && ! command -v ninja >/dev/null 2>&1; then
  echo -e "${YELLOW} -> Ninja not found, falling back to Unix Makefiles${NC}"
  CMAKE_GENERATOR="Unix Makefiles"
fi

cmake -G "${CMAKE_GENERATOR}" .. \
  -DCMAKE_TOOLCHAIN_FILE="../${TOOLCHAIN_FILE}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNX_MLIR_ENABLE_PYRUNTIME_LIGHT=ON \
  -DONNX_MLIR_ENABLE_JNI=OFF \
  -DONNX_MLIR_CCACHE_BUILD="${ONNX_MLIR_CCACHE_BUILD}" \
  -Dabsl_DIR="${HOST_ABSL_DIR}" \
  -DMLIR_DIR="${HOST_MLIR_DIR}" \
  -DLLVM_DIR="${HOST_LLVM_DIR}" \
  -DPython3_INCLUDE_DIR="${HOST_PY_INC}" \
  -DPython3_LIBRARY="${HOST_PY_LIB}"

cmake --build . --parallel "${JOBS}" --target cruntime OMTensorUtils OMExecutionSession

LIB_DIR=$(pwd)/Release/lib
echo
echo -e "${GREEN}>>> Build Complete!${NC}"
echo -e ">>> Libraries: ${GREEN}${LIB_DIR}${NC}"
ls -lh "${LIB_DIR}"/libonnx_mlir_cruntime_wrapper.* 2>/dev/null || true
ls -lh "${LIB_DIR}"/libOMTensor.* 2>/dev/null || true
