#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
LOCK_FILE="${REPO_ROOT}/env/toolchain.lock"

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "toolchain lock file not found: ${LOCK_FILE}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${LOCK_FILE}"

PREFIX="${NPUX_PREFIX:-${NPUX_PREFIX_DEFAULT}}"
CONDA_ENV="${NPUX_CONDA_ENV_NAME:-${NPUX_CONDA_ENV_DEFAULT}}"
HOST_BUILD_DIR="${NPUX_HOST_BUILD_DIR:-${NPUX_HOST_BUILD_DIR_DEFAULT}}"
CROSS_BUILD_DIR="${NPUX_CROSS_BUILD_DIR:-${NPUX_CROSS_BUILD_DIR_DEFAULT}}"
WITH_CROSS=1
SKIP_LLVM_BUILD=0
SKIP_VERIFY=0
HOST_ONLY=0
FORCE_COMPONENT=""
JOBS="${NPUX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

log() {
  echo ">>> $*"
}

warn() {
  echo ">>> [WARN] $*" >&2
}

die() {
  echo ">>> [ERROR] $*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || die "required command not found: ${cmd}"
}

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --prefix <dir>            Install prefix. Default: ${PREFIX}
  --conda-env <name>        Conda env name. Default: ${CONDA_ENV}
  --host-build-dir <dir>    Host build dir relative to repo. Default: ${HOST_BUILD_DIR}
  --cross-build-dir <dir>   Cross build dir relative to repo. Default: ${CROSS_BUILD_DIR}
  --with-cross              Enable AArch64 cross toolchain install/build. Default: on
  --host-only               Skip cross toolchain install and cross build
  --skip-llvm-build         Reuse existing llvm-project checkout/build under prefix
  --skip-verify             Skip smoke verification
  --force-rebuild <name>    Force rebuild of one component: absl|protobuf|llvm|onnx|project|cross
  --jobs <N>                Parallel build jobs. Default: ${JOBS}
  --help                    Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --host-build-dir)
      HOST_BUILD_DIR="$2"
      shift 2
      ;;
    --cross-build-dir)
      CROSS_BUILD_DIR="$2"
      shift 2
      ;;
    --with-cross)
      WITH_CROSS=1
      shift
      ;;
    --host-only)
      WITH_CROSS=0
      HOST_ONLY=1
      shift
      ;;
    --skip-llvm-build)
      SKIP_LLVM_BUILD=1
      shift
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      shift
      ;;
    --force-rebuild)
      FORCE_COMPONENT="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

BUILD_ROOT="${PREFIX}/build"
SRC_ROOT="${PREFIX}/src"
DOWNLOAD_ROOT="${PREFIX}/downloads"
TOOLCHAIN_ROOT="${PREFIX}/toolchains"
LLVM_SRC_ROOT="${SRC_ROOT}/llvm-project"
LLVM_BUILD_ROOT="${BUILD_ROOT}/llvm-project"
ABSL_SRC_ROOT="${SRC_ROOT}/abseil-cpp"
ABSL_BUILD_ROOT="${BUILD_ROOT}/abseil-cpp"
PROTOBUF_SRC_ROOT="${SRC_ROOT}/protobuf"
PROTOBUF_BUILD_ROOT="${BUILD_ROOT}/protobuf"
ARM_TOOLCHAIN_DIR="${TOOLCHAIN_ROOT}/aarch64"
HOST_BUILD_ROOT="${REPO_ROOT}/${HOST_BUILD_DIR}"
CROSS_BUILD_ROOT="${REPO_ROOT}/${CROSS_BUILD_DIR}"

conda_setup() {
  require_cmd conda
  CONDA_BASE=$(conda info --base)
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
}

preflight() {
  require_cmd gcc
  require_cmd g++
  require_cmd cmake
  require_cmd git
  require_cmd curl
  require_cmd tar
  require_cmd xz
  require_cmd python3
  conda_setup

  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    log "Detected host: ${PRETTY_NAME}"
    if [[ "${ID:-}" != "ubuntu" ]]; then
      die "this bootstrap is validated on Ubuntu; detected ${ID:-unknown}"
    fi
    if [[ "${VERSION_ID:-}" != "${NPUX_VALIDATED_HOST_UBUNTU}" ]]; then
      warn "validated target is Ubuntu ${NPUX_VALIDATED_HOST_UBUNTU}, current host is ${VERSION_ID:-unknown}"
    fi
  fi

  if ! grep -qi microsoft /proc/version 2>/dev/null; then
    warn "WSL kernel signature not detected; continuing on generic Linux"
  fi

  if [[ "${REPO_ROOT}" == /mnt/* ]]; then
    warn "repo is under /mnt/*; for better build performance move it to the Linux filesystem"
  fi

  mkdir -p "${PREFIX}" "${BUILD_ROOT}" "${SRC_ROOT}" "${DOWNLOAD_ROOT}" "${TOOLCHAIN_ROOT}"
}

ensure_submodules() {
  log "Updating git submodules"
  git -C "${REPO_ROOT}" submodule update --init --recursive
}

ensure_conda_env() {
  log "Ensuring conda env ${CONDA_ENV}"
  if ! conda run -n "${CONDA_ENV}" python -V >/dev/null 2>&1; then
    conda create -y -n "${CONDA_ENV}" "python=${NPUX_PYTHON_VERSION}" pip setuptools wheel
  fi
  conda install -y -n "${CONDA_ENV}" ninja
}

run_in_conda() {
  conda run -n "${CONDA_ENV}" "$@"
}

cache_entry_equals() {
  local cache_file="$1"
  local key="$2"
  local expected="$3"
  [[ -f "${cache_file}" ]] || return 1
  rg -q "^${key}(:[A-Z]+)?=${expected//\//\\/}$" "${cache_file}"
}

install_python_requirements() {
  log "Installing Python requirements into conda env ${CONDA_ENV}"
  run_in_conda python -m pip install --upgrade "setuptools==68.2.2"
  run_in_conda python -m pip install -r "${REPO_ROOT}/requirements.txt"
}

clone_or_update() {
  local url="$1"
  local ref="$2"
  local dest="$3"
  if [[ ! -d "${dest}/.git" ]]; then
    git clone --recursive "${url}" "${dest}"
  fi
  git -C "${dest}" fetch --tags --force
  git -C "${dest}" checkout "${ref}"
  git -C "${dest}" submodule update --init --recursive
}

ensure_absl() {
  if [[ -f "${PREFIX}/lib/cmake/absl/abslConfig.cmake" && "${FORCE_COMPONENT}" != "absl" ]]; then
    log "absl already present under ${PREFIX}"
    return
  fi
  log "Building absl ${NPUX_ABSL_VERSION}"
  clone_or_update "${NPUX_ABSL_GIT_URL}" "${NPUX_ABSL_VERSION}" "${ABSL_SRC_ROOT}"
  rm -rf "${ABSL_BUILD_ROOT}"
  cmake -S "${ABSL_SRC_ROOT}" -B "${ABSL_BUILD_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DABSL_PROPAGATE_CXX_STD=ON \
    -DBUILD_SHARED_LIBS=ON
  cmake --build "${ABSL_BUILD_ROOT}" --parallel "${JOBS}"
  cmake --install "${ABSL_BUILD_ROOT}"
}

ensure_protobuf() {
  if [[ -x "${PREFIX}/bin/protoc" && -f "${PREFIX}/lib/cmake/protobuf/protobuf-config.cmake" && "${FORCE_COMPONENT}" != "protobuf" ]]; then
    log "protobuf already present under ${PREFIX}"
    return
  fi
  log "Building protobuf ${NPUX_PROTOBUF_VERSION}"
  clone_or_update "${NPUX_PROTOBUF_GIT_URL}" "v${NPUX_PROTOBUF_VERSION}" "${PROTOBUF_SRC_ROOT}"
  rm -rf "${PROTOBUF_BUILD_ROOT}"
  cmake -S "${PROTOBUF_SRC_ROOT}" -B "${PROTOBUF_BUILD_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DBUILD_SHARED_LIBS=ON \
    -Dprotobuf_BUILD_TESTS=OFF
  cmake --build "${PROTOBUF_BUILD_ROOT}" --parallel "${JOBS}"
  cmake --install "${PROTOBUF_BUILD_ROOT}"
}

ensure_llvm() {
  if [[ "${SKIP_LLVM_BUILD}" -eq 1 ]]; then
    log "Skipping llvm-project build by request"
    return
  fi
  if [[ -x "${LLVM_BUILD_ROOT}/bin/llc" && -d "${LLVM_BUILD_ROOT}/lib/cmake/mlir" && "${FORCE_COMPONENT}" != "llvm" ]]; then
    log "llvm-project already present under ${LLVM_BUILD_ROOT}"
    return
  fi
  log "Building llvm-project ${NPUX_LLVM_COMMIT}"
  clone_or_update "${NPUX_LLVM_GIT_URL}" "${NPUX_LLVM_COMMIT}" "${LLVM_SRC_ROOT}"
  rm -rf "${LLVM_BUILD_ROOT}"
  cmake -S "${LLVM_SRC_ROOT}/llvm" -B "${LLVM_BUILD_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_LIBEDIT=OFF
  cmake --build "${LLVM_BUILD_ROOT}" --parallel "${JOBS}"
}

install_local_onnx() {
  if [[ -f "${PREFIX}/.markers/onnx.installed" && "${FORCE_COMPONENT}" != "onnx" ]]; then
    log "patched onnx already installed into conda env"
    return
  fi
  log "Installing patched third_party/onnx into conda env ${CONDA_ENV}"
  local onnx_workdir="${BUILD_ROOT}/onnx-patched"
  rm -rf "${onnx_workdir}"
  mkdir -p "${onnx_workdir}"
  cp -a "${REPO_ROOT}/third_party/onnx/." "${onnx_workdir}/"
  sed -i \
    -e 's/target_link_libraries(onnx PUBLIC onnx_proto)/target_link_libraries(onnx PUBLIC onnx_proto PRIVATE ${protobuf_ABSL_USED_TARGETS})/g' \
    -e '/absl::log_initialize/a \
          absl::log_internal_check_op\
          absl::log_internal_message\
          absl::log_internal_nullguard' \
    "${onnx_workdir}/CMakeLists.txt"

  mkdir -p "${PREFIX}/.markers"
  local env_python
  local env_prefix
  env_python=$(run_in_conda python -c 'import sys; print(sys.executable)')
  env_prefix=$(dirname "$(dirname "${env_python}")")
  run_in_conda env \
    CMAKE_PREFIX_PATH="${PREFIX}" \
    LD_LIBRARY_PATH="${PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${PREFIX} -DCMAKE_INSTALL_LIBDIR=lib -DPython_EXECUTABLE=${env_python} -DPython3_EXECUTABLE=${env_python} -DPython_ROOT_DIR=${env_prefix} -DPython3_ROOT_DIR=${env_prefix}" \
    CC=gcc \
    CXX=g++ \
    python -m pip install --no-build-isolation "${onnx_workdir}"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${PREFIX}/.markers/onnx.installed"
}

ensure_cross_toolchain() {
  if [[ "${WITH_CROSS}" -eq 0 ]]; then
    return
  fi
  if [[ -x "${ARM_TOOLCHAIN_DIR}/bin/aarch64-none-linux-gnu-g++" && "${FORCE_COMPONENT}" != "cross" ]]; then
    log "AArch64 GNU toolchain already present under ${ARM_TOOLCHAIN_DIR}"
    return
  fi
  log "Installing AArch64 GNU toolchain ${NPUX_ARM_GNU_TOOLCHAIN_VERSION}"
  local archive="${DOWNLOAD_ROOT}/$(basename "${NPUX_ARM_GNU_TOOLCHAIN_URL}")"
  rm -rf "${ARM_TOOLCHAIN_DIR}"
  mkdir -p "${ARM_TOOLCHAIN_DIR}"
  curl -L "${NPUX_ARM_GNU_TOOLCHAIN_URL}" -o "${archive}"
  tar -xf "${archive}" -C "${ARM_TOOLCHAIN_DIR}" --strip-components=1
}

install_conda_hooks() {
  log "Installing conda activation hooks for ${CONDA_ENV}"
  local env_prefix
  env_prefix=$(conda env list | awk -v name="${CONDA_ENV}" '$1 == name {print $NF}')
  [[ -n "${env_prefix}" ]] || die "unable to locate conda env prefix for ${CONDA_ENV}"
  mkdir -p "${env_prefix}/etc/conda/activate.d" "${env_prefix}/etc/conda/deactivate.d"
  cat > "${env_prefix}/etc/conda/activate.d/npux-mlir.sh" <<EOF
source "${REPO_ROOT}/scripts/activate_env.sh" --prefix "${PREFIX}" --conda-env "${CONDA_ENV}" --host-build-dir "${HOST_BUILD_DIR}" --cross-build-dir "${CROSS_BUILD_DIR}" --no-conda
EOF
  cat > "${env_prefix}/etc/conda/deactivate.d/npux-mlir.sh" <<EOF
source "${REPO_ROOT}/scripts/activate_env.sh" --deactivate
EOF
}

activate_for_build() {
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/scripts/activate_env.sh" --prefix "${PREFIX}" --conda-env "${CONDA_ENV}" --host-build-dir "${HOST_BUILD_DIR}" --cross-build-dir "${CROSS_BUILD_DIR}"
}

build_project_host() {
  local cache_file="${HOST_BUILD_ROOT}/CMakeCache.txt"
  local host_needs_clean=0

  if [[ "${FORCE_COMPONENT}" == "project" ]]; then
    host_needs_clean=1
  elif [[ -f "${cache_file}" ]]; then
    if ! cache_entry_equals "${cache_file}" "MLIR_DIR" "${LLVM_BUILD_ROOT}/lib/cmake/mlir"; then
      warn "Host build cache MLIR_DIR does not match current prefix; rebuilding host build"
      host_needs_clean=1
    elif ! cache_entry_equals "${cache_file}" "LLVM_DIR" "${LLVM_BUILD_ROOT}/lib/cmake/llvm"; then
      warn "Host build cache LLVM_DIR does not match current prefix; rebuilding host build"
      host_needs_clean=1
    elif ! cache_entry_equals "${cache_file}" "absl_DIR" "${PREFIX}/lib/cmake/absl"; then
      warn "Host build cache absl_DIR does not match current prefix; rebuilding host build"
      host_needs_clean=1
    elif ! cache_entry_equals "${cache_file}" "CMAKE_HOME_DIRECTORY" "${REPO_ROOT}"; then
      warn "Host build cache source directory does not match current repo; rebuilding host build"
      host_needs_clean=1
    fi
  fi

  if [[ "${host_needs_clean}" -eq 1 ]]; then
    rm -rf "${HOST_BUILD_ROOT}"
  fi

  log "Configuring host build under ${HOST_BUILD_ROOT}"
  cmake -S "${REPO_ROOT}" -B "${HOST_BUILD_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DMLIR_DIR="${LLVM_BUILD_ROOT}/lib/cmake/mlir" \
    -DLLVM_DIR="${LLVM_BUILD_ROOT}/lib/cmake/llvm" \
    -Dabsl_DIR="${PREFIX}/lib/cmake/absl" \
    -DCMAKE_BUILD_RPATH="${PREFIX}/lib;${HOST_BUILD_ROOT}/Release/lib" \
    -DCMAKE_INSTALL_RPATH="${PREFIX}/lib;${HOST_BUILD_ROOT}/Release/lib" \
    -DONNX_MLIR_ENABLE_JAVA=OFF \
    -DONNX_MLIR_CCACHE_BUILD=OFF
  cmake --build "${HOST_BUILD_ROOT}" --parallel "${JOBS}" --target onnx-mlir onnx-mlir-opt
}

build_project_cross() {
  if [[ "${WITH_CROSS}" -eq 0 || "${HOST_ONLY}" -eq 1 ]]; then
    return
  fi
  log "Configuring cross build under ${CROSS_BUILD_ROOT}"
  "${REPO_ROOT}/cmake_arm64.sh"
}

verify_environment() {
  if [[ "${SKIP_VERIFY}" -eq 1 ]]; then
    return
  fi
  log "Running smoke verification"
  command -v onnx-mlir >/dev/null 2>&1 || die "onnx-mlir not found after activation"
  command -v onnx-mlir-opt >/dev/null 2>&1 || die "onnx-mlir-opt not found after activation"
  command -v mlir-translate >/dev/null 2>&1 || die "mlir-translate not found after activation"
  command -v llc >/dev/null 2>&1 || die "llc not found after activation"
  python -c "import onnx, google.protobuf, numpy"
  if [[ "${WITH_CROSS}" -eq 1 ]]; then
    command -v "${CROSS_CXX##*/}" >/dev/null 2>&1 || die "cross compiler not on PATH after activation"
  fi
}

main() {
  preflight
  ensure_submodules
  ensure_conda_env
  install_python_requirements
  ensure_absl
  ensure_protobuf
  ensure_llvm
  install_local_onnx
  ensure_cross_toolchain
  install_conda_hooks
  activate_for_build
  build_project_host
  build_project_cross
  verify_environment
  log "Bootstrap complete"
  log "Next step: source ${REPO_ROOT}/scripts/activate_env.sh --prefix ${PREFIX} --conda-env ${CONDA_ENV}"
}

main "$@"
