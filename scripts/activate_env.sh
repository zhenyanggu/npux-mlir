#!/usr/bin/env bash

if [[ -n "${BASH_VERSION:-}" ]]; then
  _npux_script_path="${BASH_SOURCE[0]}"
  if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "source this script instead of executing it: source scripts/activate_env.sh" >&2
    exit 1
  fi
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _npux_script_path="${(%):-%x}"
  if [[ "${ZSH_EVAL_CONTEXT:-}" != *:file ]]; then
    echo "source this script instead of executing it: source scripts/activate_env.sh" >&2
    exit 1
  fi
else
  echo "unsupported shell for activation; use bash or zsh" >&2
  exit 1
fi

SCRIPT_DIR=$(python3 -c 'import os,sys; print(os.path.dirname(os.path.realpath(sys.argv[1])))' "${_npux_script_path}")
REPO_ROOT=$(python3 -c 'import os,sys; print(os.path.dirname(sys.argv[1]))' "${SCRIPT_DIR}")
LOCK_FILE="${REPO_ROOT}/env/toolchain.lock"

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "toolchain lock file not found: ${LOCK_FILE}" >&2
  return 1
fi

# shellcheck disable=SC1090
source "${LOCK_FILE}"

PREFIX="${NPUX_PREFIX:-${NPUX_PREFIX_DEFAULT}}"
CONDA_ENV="${NPUX_CONDA_ENV_NAME:-${NPUX_CONDA_ENV_DEFAULT}}"
HOST_BUILD_DIR="${NPUX_HOST_BUILD_DIR:-${NPUX_HOST_BUILD_DIR_DEFAULT}}"
CROSS_BUILD_DIR="${NPUX_CROSS_BUILD_DIR:-${NPUX_CROSS_BUILD_DIR_DEFAULT}}"
SKIP_CONDA=0
ACTION="activate"

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
    --no-conda)
      SKIP_CONDA=1
      shift
      ;;
    --deactivate)
      ACTION="deactivate"
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      return 1
      ;;
  esac
done

restore_var() {
  local name="$1"
  local backup_var="_NPUX_OLD_${name}"
  local had_var="_NPUX_HAD_${name}"
  local backup_value=""
  local had_value="0"
  if eval '[[ ${'"${backup_var}"'+x} == x ]]'; then
    eval 'backup_value="${'"${backup_var}"'-}"'
    eval 'had_value="${'"${had_var}"':-0}"'
    if [[ "${had_value}" == "1" ]]; then
      export "${name}=${backup_value}"
    else
      unset "${name}"
    fi
    unset "${backup_var}" "${had_var}"
  else
    unset "${name}"
  fi
}

save_var_once() {
  local name="$1"
  local backup_var="_NPUX_OLD_${name}"
  local had_var="_NPUX_HAD_${name}"
  local current_value=""
  if ! eval '[[ ${'"${backup_var}"'+x} == x ]]'; then
    if eval '[[ ${'"${name}"'+x} == x ]]'; then
      eval 'current_value="${'"${name}"'-}"'
      export "${backup_var}=${current_value}"
      export "${had_var}=1"
    else
      export "${backup_var}="
      export "${had_var}=0"
    fi
  fi
}

prepend_path() {
  local name="$1"
  local value="$2"
  local current=""
  if eval '[[ ${'"${name}"'+x} == x ]]'; then
    eval 'current="${'"${name}"'-}"'
  fi
  if [[ -z "${value}" || ! -e "${value}" ]]; then
    return 0
  fi
  case ":${current}:" in
    *":${value}:"*) ;;
    *)
      if [[ -n "${current}" ]]; then
        export "${name}=${value}:${current}"
      else
        export "${name}=${value}"
      fi
      ;;
  esac
}

prepend_python_gpu_runtime_paths() {
  local python_bin=""
  local site_packages=""
  local libdir=""

  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    python_bin="${CONDA_PREFIX}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
  else
    return 0
  fi

  site_packages="$("${python_bin}" -c 'import site; paths = [p for p in site.getsitepackages() if p.endswith("site-packages")]; print(paths[0] if paths else "")' 2>/dev/null)"
  if [[ -z "${site_packages}" || ! -d "${site_packages}" ]]; then
    return 0
  fi

  prepend_path LD_LIBRARY_PATH "${site_packages}/torch/lib"

  for libdir in "${site_packages}"/nvidia/*/lib; do
    if [[ -d "${libdir}" ]]; then
      prepend_path LD_LIBRARY_PATH "${libdir}"
    fi
  done
}

if [[ "${ACTION}" == "deactivate" ]]; then
  restore_var PATH
  restore_var LD_LIBRARY_PATH
  restore_var CMAKE_PREFIX_PATH
  restore_var MLIR_DIR
  restore_var LLVM_DIR
  restore_var absl_DIR
  restore_var LLVM_SRC_ROOT
  restore_var CROSS_CXX
  restore_var CROSS_CC
  restore_var TRANSLATE_TOOL
  restore_var LLC_TOOL
  restore_var RUNTIME_LIB_DIR
  restore_var NPUX_ENV_PREFIX
  restore_var NPUX_HOST_BUILD_DIR
  restore_var NPUX_CROSS_BUILD_DIR
  unset NPUX_ENV_ACTIVE
  return 0
fi

if [[ "${SKIP_CONDA}" -eq 0 && "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found; either install/activate conda first or use --no-conda" >&2
    return 1
  fi
  CONDA_BASE=$(conda info --base)
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

save_var_once PATH
save_var_once LD_LIBRARY_PATH
save_var_once CMAKE_PREFIX_PATH
save_var_once MLIR_DIR
save_var_once LLVM_DIR
save_var_once absl_DIR
save_var_once LLVM_SRC_ROOT
save_var_once CROSS_CXX
save_var_once CROSS_CC
save_var_once TRANSLATE_TOOL
save_var_once LLC_TOOL
save_var_once RUNTIME_LIB_DIR
save_var_once NPUX_ENV_PREFIX
save_var_once NPUX_HOST_BUILD_DIR
save_var_once NPUX_CROSS_BUILD_DIR

LLVM_SRC_ROOT="${PREFIX}/src/llvm-project"
LLVM_BUILD_ROOT="${PREFIX}/build/llvm-project"
HOST_BUILD_ROOT="${REPO_ROOT}/${HOST_BUILD_DIR}"
CROSS_BUILD_ROOT="${REPO_ROOT}/${CROSS_BUILD_DIR}"

export NPUX_ENV_PREFIX="${PREFIX}"
export NPUX_HOST_BUILD_DIR="${HOST_BUILD_DIR}"
export NPUX_CROSS_BUILD_DIR="${CROSS_BUILD_DIR}"
export LLVM_SRC_ROOT
export MLIR_DIR="${LLVM_BUILD_ROOT}/lib/cmake/mlir"
export LLVM_DIR="${LLVM_BUILD_ROOT}/lib/cmake/llvm"
export absl_DIR="${PREFIX}/lib/cmake/absl"
export RUNTIME_LIB_DIR="${CROSS_BUILD_ROOT}/Release/lib"

prepend_path CMAKE_PREFIX_PATH "${PREFIX}"
prepend_path LD_LIBRARY_PATH "/usr/lib/wsl/lib"
prepend_path LD_LIBRARY_PATH "${PREFIX}/lib"
prepend_path LD_LIBRARY_PATH "${RUNTIME_LIB_DIR}"
prepend_path PATH "${HOST_BUILD_ROOT}/Release/bin"
prepend_path PATH "${LLVM_BUILD_ROOT}/bin"
prepend_path PATH "${PREFIX}/bin"
prepend_python_gpu_runtime_paths

for toolchain_dir in \
  "${PREFIX}/toolchains/aarch64/bin" \
  "${PREFIX}/toolchains/aarch64-none-linux-gnu/bin"; do
  if [[ -d "${toolchain_dir}" ]]; then
    prepend_path PATH "${toolchain_dir}"
    if [[ -x "${toolchain_dir}/aarch64-none-linux-gnu-g++" ]]; then
      export CROSS_CXX="${toolchain_dir}/aarch64-none-linux-gnu-g++"
      export CROSS_CC="${toolchain_dir}/aarch64-none-linux-gnu-gcc"
      break
    fi
    if [[ -x "${toolchain_dir}/aarch64-linux-gnu-g++" ]]; then
      export CROSS_CXX="${toolchain_dir}/aarch64-linux-gnu-g++"
      export CROSS_CC="${toolchain_dir}/aarch64-linux-gnu-gcc"
      break
    fi
  fi
done

if [[ -x "${LLVM_BUILD_ROOT}/bin/mlir-translate" ]]; then
  export TRANSLATE_TOOL="${LLVM_BUILD_ROOT}/bin/mlir-translate"
fi
if [[ -x "${LLVM_BUILD_ROOT}/bin/llc" ]]; then
  export LLC_TOOL="${LLVM_BUILD_ROOT}/bin/llc"
fi

export NPUX_ENV_ACTIVE=1

echo "NPUX environment activated"
echo "  repo: ${REPO_ROOT}"
echo "  prefix: ${PREFIX}"
echo "  conda env: ${CONDA_DEFAULT_ENV:-<not activated>}"
echo "  MLIR_DIR: ${MLIR_DIR}"
