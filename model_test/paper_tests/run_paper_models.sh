#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_TEST_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
MODEL_LIST="${SCRIPT_DIR}/paper_models.txt"

usage() {
  cat <<'USAGE'
Usage: run_paper_models.sh [build|cpu|run|clean|list] [extra make args...]

Commands:
  build  Build NPU ZCU102 executables for the paper models.
  cpu    Build CPU baseline executables for the paper models.
  run    Run generated NPU executables from model_test/output/NPU.
  clean  Remove build/output directories for the paper models.
  list   Print the paper model names.

Default command: build

Examples:
  ./run_paper_models.sh build EXPERIMENT_NAME=baseline
  ./run_paper_models.sh build EXPERIMENT_NAME=no_dma_elim NPU_REMOVE_REDUNDANT_DMA=0
  ./run_paper_models.sh run EXPERIMENT_NAME=baseline
USAGE
}

cmd="${1:-build}"
if [[ $# -gt 0 ]]; then
  shift
fi

mapfile -t models < "${MODEL_LIST}"

experiment_name=""
output_root="${MODEL_TEST_DIR}/output"
for arg in "$@"; do
  case "${arg}" in
    EXPERIMENT_NAME=*)
      experiment_name="${arg#EXPERIMENT_NAME=}"
      ;;
    OUTPUT_ROOT=*)
      output_root="${arg#OUTPUT_ROOT=}"
      ;;
  esac
done
if [[ -n "${experiment_name}" && "${output_root}" = "${MODEL_TEST_DIR}/output" ]]; then
  output_root="${MODEL_TEST_DIR}/output/${experiment_name}"
fi
output_npu_dir="${output_root}/NPU"

case "${cmd}" in
  build)
    for model in "${models[@]}"; do
      [[ -n "${model}" ]] || continue
      make -C "${MODEL_TEST_DIR}" "${model}" "$@"
    done
    ;;
  cpu)
    for model in "${models[@]}"; do
      [[ -n "${model}" ]] || continue
      make -C "${MODEL_TEST_DIR}" "${model}.cpu" "$@"
    done
    ;;
  run)
    pass_count=0
    fail_count=0
    for model in "${models[@]}"; do
      [[ -n "${model}" ]] || continue
      exe="${output_npu_dir}/${model}/${model}_zcu102"
      log="${exe}.log"
      if [[ ! -f "${exe}" ]]; then
        echo "[FAIL] ${model}: executable not found: ${exe}"
        fail_count=$((fail_count + 1))
        continue
      fi
      chmod +x "${exe}"
      echo "[run] ${model}"
      if "${exe}" > "${log}" 2>&1; then
        result_line=$(grep -E "@@MODEL_TEST_RESULT@@" "${log}" | tail -n 1 || true)
        if [[ "${result_line}" == *"status=PASS"* ]]; then
          echo "[PASS] ${model}: ${result_line}"
          pass_count=$((pass_count + 1))
        else
          echo "[FAIL] ${model}: missing PASS result marker"
          tail -n 10 "${log}"
          fail_count=$((fail_count + 1))
        fi
      else
        echo "[FAIL] ${model}: executable returned non-zero"
        tail -n 10 "${log}"
        fail_count=$((fail_count + 1))
      fi
    done
    echo "summary: pass=${pass_count} fail=${fail_count}"
    [[ "${fail_count}" -eq 0 ]]
    ;;
  clean)
    for model in "${models[@]}"; do
      [[ -n "${model}" ]] || continue
      make -C "${MODEL_TEST_DIR}" clean "${model}" "$@"
    done
    ;;
  list)
    printf '%s\n' "${models[@]}"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
