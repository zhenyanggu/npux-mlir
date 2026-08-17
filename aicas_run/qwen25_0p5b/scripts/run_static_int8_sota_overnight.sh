#!/usr/bin/env bash
# Run this from the Qwen server workspace after activating its Python env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_MODEL="${HF_MODEL:?Set HF_MODEL to the local Hugging Face Qwen2.5-0.5B directory}"
INPUT_MODEL="${INPUT_MODEL:-${QWEN_ROOT}/models/qwen25_prefill_128.onnx}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:?Set CALIBRATION_PROMPTS to a UTF-8 .txt or JSON prompt list}"
VALIDATION_PROMPTS="${VALIDATION_PROMPTS:?Set VALIDATION_PROMPTS to an independent UTF-8 .txt or JSON prompt list}"
OUTPUT_DIR="${OUTPUT_DIR:-${QWEN_ROOT}/results/static_int8_sota_$(date +%Y%m%d_%H%M%S)}"
PROVIDERS="${PROVIDERS:-CUDAExecutionProvider,CPUExecutionProvider}"
SEQ_LEN="${SEQ_LEN:-128}"

[[ -d "${HF_MODEL}" ]] || { echo "[run] HF_MODEL is not a directory: ${HF_MODEL}" >&2; exit 2; }
[[ -f "${INPUT_MODEL}" ]] || { echo "[run] INPUT_MODEL is not a file: ${INPUT_MODEL}" >&2; exit 2; }
[[ -s "${CALIBRATION_PROMPTS}" ]] || { echo "[run] CALIBRATION_PROMPTS is empty or missing: ${CALIBRATION_PROMPTS}" >&2; exit 2; }
[[ -s "${VALIDATION_PROMPTS}" ]] || { echo "[run] VALIDATION_PROMPTS is empty or missing: ${VALIDATION_PROMPTS}" >&2; exit 2; }
[[ "${CALIBRATION_PROMPTS}" != "${VALIDATION_PROMPTS}" ]] || { echo "[run] calibration and validation prompt files must be separate" >&2; exit 2; }

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/run.log") 2>&1
trap 'status=$?; echo "[run] failed=$(date -Is) status=${status}"; exit "${status}"' ERR

echo "[run] started=$(date -Is)"
echo "[run] input=${INPUT_MODEL}"
echo "[run] output=${OUTPUT_DIR}"

"${PYTHON_BIN}" -c 'import numpy, onnx, onnxruntime, torch, transformers; print("[env] numpy=" + numpy.__version__ + " onnx=" + onnx.__version__ + " ort=" + onnxruntime.__version__ + " torch=" + torch.__version__ + " transformers=" + transformers.__version__ + " cuda=" + str(torch.cuda.is_available()))'
"${PYTHON_BIN}" -c 'import sys, onnxruntime as ort; requested = [item.strip() for item in sys.argv[1].split(",") if item.strip()]; available = ort.get_available_providers(); missing = sorted(set(requested) - set(available)); print("[env] providers requested=" + str(requested) + " available=" + str(available)); sys.exit(2 if missing else 0)' "${PROVIDERS}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/optimize_static_int8_prefill.py" \
  --hf-model "${HF_MODEL}" \
  --input-model "${INPUT_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --calibration-prompts "${CALIBRATION_PROMPTS}" \
  --validation-prompts "${VALIDATION_PROMPTS}" \
  --seq-len "${SEQ_LEN}" \
  --providers "${PROVIDERS}" \
  --activation-samples "${ACTIVATION_SAMPLES:-1024}" \
  --samples-per-prompt "${SAMPLES_PER_PROMPT:-8}" \
  --activation-calibration "${ACTIVATION_CALIBRATION:-Percentile}" \
  --activation-percentile "${ACTIVATION_PERCENTILE:-99.95}" \
  --smoothquant-alphas "${SMOOTHQUANT_ALPHAS:-0.3,0.5,0.7}" \
  --awq-alphas "${AWQ_ALPHAS:-0.0,0.2,0.4,0.6,0.8,1.0}" \
  --gptq-damps "${GPTQ_DAMPS:-0.01,0.03,0.1}" \
  --gptq-device "${GPTQ_DEVICE:-auto}" \
  --gptq-block-size "${GPTQ_BLOCK_SIZE:-128}" \
  --gptq-output-block "${GPTQ_OUTPUT_BLOCK:-4096}" \
  --global-kl "${GLOBAL_KL:-0.01}" \
  --global-top1 "${GLOBAL_TOP1:-0.95}" \
  --global-greedy \
  --global-greedy-max-nrmse "${GLOBAL_GREEDY_MAX_NRMSE:-0.05}" \
  --global-greedy-max-groups "${GLOBAL_GREEDY_MAX_GROUPS:-24}" \
  --global-search-prompts "${GLOBAL_SEARCH_PROMPTS:-64}"

echo "[run] finished=$(date -Is)"
