#!/usr/bin/env bash
# Start the Qwen prefill static A8W8 optimization on gpu01 and detach safely.
set -euo pipefail

QWEN_ROOT="${QWEN_ROOT:-/home/lqma/qwen25_0p5b}"
CONDA_ROOT="${CONDA_ROOT:-/home/lqma/anaconda3}"
CONDA_ENV="${CONDA_ENV:-npu_test}"
GPU_ID="${GPU_ID:-0}"
HF_MODEL="${HF_MODEL:-/home/lqma/models/Qwen2.5-0.5B}"
INPUT_MODEL="${INPUT_MODEL:-${QWEN_ROOT}/models/qwen25_prefill_128.onnx}"
PROMPT_DIR="${PROMPT_DIR:-${QWEN_ROOT}/data}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:-${PROMPT_DIR}/bootstrap_calibration_prompts.txt}"
VALIDATION_PROMPTS="${VALIDATION_PROMPTS:-${PROMPT_DIR}/bootstrap_validation_prompts.txt}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${QWEN_ROOT}/results/static_int8_sota_overnight_${RUN_STAMP}}"
RUNNER="${QWEN_ROOT}/scripts/run_static_int8_sota_overnight.sh"

if [[ "${DETACHED:-0}" != "1" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  nohup env \
    DETACHED=1 \
    QWEN_ROOT="${QWEN_ROOT}" \
    CONDA_ROOT="${CONDA_ROOT}" \
    CONDA_ENV="${CONDA_ENV}" \
    GPU_ID="${GPU_ID}" \
    HF_MODEL="${HF_MODEL}" \
    INPUT_MODEL="${INPUT_MODEL}" \
    CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS}" \
    VALIDATION_PROMPTS="${VALIDATION_PROMPTS}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    bash "$0" >"${OUTPUT_DIR}/launcher.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "${pid}" >"${OUTPUT_DIR}/job.pid"
  echo "[launch] detached pid=${pid}"
  echo "[launch] output=${OUTPUT_DIR}"
  echo "[launch] monitor: tail -f ${OUTPUT_DIR}/run.log"
  exit 0
fi

[[ -f "${RUNNER}" ]] || { echo "[launch] runner missing: ${RUNNER}" >&2; exit 2; }
[[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]] || { echo "[launch] conda init missing" >&2; exit 2; }
# The npu_test MKL activation hook reads an optional variable. Temporarily
# disable nounset while Conda runs its third-party activation hooks.
set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

mkdir -p "${PROMPT_DIR}"
if [[ ! -e "${CALIBRATION_PROMPTS}" && ! -e "${VALIDATION_PROMPTS}" ]]; then
  python - "${CALIBRATION_PROMPTS}" "${VALIDATION_PROMPTS}" <<'PY'
import sys
from pathlib import Path

calibration_path, validation_path = map(Path, sys.argv[1:])
topics = [
    "matrix multiplication", "transformer attention", "compiler optimization",
    "static INT8 quantization", "GPU memory scheduling", "database indexing",
    "network congestion control", "distributed training", "operating systems",
    "numerical stability", "linear algebra", "probability theory",
    "data visualization", "software testing", "robotics", "computer vision",
    "natural language processing", "wireless communication", "cryptography",
    "renewable energy", "climate science", "medical image analysis",
    "financial risk modeling", "supply chain planning", "digital signal processing",
    "reinforcement learning", "parallel programming", "cache coherence",
    "formal verification", "embedded systems", "API design", "time series analysis",
]
tasks = [
    "Give a concise technical explanation of {topic} with one example.",
    "Compare two practical implementation choices for {topic} and state their tradeoffs.",
    "Write a step-by-step debugging plan for a failure related to {topic}.",
    "Summarize the key equations and assumptions behind {topic}.",
    "Design a small experiment that measures the quality of {topic}.",
    "Explain how {topic} changes when batch size and sequence length increase.",
]
chinese_tasks = [
    "请用中文解释{topic}的核心概念，并给出一个工程例子。",
    "请比较{topic}的两种实现方法，说明精度、性能和资源的取舍。",
    "请为{topic}设计一个可复现的测试方案，并列出验收指标。",
    "请分析{topic}中常见的数值误差来源以及排查步骤。",
]
prompts = []
for topic in topics:
    prompts.extend(task.format(topic=topic) for task in tasks)
    prompts.extend(task.format(topic=topic) for task in chinese_tasks)
if len(prompts) < 320:
    raise RuntimeError("bootstrap prompt corpus is unexpectedly small")
calibration_path.write_text("\n".join(prompts[:192]) + "\n", encoding="utf-8")
validation_path.write_text("\n".join(prompts[192:320]) + "\n", encoding="utf-8")
PY
elif [[ ! -s "${CALIBRATION_PROMPTS}" || ! -s "${VALIDATION_PROMPTS}" ]]; then
  echo "[launch] provide both prompt files, or remove both to generate bootstrap prompts" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export HF_MODEL INPUT_MODEL CALIBRATION_PROMPTS VALIDATION_PROMPTS OUTPUT_DIR
export PROVIDERS="${PROVIDERS:-CUDAExecutionProvider,CPUExecutionProvider}"
export ACTIVATION_SAMPLES="${ACTIVATION_SAMPLES:-1024}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-8}"
exec bash "${RUNNER}"
