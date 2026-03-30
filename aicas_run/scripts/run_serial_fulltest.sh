#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AICAS_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
PROVIDERS="${PROVIDERS:-CPUExecutionProvider}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
SEED="${SEED:-20260329}"
CALIB_PROGRESS_EVERY="${CALIB_PROGRESS_EVERY:-10}"
RESUME_FLAG="${RESUME_FLAG:---resume}"
CONFIGS_TEXT="${CONFIGS_TEXT:-cfg1,cfg2,cfg3,cfg4}"

run_py() {
  echo
  echo "[run] $*"
  "${PYTHON_BIN}" "$@"
}

cd "${AICAS_ROOT}"
mkdir -p results/int8_qdq_models results/int8_qdq_eval

IFS=',' read -r -a CONFIGS <<< "${CONFIGS_TEXT}"

run_py scripts/ort_int8_qdq_pipeline.py prepare \
  --input-json FullTest.json \
  --dev-json-out results/dev_260.json \
  --calib-json-out results/calib_520.json \
  --split-summary-json results/int8_qdq_split_summary.json \
  --seed "${SEED}" \
  ${RESUME_FLAG}

QUANT_SUMMARY_FILES=()
DEV_INT8_JSONS=()

for cfg in "${CONFIGS[@]}"; do
  run_py scripts/ort_int8_qdq_pipeline.py quantize \
    --calib-json results/calib_520.json \
    --image-root data \
    --model-dir models \
    --asset-dir . \
    --quant-out-dir results/int8_qdq_models \
    --quant-summary-json "results/int8_qdq_quant_summary_${cfg}_vision.json" \
    --providers "${PROVIDERS}" \
    --configs "${cfg}" \
    --components vision \
    --calib-progress-every "${CALIB_PROGRESS_EVERY}" \
    ${RESUME_FLAG}

  run_py scripts/ort_int8_qdq_pipeline.py quantize \
    --calib-json results/calib_520.json \
    --image-root data \
    --model-dir models \
    --asset-dir . \
    --quant-out-dir results/int8_qdq_models \
    --quant-summary-json "results/int8_qdq_quant_summary_${cfg}_decoder.json" \
    --providers "${PROVIDERS}" \
    --configs "${cfg}" \
    --components decoder \
    --calib-progress-every "${CALIB_PROGRESS_EVERY}" \
    ${RESUME_FLAG}

  QUANT_SUMMARY_FILES+=("results/int8_qdq_quant_summary_${cfg}_vision.json")
  QUANT_SUMMARY_FILES+=("results/int8_qdq_quant_summary_${cfg}_decoder.json")
done

run_py scripts/ort_int8_qdq_pipeline.py eval \
  --model-mode fp16 \
  --input-json results/dev_260.json \
  --image-root data \
  --model-dir models \
  --asset-dir . \
  --eval-out-dir results/int8_qdq_eval \
  --dataset-tag dev_260 \
  --providers "${PROVIDERS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  ${RESUME_FLAG}

for cfg in "${CONFIGS[@]}"; do
  run_py scripts/ort_int8_qdq_pipeline.py eval \
    --model-mode int8 \
    --config-id "${cfg}" \
    --input-json results/dev_260.json \
    --image-root data \
    --model-dir models \
    --asset-dir . \
    --quant-out-dir results/int8_qdq_models \
    --eval-out-dir results/int8_qdq_eval \
    --dataset-tag dev_260 \
    --providers "${PROVIDERS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    ${RESUME_FLAG}

  DEV_INT8_JSONS+=("results/int8_qdq_eval/dev_260_int8_${cfg}.json")
done

BEST_CONFIG=$(
  "${PYTHON_BIN}" - <<'PY'
import json
import os

base = "results/int8_qdq_eval"
configs = ["cfg1", "cfg2", "cfg3", "cfg4"]
best = None
best_key = None
for cfg in configs:
    path = os.path.join(base, f"dev_260_int8_{cfg}.json")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload["summary"]
    key = (
        float(summary["official"]["accuracy"]),
        float(summary["overall"]["accuracy"]),
    )
    if best_key is None or key > best_key:
        best_key = key
        best = cfg
print(best)
PY
)

echo
echo "[result] best config on dev_260: ${BEST_CONFIG}"

run_py scripts/ort_int8_qdq_pipeline.py eval \
  --model-mode fp16 \
  --input-json FullTest.json \
  --image-root data \
  --model-dir models \
  --asset-dir . \
  --eval-out-dir results/int8_qdq_eval \
  --dataset-tag fulltest \
  --providers "${PROVIDERS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  ${RESUME_FLAG}

run_py scripts/ort_int8_qdq_pipeline.py eval \
  --model-mode int8 \
  --config-id "${BEST_CONFIG}" \
  --input-json FullTest.json \
  --image-root data \
  --model-dir models \
  --asset-dir . \
  --quant-out-dir results/int8_qdq_models \
  --eval-out-dir results/int8_qdq_eval \
  --dataset-tag "fulltest_${BEST_CONFIG}" \
  --providers "${PROVIDERS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  ${RESUME_FLAG}

"${PYTHON_BIN}" - <<'PY'
import json
import os

configs = ["cfg1", "cfg2", "cfg3", "cfg4"]
merged = {
    "generated_at": "",
    "source_coverage": {},
    "quant_configs": [],
}

for cfg in configs:
    for side in ("vision", "decoder"):
        path = f"results/int8_qdq_quant_summary_{cfg}_{side}.json"
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not merged["generated_at"]:
            merged["generated_at"] = payload.get("generated_at", "")
        if not merged["source_coverage"]:
            merged["source_coverage"] = payload.get("source_coverage", {})
        for item in payload.get("quant_configs", []):
            cfg_id = item.get("config", {}).get("config_id")
            existed = None
            for old in merged["quant_configs"]:
                if old.get("config", {}).get("config_id") == cfg_id:
                    existed = old
                    break
            if existed is None:
                merged["quant_configs"].append(item)
            else:
                existed["coverage"]["vision"] = item["coverage"].get("vision", existed["coverage"].get("vision", {}))
                existed["coverage"]["decoder"] = item["coverage"].get("decoder", existed["coverage"].get("decoder", {}))
                existed["coverage"]["combined"] = item["coverage"].get("combined", existed["coverage"].get("combined", {}))

with open("results/int8_qdq_quant_summary_all.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
PY

DEV_INT8_JOINED=$(IFS=,; echo "${DEV_INT8_JSONS[*]}")

run_py scripts/ort_int8_qdq_pipeline.py report \
  --input-json FullTest.json \
  --image-root data \
  --quant-out-dir results/int8_qdq_models \
  --eval-out-dir results/int8_qdq_eval \
  --quant-summary-json results/int8_qdq_quant_summary_all.json \
  --report-path results/int8_qdq_fulltest_report.md \
  --dev-fp16-json results/int8_qdq_eval/dev_260_fp16.json \
  --dev-int8-jsons "${DEV_INT8_JOINED}" \
  --full-fp16-json results/int8_qdq_eval/fulltest_fp16.json \
  --full-int8-json "results/int8_qdq_eval/fulltest_${BEST_CONFIG}_int8_${BEST_CONFIG}.json" \
  --selected-config-id "${BEST_CONFIG}"

echo
echo "[done] report: results/int8_qdq_fulltest_report.md"
