#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AICAS_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${AICAS_ROOT}/.." && pwd)
DIST_DIR="${AICAS_ROOT}/dist"
OUT_NAME="aicas_ort_qdq_bundle_$(date +%Y%m%d_%H%M%S).tar"
COMPRESS=0

usage() {
  cat <<EOF
Usage: $0 [--out NAME.tar|NAME.tar.gz] [--compress]

Default output:
  aicas_run/dist/${OUT_NAME}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT_NAME="$2"
      shift 2
      ;;
    --compress)
      COMPRESS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "${DIST_DIR}"
OUT_PATH="${DIST_DIR}/${OUT_NAME}"

TAR_ARGS=()
if [[ "${COMPRESS}" -eq 1 ]]; then
  if [[ "${OUT_PATH}" != *.tar.gz ]]; then
    OUT_PATH="${OUT_PATH}.gz"
  fi
  TAR_ARGS=(-czf)
else
  TAR_ARGS=(-cf)
fi

cd "${REPO_ROOT}"
tar "${TAR_ARGS[@]}" "${OUT_PATH}" \
  aicas_run/FullTest.json \
  aicas_run/config.json \
  aicas_run/generation_config.json \
  aicas_run/preprocessor_config.json \
  aicas_run/processor_config.json \
  aicas_run/tokenizer.json \
  aicas_run/tokenizer_config.json \
  aicas_run/special_tokens_map.json \
  aicas_run/vocab.json \
  aicas_run/merges.txt \
  aicas_run/added_tokens.json \
  aicas_run/chat_template.json \
  aicas_run/num2words.py \
  aicas_run/server_requirements.txt \
  aicas_run/scripts/ort_int8_qdq_pipeline.py \
  aicas_run/scripts/smolvlm2_onnx_lib.py \
  aicas_run/scripts/install_server_env.sh

du -sh "${OUT_PATH}"
echo "Bundle created: ${OUT_PATH}"
