#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_TEST_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CACHE_DIR="${1:-${MODEL_TEST_DIR}/cache}"

DATASET_NAME="imagenette2-320"
DATASET_URL="https://s3.amazonaws.com/fast-ai-imageclas/${DATASET_NAME}.tgz"
ARCHIVE_PATH="${CACHE_DIR}/${DATASET_NAME}.tgz"
DATASET_ROOT="${CACHE_DIR}/${DATASET_NAME}"
VAL_ROOT="${DATASET_ROOT}/val"

mkdir -p "${CACHE_DIR}"

if [[ -d "${VAL_ROOT}" ]]; then
  echo "[skip] dataset already exists: ${VAL_ROOT}"
  echo "IMAGENET_ROOT=${VAL_ROOT}"
  exit 0
fi

download_with_curl() {
  curl -L --fail --retry 3 --retry-delay 2 -o "${ARCHIVE_PATH}" "${DATASET_URL}"
}

download_with_wget() {
  wget -O "${ARCHIVE_PATH}" "${DATASET_URL}"
}

echo "[download] ${DATASET_URL}"
if command -v curl >/dev/null 2>&1; then
  download_with_curl
elif command -v wget >/dev/null 2>&1; then
  download_with_wget
else
  echo "[error] need curl or wget to download dataset archive"
  exit 2
fi

echo "[extract] ${ARCHIVE_PATH} -> ${CACHE_DIR}"
tar -xzf "${ARCHIVE_PATH}" -C "${CACHE_DIR}"

if [[ ! -d "${VAL_ROOT}" ]]; then
  echo "[error] extracted dataset but missing expected val directory: ${VAL_ROOT}"
  exit 2
fi

echo "[done] dataset ready at: ${VAL_ROOT}"
echo "IMAGENET_ROOT=${VAL_ROOT}"

