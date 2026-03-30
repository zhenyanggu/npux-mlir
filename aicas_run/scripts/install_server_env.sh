#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AICAS_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
ENV_DIR="${AICAS_ROOT}/.venv-server"
PYTHON_BIN="python3"
RUNTIME_MODE="cpu"
ORT_VERSION="1.20.1"

usage() {
  cat <<EOF
Usage: $0 [--env-dir PATH] [--python PYTHON] [--runtime cpu|cu121|cu124] [--ort-version VERSION]

Examples:
  $0
  $0 --runtime cu124
  $0 --env-dir /data/aicas/.venv --python python3.10 --runtime cu124
  $0 --python python3.10 --runtime cpu --ort-version 1.20.1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-dir)
      ENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --runtime)
      RUNTIME_MODE="$2"
      shift 2
      ;;
    --ort-version)
      ORT_VERSION="$2"
      shift 2
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

case "${RUNTIME_MODE}" in
  cpu|cu121|cu124)
    ;;
  *)
    echo "Unsupported runtime: ${RUNTIME_MODE}" >&2
    exit 1
    ;;
esac

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
PY_MM="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"

case "${PY_MM}" in
  3.10|3.11|3.12)
    ;;
  *)
    echo "Unsupported Python version: ${PY_VERSION}" >&2
    echo "This environment requires Python 3.10+." >&2
    echo "Try one of these:" >&2
    echo "  bash scripts/install_server_env.sh --python python3.10 --runtime ${RUNTIME_MODE}" >&2
    echo "  bash scripts/install_server_env.sh --python python3.11 --runtime ${RUNTIME_MODE}" >&2
    exit 1
    ;;
esac

"${PYTHON_BIN}" -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy<2"
python -m pip install -r "${AICAS_ROOT}/server_requirements.txt"

if [[ "${RUNTIME_MODE}" == "cpu" ]]; then
  python -m pip install \
    "onnxruntime==${ORT_VERSION}" \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1
else
  python -m pip install \
    "onnxruntime-gpu==${ORT_VERSION}"
  python -m pip install \
    --index-url "https://download.pytorch.org/whl/${RUNTIME_MODE}" \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1
fi

cat <<EOF

Server environment ready.

Activate:
  source "${ENV_DIR}/bin/activate"

Check ORT providers:
  python -c "import onnxruntime as ort; print(ort.get_available_providers())"
EOF
