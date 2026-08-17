#!/usr/bin/env bash
set -euo pipefail

ssh gpu01 bash -s <<'GPU_SCRIPT'
set -eo pipefail

source /home/lqma/anaconda3/etc/profile.d/conda.sh
conda activate npu_test

python - <<'PY'
import sys
print("python", sys.version.replace("\n", " "))
for mod in ["numpy", "onnx", "google.protobuf", "pytest", "lit"]:
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "OK"))
    except Exception as exc:
        print(mod, "FAIL", repr(exc))
PY

which ninja || true
ninja --version 2>/dev/null || true
which cmake
cmake --version | head -1
which gcc
gcc --version | head -1
GPU_SCRIPT
