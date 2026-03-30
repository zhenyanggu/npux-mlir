# Server Deployment Guide

## 1. Create Bundle Locally

From repo root:

```bash
bash aicas_run/scripts/package_server_bundle.sh --compress
```

This creates a bundle under `aicas_run/dist/`.

Included files:

- `aicas_run/data`
- `aicas_run/models`
- `aicas_run/FullTest.json`
- tokenizer / processor / config files
- `aicas_run/scripts/ort_int8_qdq_pipeline.py`
- `aicas_run/scripts/smolvlm2_onnx_lib.py`
- `aicas_run/scripts/install_server_env.sh`
- `aicas_run/server_requirements.txt`

Current bundle size reference in this workspace:

- `aicas_run/data`: about `5.4G`
- `aicas_run/models`: about `970M`
- `aicas_run/FullTest.json`: about `15M`

## 2. Upload to Server

Example:

```bash
scp aicas_run/dist/aicas_ort_qdq_bundle_*.tar.gz user@server:/data/work/
```

## 3. Extract on Server

```bash
cd /data/work
tar -xzf aicas_ort_qdq_bundle_*.tar.gz
cd aicas_run
```

If you created an uncompressed `.tar`, use:

```bash
tar -xf aicas_ort_qdq_bundle_*.tar
```

## 4. Install Python Environment

Requirement: Python `3.10+`
Requirement: `numpy<2` for the current ORT wheel compatibility on many servers

Check version:

```bash
python3 --version
python3.10 --version
python3.11 --version
```

CPU runtime:

```bash
bash scripts/install_server_env.sh --python python3.10 --runtime cpu
source .venv-server/bin/activate
```

CUDA 12.4 runtime:

```bash
bash scripts/install_server_env.sh --python python3.10 --runtime cu124
source .venv-server/bin/activate
```

Check ORT providers:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## 5. Recommended First Run

Do not start with full `run-all`. First verify a small vision-only quantization.

```bash
python scripts/ort_int8_qdq_pipeline.py quantize \
  --calib-json results/calib_520.json \
  --image-root data \
  --model-dir models \
  --asset-dir . \
  --quant-out-dir results/int8_qdq_models \
  --quant-summary-json results/int8_qdq_quant_summary_vision_cfg1_26.json \
  --providers CUDAExecutionProvider,CPUExecutionProvider \
  --configs cfg1 \
  --components vision \
  --calib-per-type-override 2 \
  --calib-progress-every 5 \
  --resume
```

If that succeeds, scale up gradually.

## 6. Prepare Splits on Server

```bash
python scripts/ort_int8_qdq_pipeline.py prepare \
  --input-json FullTest.json \
  --dev-json-out results/dev_260.json \
  --calib-json-out results/calib_520.json \
  --split-summary-json results/int8_qdq_split_summary.json \
  --seed 20260329 \
  --resume
```

## 7. Full Run After Validation

```bash
python scripts/ort_int8_qdq_pipeline.py run-all \
  --input-json FullTest.json \
  --image-root data \
  --model-dir models \
  --asset-dir . \
  --providers CUDAExecutionProvider,CPUExecutionProvider \
  --max-new-tokens 100 \
  --seed 20260329 \
  --resume
```

On medium-memory machines, prefer separate `quantize` / `eval` steps instead of running everything at once.
