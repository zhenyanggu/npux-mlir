# Qwen2.5-0.5B Prefill Bring-Up

This directory tracks the Qwen2.5-0.5B text-only prefill bring-up.

The first milestone is a fixed-shape ONNX Runtime baseline:

- batch: 1
- sequence length: 128
- output: last-token logits summary

## Layout

- `models/`: local model assets and exported ONNX files.
- `scripts/`: export, inspection, and baseline scripts.
- `results/`: JSON summaries and logs.
- `tmp/`: compiler/runtime scratch outputs.

## Expected Inputs

Place or download the HuggingFace model under:

```text
aicas_run/qwen25_0p5b/models/hf/Qwen2.5-0.5B
```

The scripts also accept an explicit model path with `--hf-model`.

## Stage 1 Commands

Export fixed-shape prefill ONNX:

```bash
python aicas_run/qwen25_0p5b/scripts/export_prefill_onnx.py \
  --hf-model aicas_run/qwen25_0p5b/models/hf/Qwen2.5-0.5B \
  --output aicas_run/qwen25_0p5b/models/qwen25_prefill_128.onnx \
  --seq-len 128
```

Inspect ONNX inputs and outputs:

```bash
python aicas_run/qwen25_0p5b/scripts/inspect_onnx_io.py \
  --model aicas_run/qwen25_0p5b/models/qwen25_prefill_128.onnx \
  --output-json aicas_run/qwen25_0p5b/results/qwen25_prefill_128_io.json
```

Run ONNX Runtime baseline:

```bash
python aicas_run/qwen25_0p5b/scripts/run_prefill_ort.py \
  --hf-model aicas_run/qwen25_0p5b/models/hf/Qwen2.5-0.5B \
  --onnx-model aicas_run/qwen25_0p5b/models/qwen25_prefill_128.onnx \
  --output-json aicas_run/qwen25_0p5b/results/qwen25_prefill_128_ort_result.json \
  --seq-len 128 \
  --prompt "Hello"
```

## Current Local Status

At creation time, this checkout did not contain Qwen2.5 model assets and the active Windows shell resolved `python` to the Microsoft Store stub. Use the project `npux-mlir` environment before running the commands above.
