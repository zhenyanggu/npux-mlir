# Qwen2.5-0.5B Stage 7 Decode Cache Smoke Report

Date: 2026-08-05
Environment: `gpu01`, CUDA ORT with `CUDAExecutionProvider,CPUExecutionProvider`

## Fixed ONNX Interface

- Prefill graph: FP16, batch `1`, sequence `128`, returns logits plus 48 KV-cache tensors.
- Decode graph: FP16, batch `1`, step `1`, accepts those 48 cache tensors and returns 48 updated tensors.
- Qwen2.5-0.5B has 24 layers, 2 KV heads, and head dimension 64.
- Every cache tensor uses `[1, 2, sequence, 64]`; prefill emits length 128 and decode emits length 129.
- ONNX input/output names are explicit: `past_key_values.<layer>.{key,value}` and
  `present_key_values.<layer>.{key,value}`.

## CUDA Smoke Result

Prompt: `Qwen2.5 is`

| Check | Result |
| --- | --- |
| Prefill selected provider | `CUDAExecutionProvider` |
| Decode selected provider | `CUDAExecutionProvider` |
| Cache transfer | 48 tensors, `[1,2,128,64]` -> `[1,2,129,64]` |
| NaN / Inf in decode logits | `false` / `false` |
| Prefill elapsed time | `115.14 ms` |
| Decode elapsed time | `59.42 ms` |
| ORT/PyTorch decode top-1 | match (`264`, token `" a"`) |
| ORT/PyTorch decode top-5 overlap | `5/5` |
| Max / mean absolute decode-logit error | `0.0419922` / `0.00677363` |

This is a structural and FP16 numerical smoke result only. It does not
demonstrate NPU decode lowering or validate the stage-6 INT8/QDQ model, whose
prefill numerical quality remains a deployment blocker.

Evidence: `stage7_cache_export_summary.json`, `stage7_decode_smoke.json`.
