# Qwen2.5-0.5B Prefill Deployment Progress

## Goal

Deploy Qwen2.5-0.5B step by step, with prefill as the first milestone. The first target is not full chat generation, but a reliable fixed-shape prefill path that can be profiled, compiled, and gradually lowered to the NPU backend.

## Current Assumptions

- Target model: Qwen2.5-0.5B.
- First scope: text-only prefill, not multimodal and not full decode.
- First shape target: batch = 1, sequence length = 128.
- Preferred first baseline: ONNX Runtime.
- Compiler path should start from CPU/LLVM before attempting NPU lowering.
- NPU work should begin from smaller subgraphs before full prefill.

## Milestone 1: Model And ONNX Baseline

- [x] Create workspace layout under `aicas_run/qwen25_0p5b/`.
- [x] Download or place Qwen2.5-0.5B model and tokenizer assets.
- [x] Export a prefill-only ONNX model.
- [x] Fix the first shape target to batch = 1, sequence length = 128.
- [x] Record ONNX input names, output names, dtypes, and shapes.
- [x] Run the exported ONNX with ONNX Runtime.
- [x] Verify logits shape and check for NaN/Inf.
- [x] Save the first ORT baseline output summary.

Current status:

- Added Qwen2.5-0.5B prefill workspace and stage-1 helper scripts.
- Qwen2.5-0.5B assets are available on the 143/gpu01 shared home at `/home/lqma/models/Qwen2.5-0.5B`.
- Server environment: `/home/lqma/anaconda3/envs/npu_test`.
- Installed server packages needed for stage 1 include `optimum 2.2.0` and `onnxruntime-gpu 1.19.2`.
- Exported fixed-shape prefill ONNX on `gpu01`: `/home/lqma/qwen25_0p5b/models/qwen25_prefill_128.onnx` (`1.2G`).
- Saved local summaries:
  - `aicas_run/qwen25_0p5b/models/qwen25_prefill_128.export.json`
  - `aicas_run/qwen25_0p5b/results/qwen25_prefill_128_io.json`
  - `aicas_run/qwen25_0p5b/results/qwen25_prefill_128_ort_result.json`
- ORT baseline used `CUDAExecutionProvider,CPUExecutionProvider`, produced logits `[1, 128, 151936]`, and reported no NaN/Inf.

Expected artifacts:

- `aicas_run/qwen25_0p5b/models/qwen25_prefill_128.onnx`
- `aicas_run/qwen25_0p5b/results/qwen25_prefill_128_ort_result.json`

## Milestone 2: Qwen Runner

- [x] Add a Qwen-specific ONNX runner instead of modifying `SmolVLM2OnnxRunner`.
- [x] Load tokenizer and model config from Qwen assets.
- [x] Build `input_ids`, `attention_mask`, and `position_ids`.
- [x] Run prefill and collect the last-token logits.
- [x] Add simple prompts for repeatable smoke testing.
- [x] Confirm repeated runs are deterministic for the same input.

Suggested file:

- `aicas_run/qwen25_0p5b/scripts/qwen25_onnx_lib.py`

Current status:

- Added reusable runner: `aicas_run/qwen25_0p5b/scripts/qwen25_onnx_lib.py`.
- Added smoke script: `aicas_run/qwen25_0p5b/scripts/run_qwen25_prefill_smoke.py`.
- Added PyTorch-vs-ONNX comparison: `aicas_run/qwen25_0p5b/scripts/run_torch_vs_onnx_prefill.py`.
- Smoke result on `gpu01`: 4 prompts, average ONNX prefill latency `38.44 ms`, no NaN/Inf.
- PyTorch-vs-ONNX result on `gpu01`: top1 matched `4/4`, average top5 overlap `5.0/5`, max abs diff `0.119140625`, average mean abs diff `0.01226972439326346`.
- Saved local summaries:
  - `aicas_run/qwen25_0p5b/results/qwen25_prefill_128_smoke.json`
  - `aicas_run/qwen25_0p5b/results/qwen25_prefill_128_torch_vs_onnx.json`

## Milestone 3: Operator And Shape Analysis

- [x] Count ONNX operators.
- [x] Count `MatMul` and `Gemm`.
- [x] Extract MatMul shapes as M/K/N where possible.
- [x] Check whether RMSNorm and RoPE are standard ONNX subgraphs or custom ops.
- [x] Check for `com.microsoft` custom ops.
- [x] Identify unsupported or risky ops for onnx-mlir.
- [x] Save operator and MatMul reports.

Expected artifacts:

- `aicas_run/qwen25_0p5b/results/qwen25_prefill_op_summary.json`
- `aicas_run/qwen25_0p5b/results/qwen25_prefill_matmul_shapes.json`
- `aicas_run/qwen25_0p5b/results/qwen25_prefill_unsupported_ops.md`

Current status:

- Added analyzer: `aicas_run/qwen25_0p5b/scripts/analyze_prefill_onnx.py`.
- Ran ONNX shape inference on `gpu01` before extracting MatMul/Gemm shapes.
- Full prefill ONNX has `2998` nodes and `291` initializers.
- No custom domains were found; in particular, no `com.microsoft` custom ops were found.
- `MatMul = 218`, `Gemm = 0`.
- `169/218` MatMul ops use static initializer as the second input.
- All `218` MatMul ops have inferred M/K/N shapes.
- Major MatMul shape groups:
  - `128x896x128`: 48 ops, all static-B
  - `128x896x4864`: 48 ops, all static-B
  - `128x4864x896`: 24 ops, all static-B
  - `128x64x128`: 24 ops, attention score MatMul, dynamic-B
  - `128x896x896`: 24 ops, all static-B
  - `128x128x64`: 24 ops, attention value MatMul, dynamic-B
  - `128x896x151936`: 1 op, LM head, static-B
- Risk ops from the current heuristic list:
  - `Equal`: 55
  - `Where`: 55
  - `ScatterND`: 1
- RMSNorm appears as standard ONNX scalar/reduction subgraphs (`ReduceMean`, `Pow`, `Add`, `Sqrt`, `Div`, `Mul`) rather than a custom op.
- RoPE/attention path appears as standard ONNX ops rather than custom domains.

## Milestone 4: onnx-mlir CPU Compile

- [ ] Compile the fixed-shape prefill ONNX through the CPU/LLVM path.
- [ ] Record whether external constants are emitted.
- [ ] Run the compiled CPU artifact if possible.
- [ ] Compare compiled output with ORT baseline.
- [ ] Document the first compile blocker if CPU compile fails.

Example command:

```bash
build/Release/bin/onnx-mlir \
  aicas_run/qwen25_0p5b/models/qwen25_prefill_128.onnx \
  -o aicas_run/qwen25_0p5b/tmp/qwen25_prefill_128
```

## Milestone 5: Subgraph-First NPU Bring-Up

- [ ] Extract or construct an MLP block subgraph.
- [ ] Try NPU lowering for the MLP block.
- [ ] Extract or construct the LM head subgraph.
- [ ] Try NPU lowering for the LM head.
- [ ] Extract or construct an attention block subgraph.
- [ ] Identify RoPE, mask, softmax, and layout blockers.
- [ ] Record NPU coverage for each subgraph.

Preferred order:

1. MLP block
2. LM head
3. Attention block
4. Full prefill

## Milestone 6: Full Prefill NPU Attempt

- [ ] Run full prefill through the NPU-oriented pipeline.
- [ ] Check whether major MatMul ops lower to NPU-compatible forms.
- [ ] Check memory planning behavior.
- [ ] Collect unsupported op list from the full graph.
- [ ] Collect latency or estimated latency if execution is available.
- [ ] Compare output against ORT baseline where possible.

## Milestone 7: Decode Preparation

- [ ] Inspect Qwen2.5 past KV input/output naming.
- [ ] Define fixed decode shape for batch = 1, step = 1.
- [ ] Build a decode-only ONNX smoke model.
- [ ] Reuse prefill logits/KV outputs as decode inputs.
- [ ] Start decode only after prefill path is stable.

## Notes

- Do not treat the existing `aicas_run` SmolVLM2 results as Qwen2.5 results.
- Reuse the existing `aicas_run` scripts as references, not as direct drop-in runners.
- Avoid full-model NPU compilation as the first task; use subgraphs to isolate blockers.
- Keep every milestone tied to concrete artifacts so progress is easy to audit.
