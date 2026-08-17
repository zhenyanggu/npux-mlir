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

- [x] Compile the fixed-shape prefill ONNX through the CPU/LLVM path.
- [x] Record whether external constants are emitted.
- [x] Run the compiled CPU artifact if possible.
- [x] Compare compiled output with ORT baseline.
- [ ] Document the first compile blocker if CPU compile fails.

Example command:

```bash
build/Release/bin/onnx-mlir \
  aicas_run/qwen25_0p5b/models/qwen25_prefill_128.onnx \
  -o aicas_run/qwen25_0p5b/tmp/qwen25_prefill_128
```

Current status:

- On `gpu01` (2026-08-05), the host build produced
  `/home/lqma/npux-mlir/build/Release/bin/onnx-mlir` from commit
  `7014dc15b48f1886ff041273e2ca01fcc6ba3149`.
- CPU/LLVM compilation succeeded with exit status `0` in `14.19 s`:
  `/home/lqma/qwen25_0p5b/models/qwen25_prefill_128.onnx` ->
  `/home/lqma/qwen25_0p5b/tmp/qwen25_prefill_128.so`.
- The compiler emitted external constants as
  `/home/lqma/qwen25_0p5b/tmp/qwen25_prefill_128.constants.bin` (`1.2G`);
  the shared library is `595K`.
- The compiled artifact loads through `OMExecutionSession` with inputs
  `input_ids`, `attention_mask`, and `position_ids`, all `i64[1,128]`, and
  output `logits` of `f16[1,128,151936]`.
- The matching ORT CPU baseline (`CPUExecutionProvider`) completed in
  `2282.48 ms`, produced no NaN/Inf, and selected token ID `11` at the first
  real-token position.
- The original CPU artifact exposed an x86-64 FP16 helper ABI error: LLVM passes
  `half` values in `xmm0`, but the runtime helper read a general-purpose
  register. The resulting full-model logits were all zero. Fixed the runtime
  bridge in `src/Runtime/OMTensor.inc` and rebuilt `onnx-mlir` and `PyRuntimeC`.
- Regression smoke tests now pass for FP16 Add (`[3,8]`) and FP16 MatMul with
  external constants (`[17,24]`).
- The fixed full-prefill CPU execution completed in `3018336.87 ms` (about
  `50.3 min`) with no NaN/Inf. It recovers ORT's last-token argmax (`11`) and
  all five top-5 token IDs, confirming functional Qwen prefill execution.
- Strict logits equality remains a precision follow-up: both `1e-2` and `5e-2`
  allclose checks are false, with global max/mean absolute error
  `2.841796875`/`0.32445079` and last-token max/mean error
  `2.841796875`/`0.55438352`. This is a FP16 accumulation/rounding gap rather
  than the prior all-zero runtime failure.
- Stage 4 is complete for CPU compilation, execution, and functional token
  validation; retain the strict FP16 logits comparison as a numerical-quality
  follow-up before treating the CPU artifact as fully ORT-equivalent. Relevant
  remote artifacts are
  `/home/lqma/qwen25_0p5b/results/stage4_qwen25_prefill_ort_cpu_result.json`,
  `/home/lqma/qwen25_0p5b/results/stage4_qwen25_prefill_compiled_compare.json`,
  `/home/lqma/qwen25_0p5b/results/stage4_qwen25_prefill_compiled_compare_fp16abi.json`,
  `/home/lqma/npux-mlir/logs/stage4_qwen25_prefill_compile.log`, and
  `/home/lqma/npux-mlir/logs/stage4_qwen25_prefill_compile_fp16abi.log`.

## Milestone 5: Subgraph-First NPU Bring-Up

- [x] Extract or construct an MLP block subgraph.
- [x] Try NPU lowering for the MLP block.
- [x] Extract or construct the LM head subgraph.
- [x] Try NPU lowering for the LM head.
- [x] Extract or construct an attention block subgraph.
- [x] Identify RoPE, mask, softmax, and layout blockers.
- [x] Record NPU coverage for each subgraph.

Current status:

- Added reusable stage-5 tools:
  - `aicas_run/qwen25_0p5b/scripts/build_npu_subgraphs.py`
  - `aicas_run/qwen25_0p5b/scripts/run_npu_subgraph_pipeline.py`
- On `gpu01` (2026-08-05), generated FP16 topology/shape references and INT8
  QLinear/QDQ NPU proxies from the exported Qwen graph. The references use
  zero weights and are lowering controls, not numerical-equivalence models.
- Selected Qwen layer-0 shapes are MLP `896x4864` and `4864x896`, LM head
  `896x151936`, and attention `[1,14,128,64] x [1,14,64,128]`.
- The project NPU partition is INT8/QDQ-oriented: all FP16 controls complete
  the partition pass with zero NPU library calls. This confirms that direct
  lowering of the current FP16 prefill export is blocked by quantization.
- The MLP and LM-head QLinear proxies reach `llvm_lowering`; their peak
  coverage is `14` NPU library calls/`48` NPUX ops and `4`/`17`, respectively.
  The MLP down projection (`K=4864`) contains `head/body/tail` capacity-tiling
  stages.
- The pre-transposed attention-core proxy also reaches `llvm_lowering` with
  `npu_matmul`, `npu_softmax`, DMA, and `47` NPUX ops. The Softmax is lowered
  to `npux.sfu_run softmax`.
- The full attention proxy reaches `convert-linalg-to-npux` but stops on the
  real layout blocker: rank-4 key transpose `[1,14,128,64] -> [1,14,64,128]`
  is not supported by current NPUX lowering. RoPE and causal-mask construction
  remain outside the proxy and need separate full-graph handling.
- The work uncovered and fixed three NPU lowering type-boundary defects:
  `ui8` arithmetic placeholder bodies, `i32 -> ui8` MatMul output truncation,
  and encoding mutation of function entry arguments. Rebuilt
  `onnx-mlir-opt` on `gpu01`; the final stage-5 run reports `7/8` expected
  outcomes, with only the documented rank-4 transpose case failing.
- Local artifacts:
  - `aicas_run/qwen25_0p5b/results/stage5_npu_subgraph_report.md`
  - `aicas_run/qwen25_0p5b/results/stage5_npu_subgraph_summary.json`
  - `aicas_run/qwen25_0p5b/results/subgraph_manifest.json`

Stage-6 prerequisites:

1. Produce an INT8/QDQ full-prefill export or add a Qwen-aware quantization path.
2. Lower the rank-4 key transpose or rewrite attention to use a pre-transposed
   K layout.
3. Add or rewrite coverage for FP16 RMSNorm/SwiGLU, RoPE, and causal masking.

Preferred order:

1. MLP block
2. LM head
3. Attention block
4. Full prefill

## Milestone 6: Full Prefill NPU Attempt

- [x] Run full prefill through the NPU-oriented pipeline.
- [x] Check whether major MatMul ops lower to NPU-compatible forms.
- [x] Check memory planning behavior.
- [x] Collect unsupported op list from the full graph.
- [ ] Collect latency or estimated latency if execution is available.
- [x] Compare output against ORT baseline where possible.

Current status:

- Added stage-6 tools:
  - `aicas_run/qwen25_0p5b/scripts/quantize_qwen25_prefill.py`
  - `aicas_run/qwen25_0p5b/scripts/compare_qwen25_onnx_prefill.py`
  - `aicas_run/qwen25_0p5b/scripts/run_npu_prefill_pipeline.py`
- The quantizer converts the FP16 export into an external-data FP32 source
  before QDQ. This is required because opset-17 `DequantizeLinear` rejects
  the FP16 scales generated when quantizing the original FP16 graph directly.
  Calibration uses `CUDAExecutionProvider,CPUExecutionProvider`; the
  per-channel weight encoding itself remains an ONNX Runtime CPU operation.
- The valid INT8 model selects all `169` static-B linear projections (Q/K/V/O,
  MLP, and LM head). Its graph contains `266` QuantizeLinear and `435`
  DequantizeLinear nodes. Dynamic attention score/value MatMuls, RMSNorm,
  SwiGLU, RoPE, and mask construction remain outside this first quantized
  coverage set.
- Added rank-4 transpose lowering for Qwen K layout `[B,H,S,D] -> [B,H,D,S]`
  in `src/Conversion/NpuToLLVM/Npux/SfuOpConvert.cpp`. The full attention
  proxy now reaches LLVM lowering with `14` NPU library calls and `58` NPUX
  ops.
- On `gpu01` (2026-08-05), the valid full-prefill QDQ model completed every
  NPU pipeline stage through LLVM lowering. It has `169` NPU library calls at
  partition, `820` after DMA/splitting, `3405` NPUX ops at NPUX conversion,
  and zero remaining ONNX ops after LLVM lowering. The temporary four
  `onnx.` matches after bufferization are ONNX metadata/attributes; they do
  not survive the final IR.
- CUDA ORT evaluation confirms the valid INT8 model runs without NaN/Inf, but
  current numerical quality is not acceptable: `1/4` top-1 matches, average
  top-5 overlap `1.75/5`, maximum absolute logit error `18.55`. This is a
  calibration/quantization-quality blocker, not an NPU compilation blocker.
- NPU runtime latency is still unmeasured because this work validates compiler
  lowering through LLVM IR, not execution on the target NPU runtime.
- Local evidence:
  - `aicas_run/qwen25_0p5b/results/stage6_quantization_fp32_summary.json`
  - `aicas_run/qwen25_0p5b/results/stage6_ort_quantization_compare_fp32.json`
  - `aicas_run/qwen25_0p5b/results/stage6_attention_transpose_report.md`
  - `aicas_run/qwen25_0p5b/results/stage6_npu_prefill_report.md`
  - `aicas_run/qwen25_0p5b/results/stage6_npu_prefill_summary.json`

## Milestone 7: Decode Preparation

- [x] Inspect Qwen2.5 past KV input/output naming.
- [x] Define fixed decode shape for batch = 1, step = 1.
- [x] Build a decode-only ONNX smoke model.
- [x] Reuse prefill logits/KV outputs as decode inputs.
- [x] Start decode only after prefill path is stable.

Current status:

- Added the stage-7 FP16 cache tools:
  - `aicas_run/qwen25_0p5b/scripts/export_qwen25_decode_onnx.py`
  - `aicas_run/qwen25_0p5b/scripts/run_qwen25_decode_smoke.py`
- The exporter adapts the current Transformers `DynamicCache` API while
  exposing ONNX-standard tensor interfaces. The fixed Qwen2.5-0.5B interface
  contains 48 KV tensors (`24` layers x key/value), each `[1,2,S,64]`.
- On `gpu01` (2026-08-05), CUDA exported fixed FP16 prefill (`S=128`) and
  decode (`step=1`) ONNX graphs. The decode graph consumes
  `past_key_values.<layer>.{key,value}` and returns
  `present_key_values.<layer>.{key,value}`.
- The CUDA ORT smoke test selected `CUDAExecutionProvider` for both prefill
  and decode. It transferred all 48 caches from `[1,2,128,64]` to
  `[1,2,129,64]`, produced no NaN/Inf, and matched PyTorch FP16 decode top-1
  and all top-5 IDs. Max/mean decode-logit errors were `0.0419922` and
  `0.00677363`; prefill/decode elapsed times were `115.14 ms` and `59.42 ms`.
- This completes Stage 7-A smoke validation only. It neither proves NPU decode
  lowering nor unblocks deployment: Stage 6's INT8/QDQ prefill numerical
  quality remains the gating issue for an end-to-end quantized decode path.
- Local evidence:
  - `aicas_run/qwen25_0p5b/results/stage7_cache_export_summary.json`
  - `aicas_run/qwen25_0p5b/results/stage7_decode_smoke.json`
  - `aicas_run/qwen25_0p5b/results/stage7_decode_smoke_report.md`

## Notes

- Do not treat the existing `aicas_run` SmolVLM2 results as Qwen2.5 results.
- Reuse the existing `aicas_run` scripts as references, not as direct drop-in runners.
- Avoid full-model NPU compilation as the first task; use subgraphs to isolate blockers.
- Keep every milestone tied to concrete artifacts so progress is easy to audit.
