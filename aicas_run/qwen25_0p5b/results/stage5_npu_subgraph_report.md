# Qwen2.5-0.5B Stage 5 NPU Subgraph Report

The FP16 controls retain Qwen topology and shapes. The QLinear proxies retain the same core MatMul shapes and exercise the current INT8 NPU lowering path.

| Subgraph | Kind | Result | Last successful stage | NPU library calls | NPUX ops | K loop markers |
| --- | --- | --- | --- | --- | --- | --- |
| attention_core_qlinear_npu_proxy | qlinear_proxy | pass | llvm_lowering | 11 | 47 | 2 |
| attention_core_fp16_reference | fp16_control | pass | npu_partition | 0 | 0 | 0 |
| attention_qlinear_npu_proxy | qlinear_proxy | fail | fold_memref_aliases | 14 | 0 | 2 |
| attention_fp16_reference | fp16_control | pass | npu_partition | 0 | 0 | 0 |
| lm_head_qlinear_npu_proxy | qlinear_proxy | pass | llvm_lowering | 4 | 17 | 1 |
| lm_head_fp16_reference | fp16_control | pass | npu_partition | 0 | 0 | 0 |
| mlp_qlinear_npu_proxy | qlinear_proxy | pass | llvm_lowering | 14 | 48 | 4 |
| mlp_fp16_reference | fp16_control | pass | npu_partition | 0 | 0 | 0 |

## Blocking Stages

- `attention_qlinear_npu_proxy` at `convert_linalg_to_npux`: /home/lqma/qwen25_0p5b/results/stage5_npu_subgraphs/attention_qlinear_npu_proxy/14_fold_memref_aliases.mlir:24:5: error: unsupported npu_transpose permutation for NPU lowering; rank=4

## Interpretation

- The current NPU partition is INT8/QDQ-oriented. Zero NPU library calls in an FP16 control is expected evidence of the quantization gate, not a successful FP16 NPU lowering.
- The MLP proxy covers Qwen's 896x4864 and 4864x896 projection shapes. RMSNorm and SwiGLU remain outside the proxy because the full Qwen path is FP16 and SiLU/gating is not an NPU partition target.
- The LM head proxy retains the 896x151936 projection shape, including its very large output channel dimension.
- The full attention proxy covers key layout transpose, score/value MatMul, and Softmax. The attention-core proxy receives a pre-transposed key to separate compute coverage from layout support. Both intentionally omit RoPE and causal-mask construction.

## Artifacts

- Per-stage MLIR and command logs are stored next to each case directory.
- The JSON summary captures every command return code and coverage count.
