# Qwen2.5-0.5B Stage 6 NPU Prefill Report

Source model: `/home/lqma/qwen25_0p5b/models/qwen25_prefill_128_int8_static_fp32.onnx`
NPU targets: `MatMul,Softmax,Transpose`

| Stage | Result | NPU calls | NPUX ops | Remaining ONNX ops |
| --- | --- | --- | --- | --- |
| import | pass | 0 | 0 | 3459 |
| recompose | pass | 0 | 0 | 2290 |
| npu_partition | pass | 169 | 0 | 2121 |
| npu_tiling | pass | 217 | 0 | 1948 |
| npu_insert_dma | pass | 820 | 0 | 1948 |
| npu_op_splitting | pass | 820 | 0 | 1948 |
| npu_bufferization | pass | 820 | 0 | 4 |
| convert_vector_to_scf | pass | 820 | 0 | 4 |
| lower_affine | pass | 820 | 0 | 4 |
| lower_krnl_region | pass | 820 | 0 | 4 |
| buffer_loop_hoisting | pass | 820 | 0 | 4 |
| buffer_dealloc | pass | 820 | 0 | 4 |
| allocation_liveness | pass | 820 | 0 | 4 |
| convert_bufferization_to_memref | pass | 820 | 0 | 4 |
| fold_memref_aliases | pass | 820 | 0 | 4 |
| convert_linalg_to_npux | pass | 0 | 3405 | 4 |
| lower_npu_subview | pass | 0 | 3405 | 4 |
| npux_compute_fusion | pass | 0 | 3236 | 4 |
| convert_linalg_to_loops | pass | 0 | 3236 | 4 |
| npu_memory_plan | pass | 0 | 3236 | 4 |
| erase_npu_memory_space | pass | 0 | 3236 | 4 |
| llvm_lowering | pass | 0 | 0 | 0 |
