# Qwen2.5-0.5B Prefill ONNX Compile Risk Report

Model: `/home/lqma/qwen25_0p5b/models/qwen25_prefill_128.onnx`
Node count: `2998`
Initializer count: `291`

## Custom Domains

- None

## Risk Ops

- `Equal`: 55
- `Where`: 55
- `ScatterND`: 1

## Notes

- Shape/layout ops are not necessarily unsupported, but they may dominate lowering complexity.
- MatMul/Gemm details are stored in the MatMul shape JSON.
