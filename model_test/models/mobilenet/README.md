# MobileNetV2 Test

Place the source ONNX model at:

```text
model_test/cache/mobilenetv2-12.onnx
```

This test intentionally does not quantize `Conv`, because MobileNetV2 contains
depthwise/group convolutions that the current NPU Conv path does not support.

If lowering still tries to place grouped Conv on the NPU, build with `NPU_OPS`
excluding `Conv`, for example:

```bash
make -B mobilenet EXPERIMENT_NAME=fusion_baseline \
  NPU_OPS=Add,MatMul,LayerNorm,Softmax,Gelu,Gemm,Transpose,MaxPool \
  NPU_REMOVE_REDUNDANT_DMA=1
```

