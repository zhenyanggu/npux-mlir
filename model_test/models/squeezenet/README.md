# SqueezeNet Test

Place the source ONNX model at:

```text
model_test/cache/squeezenet.onnx
```

The test uses the shared Imagenette subset under `model_test/cache/imagenette2-320`
to generate calibration data, inputs, labels, and golden outputs. This model is a
lightweight CNN candidate for the layer-fusion ablation. If lowering fails, check
whether the downloaded ONNX contains unsupported operators such as concat-style
fire-module patterns.

