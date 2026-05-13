# LeNet-5 Test

Place the source ONNX model at:

```text
model_test/cache/lenet5.onnx
```

Expected model interface:

```text
input : tensor<1x1x32x32xf32>
output: tensor<1x10xf32>
```

The test reuses the MNIST dataset from `model_test/cache/mnist`. Original
28x28 images are zero-padded to 32x32 before calibration, golden generation, and
board execution, matching the classic LeNet-5 input size.

