import os
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

MODEL_NAME = "resnet50_block_04_stage2_identity"
INPUT_SHAPE = (1, 512, 28, 28)
OUTPUT_SHAPE = (1, 512, 28, 28)
MID_CHANNELS = 128
OUT_CHANNELS = 512
STRIDE = 1
USE_PROJECTION = 0


class ResidualBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = INPUT_SHAPE[1]
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=MID_CHANNELS,
            kernel_size=1,
            stride=STRIDE,
            padding=0,
            bias=True,
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=MID_CHANNELS,
            out_channels=MID_CHANNELS,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=MID_CHANNELS,
            out_channels=OUT_CHANNELS,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        if USE_PROJECTION:
            self.proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=OUT_CHANNELS,
                kernel_size=1,
                stride=STRIDE,
                padding=0,
                bias=True,
            )
        else:
            self.proj = None
        self.out_relu = nn.ReLU()

    def forward(self, x):
        x = x + 1e-3
        identity = x if self.proj is None else self.proj(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = out + identity
        out = self.out_relu(out)
        out = out - 1e-3
        return out


class RandomDataReader(CalibrationDataReader):
    def __init__(self, input_name, input_shape, num_batches=10, seed=2026):
        self.input_name = input_name
        self.input_shape = tuple(input_shape)
        self.num_batches = num_batches
        self.seed = seed
        self._iterator = iter(self._generate())

    def _generate(self):
        rng = np.random.default_rng(self.seed)
        data = []
        for _ in range(self.num_batches):
            sample = rng.standard_normal(size=self.input_shape).astype(np.float32)
            data.append({self.input_name: sample})
        return data

    def get_next(self):
        return next(self._iterator, None)


def ensure_block_ops_exist(model_path):
    model = onnx.load(model_path)
    op_types = {node.op_type for node in model.graph.node}
    required = {"Conv", "Relu", "Add"}
    if not required.issubset(op_types):
        raise RuntimeError(
            f"Residual block model missing required ops {sorted(required)}; "
            f"present={sorted(op_types)}"
        )


def main():
    workdir = os.path.dirname(os.path.abspath(__file__))
    fp32_model_path = os.path.join(workdir, "model_fp32.onnx")
    model_path = os.path.join(workdir, "model.onnx")
    input_path = os.path.join(workdir, f"{MODEL_NAME}_input.bin")
    golden_path = os.path.join(workdir, f"{MODEL_NAME}_output_golden.bin")

    torch.manual_seed(2026)
    np.random.seed(2026)

    model = ResidualBottleneck().eval()
    dummy_input = torch.randn(*INPUT_SHAPE, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        fp32_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=20,
        do_constant_folding=True,
    )

    quant_done = False
    try:
        reader = RandomDataReader("input", INPUT_SHAPE, num_batches=10, seed=2026)
        quantize_static(
            model_input=fp32_model_path,
            model_output=model_path,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=["Conv", "Relu", "Add"],
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
        )
        quant_done = True
    except Exception as err:
        print(f"[warn] quantization failed for {MODEL_NAME}: {err}; fallback to fp32 model")

    if quant_done:
        if os.path.exists(fp32_model_path):
            os.remove(fp32_model_path)
    else:
        if os.path.exists(model_path):
            os.remove(model_path)
        os.rename(fp32_model_path, model_path)

    ensure_block_ops_exist(model_path)

    test_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    test_input.tofile(input_path)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: test_input})[0]
    output.astype(np.float32).tofile(golden_path)

    print("Generated files:")
    print(f"  {os.path.basename(model_path)}")
    print(f"  {os.path.basename(input_path)}")
    print(f"  {os.path.basename(golden_path)}")
    print(f"Case: {MODEL_NAME}, op=ResidualBlock")
    print(f"Input shape: {INPUT_SHAPE}")
    print(f"Output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
