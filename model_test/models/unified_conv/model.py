import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WrappedConvModel(nn.Module):
    """Build the unified Conv testcase with the same structure as the original Conv case."""

    def __init__(self):
        """Create a 64-channel Conv2d layer that keeps the spatial shape unchanged."""
        super().__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        """Wrap Conv with tiny offsets to keep the target node isolated in the graph."""
        x = x + 1e-3
        x = self.conv(x)
        x = x - 1e-3
        return x


class RandomDataReader(CalibrationDataReader):
    """Generate deterministic calibration batches for ONNX Runtime static quantization."""

    def __init__(self, input_name, input_shape, num_batches=10, seed=2026):
        """Store the calibration contract and prepare the lazy iterator."""
        self.input_name = input_name
        self.input_shape = tuple(input_shape)
        self.num_batches = num_batches
        self.seed = seed
        self._iterator = iter(self._generate())

    def _generate(self):
        """Materialize float32 calibration samples under one fixed RNG seed."""
        rng = np.random.default_rng(self.seed)
        data = []
        for _ in range(self.num_batches):
            sample = rng.standard_normal(size=self.input_shape).astype(np.float32)
            data.append({self.input_name: sample})
        return data

    def get_next(self):
        """Return the next calibration sample expected by ONNX Runtime."""
        return next(self._iterator, None)


def check_symmetric_zero_points(quant_model_path):
    """Verify that every int8 zero point in the quantized model stays at 0."""
    model = onnx.load(quant_model_path)
    for initializer in model.graph.initializer:
        if "zero_point" not in initializer.name:
            continue
        if initializer.data_type != onnx.TensorProto.INT8:
            continue
        values = np.frombuffer(initializer.raw_data, dtype=np.int8)
        if values.size and not np.all(values == 0):
            raise RuntimeError(
                f"INT8 zero point is not 0 in initializer '{initializer.name}': {values}"
            )


def main():
    """Export, quantize, and serialize the unified Conv testcase artifacts."""
    model_name = "unified_conv"
    workdir = os.path.dirname(os.path.abspath(__file__))
    fp32_model_path = os.path.join(workdir, "model_fp32.onnx")
    fp32_external_data_path = os.path.join(workdir, "model_fp32.onnx.data")
    model_path = os.path.join(workdir, "model.onnx")
    input_path = os.path.join(workdir, f"{model_name}_input.bin")
    golden_path = os.path.join(workdir, f"{model_name}_output_golden.bin")

    input_shape = (1, 64, 32, 48)

    torch.manual_seed(2026)
    np.random.seed(2026)

    model = WrappedConvModel().eval()
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        fp32_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=20,
        do_constant_folding=True,
    )

    reader = RandomDataReader("input", input_shape, 10)
    quantize_static(
        model_input=fp32_model_path,
        model_output=model_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv"],
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )

    check_symmetric_zero_points(model_path)

    test_input = np.random.randn(*input_shape).astype(np.float32)
    test_input.tofile(input_path)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: test_input})[0]
    output.astype(np.float32).tofile(golden_path)

    if os.path.exists(fp32_model_path):
        os.remove(fp32_model_path)
    if os.path.exists(fp32_external_data_path):
        os.remove(fp32_external_data_path)

    print("Generated files:")
    print(f"  {os.path.basename(model_path)}")
    print(f"  {os.path.basename(input_path)}")
    print(f"  {os.path.basename(golden_path)}")
    print(f"Input shape: {input_shape}, dtype: float32")


if __name__ == "__main__":
    main()
