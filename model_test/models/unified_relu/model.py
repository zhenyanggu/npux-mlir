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


class ConvReluModel(nn.Module):
    """Build the unified Conv+Relu testcase with the original relu-case topology."""

    def __init__(self):
        """Create the Conv2d and ReLU layers used by the testcase."""
        super().__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Wrap Conv and ReLU with tiny offsets so the exported graph stays stable."""
        x = x + 1e-3
        x = self.conv(x)
        x = self.relu(x)
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


def check_qdq_pattern(quant_model_path, op_type):
    """Verify that the target op is surrounded by DQ on inputs and Q on outputs."""
    model = onnx.load(quant_model_path)
    nodes = list(model.graph.node)

    dq_outputs = set()
    q_inputs = set()
    for node in nodes:
        if node.op_type == "DequantizeLinear" and len(node.output) > 0:
            dq_outputs.add(node.output[0])
        if node.op_type == "QuantizeLinear" and len(node.input) > 0:
            q_inputs.add(node.input[0])

    matched = 0
    total = 0
    for node in nodes:
        if node.op_type != op_type:
            continue
        total += 1
        has_dq_before = any(inp in dq_outputs for inp in node.input)
        has_q_after = any(out in q_inputs for out in node.output)
        if has_dq_before and has_q_after:
            matched += 1

    if total == 0:
        raise RuntimeError(f"No '{op_type}' node found after quantization")
    if matched != total:
        raise RuntimeError(
            f"QDQ pattern check failed for '{op_type}': matched {matched}/{total}"
        )


def main():
    """Export, quantize, and serialize the unified Conv+Relu testcase artifacts."""
    model_name = "unified_relu"
    workdir = os.path.dirname(os.path.abspath(__file__))
    fp32_model_path = os.path.join(workdir, "model_fp32.onnx")
    fp32_external_data_path = os.path.join(workdir, "model_fp32.onnx.data")
    model_path = os.path.join(workdir, "model.onnx")
    input_path = os.path.join(workdir, f"{model_name}_input.bin")
    golden_path = os.path.join(workdir, f"{model_name}_output_golden.bin")

    input_shape = (1, 64, 32, 48)

    torch.manual_seed(2026)
    np.random.seed(2026)

    model = ConvReluModel().eval()
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
        op_types_to_quantize=["Conv", "Relu"],
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )

    check_symmetric_zero_points(model_path)
    check_qdq_pattern(model_path, "Conv")
    check_qdq_pattern(model_path, "Relu")

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
    print("QDQ check:")
    print("  Conv: DQ -> Conv -> Q")
    print("  Relu: DQ -> Relu -> Q")
    print(f"Input shape: {input_shape}, dtype: float32")


if __name__ == "__main__":
    main()
