import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


class RandomDataReader(CalibrationDataReader):
    """Generate deterministic float32 calibration batches for static quantization."""

    def __init__(self, input_name, input_shape, num_batches=10, seed=2026):
        """Remember the input contract and prepare a lazy batch iterator."""
        self.input_name = input_name
        self.input_shape = tuple(input_shape)
        self.num_batches = num_batches
        self.seed = seed
        self._iterator = iter(self._generate())

    def _generate(self):
        """Materialize calibration samples with a fixed random seed."""
        rng = np.random.default_rng(self.seed)
        data = []
        for _ in range(self.num_batches):
            sample = rng.standard_normal(size=self.input_shape).astype(np.float32)
            data.append({self.input_name: sample})
        return data

    def get_next(self):
        """Return the next calibration sample expected by ONNX Runtime."""
        return next(self._iterator, None)


def configure_seed(seed=2026):
    """Synchronize NumPy and PyTorch RNG state for reproducible artifacts."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def validate_zero_points_are_symmetric(model_path):
    """Ensure every generated int8 zero point stays at the symmetric value 0."""
    model = onnx.load(model_path)
    for initializer in model.graph.initializer:
        if "zero_point" not in initializer.name:
            continue
        if initializer.data_type != TensorProto.INT8:
            continue
        values = np.frombuffer(initializer.raw_data, dtype=np.int8)
        if values.size and not np.all(values == 0):
            raise RuntimeError(
                f"INT8 zero point is not 0 in initializer '{initializer.name}': {values}"
            )


def validate_op_exists(model_path, op_type):
    """Check that the exported ONNX graph contains the target operator type."""
    model = onnx.load(model_path)
    if not any(node.op_type == op_type for node in model.graph.node):
        raise RuntimeError(f"Target op '{op_type}' is missing in '{model_path}'")


def export_quantized_model(
    model,
    input_shape,
    model_name,
    workdir,
    op_types_to_quantize,
    seed=2026,
    calibration_batches=10,
):
    """Export a torch model to ONNX, quantize it, and emit input/golden artifacts."""
    fp32_model_path = os.path.join(workdir, "model_fp32.onnx")
    fp32_external_data_path = os.path.join(workdir, "model_fp32.onnx.data")
    model_path = os.path.join(workdir, "model.onnx")
    input_path = os.path.join(workdir, f"{model_name}_input.bin")
    golden_path = os.path.join(workdir, f"{model_name}_output_golden.bin")

    configure_seed(seed)
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    torch.onnx.export(
        model.eval(),
        dummy_input,
        fp32_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=20,
        do_constant_folding=True,
    )

    reader = RandomDataReader("input", input_shape, calibration_batches, seed=seed)
    quantize_static(
        model_input=fp32_model_path,
        model_output=model_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=list(op_types_to_quantize),
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )

    validate_zero_points_are_symmetric(model_path)
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

    return model_path, input_path, golden_path


def symmetric_scale_from_max_abs(max_abs):
    """Convert a floating-point range into a symmetric int8 quantization scale."""
    return float(max(max_abs / 127.0, 1e-6))


def quantize_to_int8(array, scale):
    """Round a float32 tensor into the signed int8 range under one scale."""
    return np.clip(np.round(array / scale), -128, 127).astype(np.int8)


def save_float_io(model_path, input_path, golden_path, input_array):
    """Run ONNX Runtime once and serialize the test input and golden output."""
    input_array.astype(np.float32).tofile(input_path)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_array.astype(np.float32)})[0]
    output.astype(np.float32).tofile(golden_path)


def print_generated_summary(model_name, input_shape, extra_lines=None):
    """Print stable generation logs that match the existing model_test style."""
    print("Generated files:")
    print("  model.onnx")
    print(f"  {model_name}_input.bin")
    print(f"  {model_name}_output_golden.bin")
    print(f"Input shape: {tuple(input_shape)}, dtype: float32")
    for line in extra_lines or []:
        print(line)
