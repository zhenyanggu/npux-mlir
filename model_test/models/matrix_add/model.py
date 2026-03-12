import argparse
import os

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


class MatrixAddModel(nn.Module):
    def __init__(self, rows, cols, seed):
        super().__init__()
        rng = np.random.default_rng(seed)
        bias = rng.standard_normal(size=(rows, cols)).astype(np.float32) * 0.125
        self.register_buffer("bias", torch.from_numpy(bias))

    def forward(self, x):
        x = x + 1e-3
        x = x + self.bias
        x = x - 1e-3
        return x


class RandomDataReader(CalibrationDataReader):
    def __init__(self, input_name, input_shape, num_batches, seed):
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


def check_symmetric_zero_points(model_path):
    model = onnx.load(model_path)
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


def check_qdq_add_pattern(model_path):
    model = onnx.load(model_path)
    producer = {}
    consumers = {}
    for node in model.graph.node:
        for output in node.output:
            producer[output] = node
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    add_total = 0
    add_qdq_count = 0
    for node in model.graph.node:
        if node.op_type != "Add":
            continue
        add_total += 1
        has_dq_input = any(
            input_name in producer and producer[input_name].op_type == "DequantizeLinear"
            for input_name in node.input
        )
        has_q_output = any(
            consumer.op_type == "QuantizeLinear"
            for out_name in node.output
            for consumer in consumers.get(out_name, [])
        )
        if has_dq_input and has_q_output:
            add_qdq_count += 1

    if add_total == 0:
        raise RuntimeError("No Add node found in quantized model")
    if add_qdq_count == 0:
        raise RuntimeError("QDQ pattern check failed: no DQ -> Add -> Q pattern found")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate matrix_add model/input/golden")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--rows", type=int, default=128, help="Matrix row count")
    parser.add_argument("--cols", type=int, default=256, help="Matrix col count")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--calib-batches", type=int, default=10, help="Calibration batch count")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch <= 0 or args.rows <= 0 or args.cols <= 0:
        raise ValueError("batch/rows/cols must all be > 0")

    model_name = "matrix_add"
    workdir = os.path.dirname(os.path.abspath(__file__))
    fp32_model_path = os.path.join(workdir, "model_fp32.onnx")
    model_path = os.path.join(workdir, "model.onnx")
    input_path = os.path.join(workdir, f"{model_name}_input.bin")
    golden_path = os.path.join(workdir, f"{model_name}_output_golden.bin")

    input_shape = (args.batch, args.rows, args.cols)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = MatrixAddModel(args.rows, args.cols, args.seed).eval()
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

    reader = RandomDataReader("input", input_shape, args.calib_batches, args.seed)
    quantize_static(
        model_input=fp32_model_path,
        model_output=model_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Add"],
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )

    check_symmetric_zero_points(model_path)
    check_qdq_add_pattern(model_path)

    test_input = np.random.randn(*input_shape).astype(np.float32)
    test_input.tofile(input_path)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: test_input})[0]
    output.astype(np.float32).tofile(golden_path)

    if os.path.exists(fp32_model_path):
        os.remove(fp32_model_path)

    print("Generated files:")
    print(f"  {os.path.basename(model_path)}")
    print(f"  {os.path.basename(input_path)}")
    print(f"  {os.path.basename(golden_path)}")
    print(
        f"Input shape: {input_shape}, dtype: float32, "
        f"calibration batches: {args.calib_batches}, seed: {args.seed}"
    )
    print("Quantization: static QDQ, symmetric int8, op_types_to_quantize=[Add]")
    print("Pattern check: PASS (found at least one DequantizeLinear -> Add -> QuantizeLinear)")


if __name__ == "__main__":
    main()
