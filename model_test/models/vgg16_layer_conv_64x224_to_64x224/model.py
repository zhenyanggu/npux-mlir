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

MODEL_NAME = "vgg16_layer_conv_64x224_to_64x224"
OP_TYPE = "Conv"
INPUT_SHAPE = (1, 64, 224, 224)
OUTPUT_SHAPE = (1, 64, 224, 224)
KERNEL = (3, 3)
STRIDE = (1, 1)
PADS = (1, 1, 1, 1)
AXIS = 1


class SingleLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        if OP_TYPE == "Conv":
            in_c = INPUT_SHAPE[1]
            out_c = OUTPUT_SHAPE[1]
            padding = (PADS[0], PADS[1])
            self.op = nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=tuple(KERNEL),
                stride=tuple(STRIDE),
                padding=padding,
                bias=True,
            )
        elif OP_TYPE == "Relu":
            self.op = nn.ReLU()
        elif OP_TYPE == "MaxPool":
            padding = (PADS[0], PADS[1])
            self.op = nn.MaxPool2d(
                kernel_size=tuple(KERNEL),
                stride=tuple(STRIDE),
                padding=padding,
            )
        elif OP_TYPE == "Gemm":
            self.op = nn.Linear(INPUT_SHAPE[-1], OUTPUT_SHAPE[-1], bias=True)
        elif OP_TYPE == "Flatten":
            self.op = None
        else:
            raise RuntimeError(f"Unsupported OP_TYPE: {OP_TYPE}")

    def forward(self, x):
        x = x + 1e-3
        if OP_TYPE == "Flatten":
            x = torch.flatten(x, start_dim=AXIS)
        else:
            x = self.op(x)
        x = x - 1e-3
        return x


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


def ensure_layer_op_exists(model_path):
    model = onnx.load(model_path)
    op_types = {node.op_type for node in model.graph.node}
    expected_alias = {
        "Conv": {"Conv"},
        "Relu": {"Relu"},
        "MaxPool": {"MaxPool"},
        "Flatten": {"Flatten", "Reshape"},
        "Gemm": {"Gemm", "MatMul"},
    }
    required = expected_alias[OP_TYPE]
    if op_types.isdisjoint(required):
        raise RuntimeError(
            f"Expected layer op {OP_TYPE} not found; "
            f"accepted={sorted(required)}, present={sorted(op_types)}"
        )


def main():
    workdir = os.path.dirname(os.path.abspath(__file__))
    fp32_model_path = os.path.join(workdir, "model_fp32.onnx")
    model_path = os.path.join(workdir, "model.onnx")
    input_path = os.path.join(workdir, f"{MODEL_NAME}_input.bin")
    golden_path = os.path.join(workdir, f"{MODEL_NAME}_output_golden.bin")

    torch.manual_seed(2026)
    np.random.seed(2026)

    model = SingleLayerModel().eval()
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

    quantizable_ops = {"Conv", "Relu", "MaxPool", "Gemm"}
    if OP_TYPE in quantizable_ops:
        reader = RandomDataReader("input", INPUT_SHAPE, num_batches=10, seed=2026)
        quantize_static(
            model_input=fp32_model_path,
            model_output=model_path,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=[OP_TYPE],
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
        )
        if os.path.exists(fp32_model_path):
            os.remove(fp32_model_path)
    else:
        if os.path.exists(model_path):
            os.remove(model_path)
        os.rename(fp32_model_path, model_path)

    ensure_layer_op_exists(model_path)

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
    print(f"Case: {MODEL_NAME}, op={OP_TYPE}")
    print(f"Input shape: {INPUT_SHAPE}")
    print(f"Output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
