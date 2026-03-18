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

MODEL_NAME = "resnet50_layer_021_convrelu_1024x14x14_to_512x7x7_k1x1_s2x2_p0x0"
OP_TYPE = "ConvRelu"
INPUT_SHAPE = (1, 1024, 14, 14)
OUTPUT_SHAPE = (1, 512, 7, 7)
KERNEL = (1, 1)
STRIDE = (2, 2)
PADS = (0, 0, 0, 0)
DILATIONS = (1, 1)
GROUP = 1
AXIS = 1
AXES = (2, 3)
KEEPDIMS = 1


class SingleLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        if OP_TYPE in {"Conv", "ConvRelu"}:
            in_c = INPUT_SHAPE[1]
            out_c = OUTPUT_SHAPE[1]
            padding = (PADS[0], PADS[1])
            self.op = nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=tuple(KERNEL),
                stride=tuple(STRIDE),
                padding=padding,
                dilation=tuple(DILATIONS),
                groups=GROUP,
                bias=True,
            )
            self.relu = nn.ReLU() if OP_TYPE == "ConvRelu" else None
        elif OP_TYPE == "Relu":
            self.op = nn.ReLU()
        elif OP_TYPE == "Add":
            self.op = None
        elif OP_TYPE == "MaxPool":
            padding = (PADS[0], PADS[1])
            self.op = nn.MaxPool2d(
                kernel_size=tuple(KERNEL),
                stride=tuple(STRIDE),
                padding=padding,
            )
        elif OP_TYPE == "ReduceMean":
            self.op = None
        elif OP_TYPE == "Flatten":
            self.op = None
        elif OP_TYPE == "Gemm":
            self.op = nn.Linear(INPUT_SHAPE[-1], OUTPUT_SHAPE[-1], bias=True)
        else:
            raise RuntimeError(f"Unsupported OP_TYPE: {OP_TYPE}")

    def forward(self, x):
        x = x + 1e-3
        if OP_TYPE == "ConvRelu":
            x = self.relu(self.op(x))
        elif OP_TYPE == "Add":
            x = x + x
        elif OP_TYPE == "ReduceMean":
            x = torch.mean(x, dim=tuple(AXES), keepdim=bool(KEEPDIMS))
        elif OP_TYPE == "Flatten":
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
        "ConvRelu": {"Conv", "Relu"},
        "Relu": {"Relu"},
        "Add": {"Add"},
        "MaxPool": {"MaxPool", "MaxPoolSingleOut"},
        "ReduceMean": {"ReduceMean", "ReduceMeanV13"},
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

    quantizable_ops = {"Conv", "ConvRelu", "Relu", "MaxPool", "Gemm", "Add"}
    quant_done = False
    if OP_TYPE in quantizable_ops:
        try:
            if OP_TYPE == "ConvRelu":
                q_ops = ["Conv", "Relu"]
            else:
                q_ops = [OP_TYPE]
            reader = RandomDataReader("input", INPUT_SHAPE, num_batches=10, seed=2026)
            quantize_static(
                model_input=fp32_model_path,
                model_output=model_path,
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                op_types_to_quantize=q_ops,
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
