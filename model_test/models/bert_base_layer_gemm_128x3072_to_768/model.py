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

MODEL_NAME = "bert_base_layer_gemm_128x3072_to_768"
INPUT_SHAPE = (128, 3072)
OUT_FEATURES = 768

class GemmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(INPUT_SHAPE[-1], OUT_FEATURES, bias=True)

    def forward(self, x):
        x = x + 1e-3
        x = self.fc(x)
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


def check_symmetric_zero_points(model_path):
    model = onnx.load(model_path)
    for initializer in model.graph.initializer:
        if "zero_point" not in initializer.name:
            continue
        if initializer.data_type != onnx.TensorProto.INT8:
            continue
        values = np.frombuffer(initializer.raw_data, dtype=np.int8)
        if values.size and not np.all(values == 0):
            raise RuntimeError(f"INT8 zero point is not 0 in initializer '{initializer.name}'")


def main():
    workdir = os.path.dirname(os.path.abspath(__file__))
    fp32_model_path = os.path.join(workdir, "model_fp32.onnx")
    model_path = os.path.join(workdir, "model.onnx")
    input_path = os.path.join(workdir, f"{MODEL_NAME}_input.bin")
    golden_path = os.path.join(workdir, f"{MODEL_NAME}_output_golden.bin")

    torch.manual_seed(2026)
    np.random.seed(2026)

    model = GemmModel().eval()
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

    reader = RandomDataReader("input", INPUT_SHAPE, num_batches=10, seed=2026)
    quantize_static(
        model_input=fp32_model_path,
        model_output=model_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Gemm"],
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )

    check_symmetric_zero_points(model_path)

    test_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
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
    print(f"Case: {MODEL_NAME}, input shape: {INPUT_SHAPE}, out_features: {OUT_FEATURES}")

if __name__ == "__main__":
    main()
