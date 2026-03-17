#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


CASES = [
    {
        "name": "vgg16_layer_conv_3x224_to_64x224",
        "op": "Conv",
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, 64, 224, 224],
        "kernel": [3, 3],
        "stride": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    {
        "name": "vgg16_layer_conv_3x112_to_64x112",
        "op": "Conv",
        "input_shape": [1, 3, 112, 112],
        "output_shape": [1, 64, 112, 112],
        "kernel": [3, 3],
        "stride": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    {
        "name": "vgg16_layer_conv_3x56_to_64x56",
        "op": "Conv",
        "input_shape": [1, 3, 56, 56],
        "output_shape": [1, 64, 56, 56],
        "kernel": [3, 3],
        "stride": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    {
        "name": "vgg16_layer_conv_3x224_to_32x224",
        "op": "Conv",
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, 32, 224, 224],
        "kernel": [3, 3],
        "stride": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    {
        "name": "vgg16_layer_conv_32x224_to_64x224",
        "op": "Conv",
        "input_shape": [1, 32, 224, 224],
        "output_shape": [1, 64, 224, 224],
        "kernel": [3, 3],
        "stride": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    {
        "name": "vgg16_layer_relu_64x224",
        "op": "Relu",
        "input_shape": [1, 64, 224, 224],
        "output_shape": [1, 64, 224, 224],
    },
    {
        "name": "vgg16_layer_conv_64x224_to_64x224",
        "op": "Conv",
        "input_shape": [1, 64, 224, 224],
        "output_shape": [1, 64, 224, 224],
        "kernel": [3, 3],
        "stride": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    {
        "name": "vgg16_layer_maxpool_64x224_to_64x112",
        "op": "MaxPool",
        "input_shape": [1, 64, 224, 224],
        "output_shape": [1, 64, 112, 112],
        "kernel": [2, 2],
        "stride": [2, 2],
        "pads": [0, 0, 0, 0],
    },
    {
        "name": "vgg16_layer_conv_64x112_to_128x112",
        "op": "Conv",
        "input_shape": [1, 64, 112, 112],
        "output_shape": [1, 128, 112, 112],
        "kernel": [3, 3],
        "stride": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    {
        "name": "vgg16_layer_conv_64x56_to_128x56",
        "op": "Conv",
        "input_shape": [1, 64, 56, 56],
        "output_shape": [1, 128, 56, 56],
        "kernel": [3, 3],
        "stride": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    {
        "name": "vgg16_layer_relu_128x112",
        "op": "Relu",
        "input_shape": [1, 128, 112, 112],
        "output_shape": [1, 128, 112, 112],
    },
    {
        "name": "vgg16_layer_conv_128x112_to_128x112",
        "op": "Conv",
        "input_shape": [1, 128, 112, 112],
        "output_shape": [1, 128, 112, 112],
        "kernel": [3, 3],
        "stride": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    {
        "name": "vgg16_layer_maxpool_128x112_to_128x56",
        "op": "MaxPool",
        "input_shape": [1, 128, 112, 112],
        "output_shape": [1, 128, 56, 56],
        "kernel": [2, 2],
        "stride": [2, 2],
        "pads": [0, 0, 0, 0],
    },
    {
        "name": "vgg16_layer_conv_128x56_to_256x56",
        "op": "Conv",
        "input_shape": [1, 128, 56, 56],
        "output_shape": [1, 256, 56, 56],
        "kernel": [3, 3],
        "stride": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    {
        "name": "vgg16_layer_relu_256x56",
        "op": "Relu",
        "input_shape": [1, 256, 56, 56],
        "output_shape": [1, 256, 56, 56],
    },
    {
        "name": "vgg16_layer_conv_256x56_to_256x56",
        "op": "Conv",
        "input_shape": [1, 256, 56, 56],
        "output_shape": [1, 256, 56, 56],
        "kernel": [3, 3],
        "stride": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    {
        "name": "vgg16_layer_maxpool_256x56_to_256x28",
        "op": "MaxPool",
        "input_shape": [1, 256, 56, 56],
        "output_shape": [1, 256, 28, 28],
        "kernel": [2, 2],
        "stride": [2, 2],
        "pads": [0, 0, 0, 0],
    },
    {
        "name": "vgg16_layer_conv_256x28_to_512x28",
        "op": "Conv",
        "input_shape": [1, 256, 28, 28],
        "output_shape": [1, 512, 28, 28],
        "kernel": [3, 3],
        "stride": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    {
        "name": "vgg16_layer_relu_512x28",
        "op": "Relu",
        "input_shape": [1, 512, 28, 28],
        "output_shape": [1, 512, 28, 28],
    },
    {
        "name": "vgg16_layer_conv_512x28_to_512x28",
        "op": "Conv",
        "input_shape": [1, 512, 28, 28],
        "output_shape": [1, 512, 28, 28],
        "kernel": [3, 3],
        "stride": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    {
        "name": "vgg16_layer_maxpool_512x28_to_512x14",
        "op": "MaxPool",
        "input_shape": [1, 512, 28, 28],
        "output_shape": [1, 512, 14, 14],
        "kernel": [2, 2],
        "stride": [2, 2],
        "pads": [0, 0, 0, 0],
    },
    {
        "name": "vgg16_layer_conv_512x14_to_512x14",
        "op": "Conv",
        "input_shape": [1, 512, 14, 14],
        "output_shape": [1, 512, 14, 14],
        "kernel": [3, 3],
        "stride": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    {
        "name": "vgg16_layer_relu_512x14",
        "op": "Relu",
        "input_shape": [1, 512, 14, 14],
        "output_shape": [1, 512, 14, 14],
    },
    {
        "name": "vgg16_layer_maxpool_512x14_to_512x7",
        "op": "MaxPool",
        "input_shape": [1, 512, 14, 14],
        "output_shape": [1, 512, 7, 7],
        "kernel": [2, 2],
        "stride": [2, 2],
        "pads": [0, 0, 0, 0],
    },
    {
        "name": "vgg16_layer_flatten_512x7_to_25088",
        "op": "Flatten",
        "input_shape": [1, 512, 7, 7],
        "output_shape": [1, 25088],
        "axis": 1,
    },
    {
        "name": "vgg16_layer_gemm_25088_to_4096",
        "op": "Gemm",
        "input_shape": [1, 25088],
        "output_shape": [1, 4096],
    },
    {
        "name": "vgg16_layer_relu_4096",
        "op": "Relu",
        "input_shape": [1, 4096],
        "output_shape": [1, 4096],
    },
    {
        "name": "vgg16_layer_flatten_4096_to_4096",
        "op": "Flatten",
        "input_shape": [1, 4096],
        "output_shape": [1, 4096],
        "axis": 1,
    },
    {
        "name": "vgg16_layer_gemm_4096_to_4096",
        "op": "Gemm",
        "input_shape": [1, 4096],
        "output_shape": [1, 4096],
    },
    {
        "name": "vgg16_layer_gemm_4096_to_1000",
        "op": "Gemm",
        "input_shape": [1, 4096],
        "output_shape": [1, 1000],
    },
]

SUPPORTED_OPS = {"Conv", "MaxPool", "Gemm"}


MODEL_PY_TEMPLATE = """\
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

MODEL_NAME = "{model_name}"
OP_TYPE = "{op_type}"
INPUT_SHAPE = {input_shape}
OUTPUT_SHAPE = {output_shape}
KERNEL = {kernel}
STRIDE = {stride}
PADS = {pads}
AXIS = {axis}


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
            raise RuntimeError(f"Unsupported OP_TYPE: {{OP_TYPE}}")

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
            data.append({{self.input_name: sample}})
        return data

    def get_next(self):
        return next(self._iterator, None)


def ensure_layer_op_exists(model_path):
    model = onnx.load(model_path)
    op_types = {{node.op_type for node in model.graph.node}}
    expected_alias = {{
        "Conv": {{"Conv"}},
        "Relu": {{"Relu"}},
        "MaxPool": {{"MaxPool"}},
        "Flatten": {{"Flatten", "Reshape"}},
        "Gemm": {{"Gemm", "MatMul"}},
    }}
    required = expected_alias[OP_TYPE]
    if op_types.isdisjoint(required):
        raise RuntimeError(
            f"Expected layer op {{OP_TYPE}} not found; "
            f"accepted={{sorted(required)}}, present={{sorted(op_types)}}"
        )


def main():
    workdir = os.path.dirname(os.path.abspath(__file__))
    fp32_model_path = os.path.join(workdir, "model_fp32.onnx")
    model_path = os.path.join(workdir, "model.onnx")
    input_path = os.path.join(workdir, f"{{MODEL_NAME}}_input.bin")
    golden_path = os.path.join(workdir, f"{{MODEL_NAME}}_output_golden.bin")

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

    quantizable_ops = {{"Conv", "Relu", "MaxPool", "Gemm"}}
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
            extra_options={{"ActivationSymmetric": True, "WeightSymmetric": True}},
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
    output = sess.run(None, {{input_name: test_input}})[0]
    output.astype(np.float32).tofile(golden_path)

    print("Generated files:")
    print(f"  {{os.path.basename(model_path)}}")
    print(f"  {{os.path.basename(input_path)}}")
    print(f"  {{os.path.basename(golden_path)}}")
    print(f"Case: {{MODEL_NAME}}, op={{OP_TYPE}}")
    print(f"Input shape: {{INPUT_SHAPE}}")
    print(f"Output shape: {{tuple(output.shape)}}")


if __name__ == "__main__":
    main()
"""


MAIN_CPP_TEMPLATE = """\
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "OnnxMlirRuntime.h"

extern "C" OMTensorList *run_main_graph(OMTensorList *);

namespace {{
namespace fs = std::filesystem;

struct VerificationSummary {{
  float maxAbsError;
  double mse;
  int64_t errorCount;
  int64_t totalCount;
  bool passed;
}};

struct ErrorItem {{
  int64_t index;
  float actual;
  float golden;
  float absDiff;
}};

bool endsWith(const std::string &value, const std::string &suffix) {{
  if (value.size() < suffix.size())
    return false;
  return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}}

std::string modelPrefixFromExecutable(const fs::path &exePath) {{
  std::string name = exePath.filename().string();
  const std::string suffix = "_zcu102";
  if (endsWith(name, suffix))
    name.resize(name.size() - suffix.size());
  return name.empty() ? "model" : name;
}}

std::string resolveDataFile(const fs::path &exeDir, const std::string &modelPrefix,
    const std::string &logicalName) {{
  const fs::path prefixed = exeDir / (modelPrefix + "_" + logicalName);
  if (fs::exists(prefixed))
    return prefixed.string();
  const fs::path legacy = exeDir / logicalName;
  if (fs::exists(legacy))
    return legacy.string();
  return prefixed.string();
}}

std::vector<float> loadBinaryFloatFile(const std::string &filename, int64_t expectedElements) {{
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {{
    std::cerr << "Error: cannot open file: " << filename << std::endl;
    std::exit(1);
  }}
  file.seekg(0, std::ios::end);
  const std::streamsize bytes = file.tellg();
  file.seekg(0, std::ios::beg);
  const auto expectedBytes = static_cast<std::streamsize>(expectedElements * sizeof(float));
  if (bytes != expectedBytes) {{
    std::cerr << "Error: file size mismatch for " << filename << ", expected: "
              << expectedBytes << ", got: " << bytes << std::endl;
    std::exit(1);
  }}
  std::vector<float> data(expectedElements);
  if (!file.read(reinterpret_cast<char *>(data.data()), bytes)) {{
    std::cerr << "Error: failed to read file: " << filename << std::endl;
    std::exit(1);
  }}
  return data;
}}

void printErrorDetails(const std::vector<ErrorItem> &errors, int64_t count, int64_t topK = 10) {{
  std::cout << "\\n=== Verification Report ===" << std::endl;
  std::cout << std::setw(12) << "index" << std::setw(16) << "actual"
            << std::setw(16) << "golden" << std::setw(16) << "abs diff" << std::endl;

  if (count <= 128) {{
    for (const auto &item : errors) {{
      std::cout << std::setw(12) << item.index << std::setw(16) << item.actual
                << std::setw(16) << item.golden << std::setw(16) << item.absDiff << std::endl;
    }}
    std::cout << "Displayed all " << count << " elements." << std::endl;
    return;
  }}

  const int64_t k = std::min<int64_t>(topK, count);
  std::vector<ErrorItem> sorted = errors;
  std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(),
      [](const ErrorItem &a, const ErrorItem &b) {{
        if (a.absDiff == b.absDiff)
          return a.index < b.index;
        return a.absDiff > b.absDiff;
      }});

  std::cout << "Top-" << k << " largest absolute errors:" << std::endl;
  for (int64_t i = 0; i < k; ++i) {{
    const auto &item = sorted[static_cast<size_t>(i)];
    std::cout << std::setw(12) << item.index << std::setw(16) << item.actual
              << std::setw(16) << item.golden << std::setw(16) << item.absDiff << std::endl;
  }}
}}

VerificationSummary compareOutputs(const float *actual,
    const std::vector<float> &golden, int64_t count, float threshold = 0.1f) {{
  float maxAbsError = 0.0f;
  double mse = 0.0;
  int64_t errorCount = 0;
  std::vector<ErrorItem> errors;
  errors.reserve(static_cast<size_t>(count));

  for (int64_t i = 0; i < count; ++i) {{
    const float diff = std::abs(actual[i] - golden[static_cast<size_t>(i)]);
    maxAbsError = std::max(maxAbsError, diff);
    mse += static_cast<double>(diff) * static_cast<double>(diff);
    if (diff > threshold)
      ++errorCount;
    errors.push_back({{i, actual[i], golden[static_cast<size_t>(i)], diff}});
  }}
  mse /= static_cast<double>(count);

  printErrorDetails(errors, count);
  std::cout << "Max Absolute Error: " << maxAbsError << std::endl;
  std::cout << "Mean Squared Error: " << mse << std::endl;
  std::cout << "Error Count (错误点数量/总点数量): " << errorCount << "/" << count
            << std::endl;
  const bool passed = (errorCount == 0);
  std::cout << "Threshold Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return {{maxAbsError, mse, errorCount, count, passed}};
}}

void printFinalSummary(const VerificationSummary &summary) {{
  std::cout << "@@MODEL_TEST_RESULT@@ errors=" << summary.errorCount << "/"
            << summary.totalCount << " status="
            << (summary.passed ? "PASS" : "FAIL") << std::endl;
}}

void verifyExpectedShape(const int64_t *shape, int64_t rank, const std::vector<int64_t> &expected) {{
  bool ok = (rank == static_cast<int64_t>(expected.size()));
  if (ok) {{
    for (int64_t i = 0; i < rank; ++i) {{
      if (shape[i] != expected[static_cast<size_t>(i)]) {{
        ok = false;
        break;
      }}
    }}
  }}
  std::cout << "[VGG16 Layer Case Shape Check] " << (ok ? "PASS" : "WARNING") << std::endl;
}}

}} // namespace

int main(int argc, char **argv) {{
  const std::vector<int64_t> inputShape = {{{input_shape}}};
  const std::vector<int64_t> expectedOutputShape = {{{output_shape}}};

  int64_t inputElements = 1;
  for (int64_t d : inputShape)
    inputElements *= d;

  const fs::path exePath = (argc > 0) ? fs::path(argv[0]) : fs::path();
  const fs::path exeDir = exePath.has_parent_path() ? exePath.parent_path() : fs::path(".");
  const std::string modelPrefix = modelPrefixFromExecutable(exePath);

  const std::string inputFile = resolveDataFile(exeDir, modelPrefix, "input.bin");
  const std::string goldenFile = resolveDataFile(exeDir, modelPrefix, "output_golden.bin");

  std::vector<float> inputData = loadBinaryFloatFile(inputFile, inputElements);
  OMTensor *inputTensor = omTensorCreate(
      inputData.data(), const_cast<int64_t *>(inputShape.data()),
      static_cast<int64_t>(inputShape.size()), ONNX_TYPE_FLOAT);
  OMTensor *inputs[] = {{inputTensor}};
  OMTensorList *inputList = omTensorListCreate(inputs, 1);
  OMTensorList *outputList = run_main_graph(inputList);
  if (!outputList) {{
    std::cerr << "Error: inference failed" << std::endl;
    omTensorListDestroy(inputList);
    return 1;
  }}

  OMTensor *outputTensor = omTensorListGetOmtByIndex(outputList, 0);
  float *outputData = reinterpret_cast<float *>(omTensorGetDataPtr(outputTensor));
  const int64_t *outputShape = omTensorGetShape(outputTensor);
  const int64_t outputRank = omTensorGetRank(outputTensor);

  int64_t outputElements = 1;
  for (int64_t i = 0; i < outputRank; ++i)
    outputElements *= outputShape[i];

  std::vector<float> golden = loadBinaryFloatFile(goldenFile, outputElements);
  const VerificationSummary summary = compareOutputs(outputData, golden, outputElements);
  verifyExpectedShape(outputShape, outputRank, expectedOutputShape);

  omTensorListDestroy(inputList);
  omTensorListDestroy(outputList);
  printFinalSummary(summary);
  return 0;
}}
"""


def to_csv(values: list[int]) -> str:
    return ", ".join(str(v) for v in values)


def render_model_py(case: dict) -> str:
    return MODEL_PY_TEMPLATE.format(
        model_name=case["name"],
        op_type=case["op"],
        input_shape=tuple(case["input_shape"]),
        output_shape=tuple(case["output_shape"]),
        kernel=tuple(case.get("kernel", [])),
        stride=tuple(case.get("stride", [])),
        pads=tuple(case.get("pads", [])),
        axis=case.get("axis", 1),
    )


def render_main_cpp(case: dict) -> str:
    return MAIN_CPP_TEMPLATE.format(
        input_shape=to_csv(case["input_shape"]),
        output_shape=to_csv(case["output_shape"]),
    )


def main() -> None:
    active_cases = [case for case in CASES if case["op"] in SUPPORTED_OPS]

    for case in active_cases:
        model_dir = MODELS_DIR / case["name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.py").write_text(render_model_py(case), encoding="utf-8")
        (model_dir / "main.cpp").write_text(render_main_cpp(case), encoding="utf-8")
        (model_dir / ".disabled").write_text("", encoding="utf-8")

    readme = MODELS_DIR / "vgg16_layer_cases.README.md"
    readme.write_text(
        dedent(
            """\
            # VGG16 Layer Cases

            These testcases are generated from unique VGG16 layer signatures (op + shape),
            filtered by hardware-supported ops: Conv / MaxPool / Gemm.
            They are disabled by default to avoid affecting `make all`.

            Enable one case:
            - `rm -f models/<case_name>/.disabled`
            - `make <case_name>`

            Or run directly by model name even if disabled:
            - `make run_one MODEL=<case_name>`
            """
        ),
        encoding="utf-8",
    )

    print(f"Generated {len(active_cases)} VGG16 layer case directories under {MODELS_DIR}")


if __name__ == "__main__":
    main()
