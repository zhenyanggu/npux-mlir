#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

SUPPORTED_MLIR_OPS = {
    "Conv": "Conv",
    "Relu": "Relu",
    "Add": "Add",
    "MaxPoolSingleOut": "MaxPool",
    "ReduceMeanV13": "ReduceMean",
    "Flatten": "Flatten",
    "Gemm": "Gemm",
}

SINGLE_LAYER_MODEL_PY_TEMPLATE = """\
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

MODEL_NAME = "__MODEL_NAME__"
OP_TYPE = "__OP_TYPE__"
INPUT_SHAPE = __INPUT_SHAPE__
OUTPUT_SHAPE = __OUTPUT_SHAPE__
KERNEL = __KERNEL__
STRIDE = __STRIDE__
PADS = __PADS__
DILATIONS = __DILATIONS__
GROUP = __GROUP__
AXIS = __AXIS__
AXES = __AXES__
KEEPDIMS = __KEEPDIMS__


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
"""


RESIDUAL_BLOCK_MODEL_PY_TEMPLATE = """\
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

MODEL_NAME = "__MODEL_NAME__"
INPUT_SHAPE = __INPUT_SHAPE__
OUTPUT_SHAPE = __OUTPUT_SHAPE__
MID_CHANNELS = __MID_CHANNELS__
OUT_CHANNELS = __OUT_CHANNELS__
STRIDE = __STRIDE__
USE_PROJECTION = __USE_PROJECTION__


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
"""


MAIN_CPP_TEMPLATE = """\
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "OnnxMlirRuntime.h"

extern "C" OMTensorList *run_main_graph(OMTensorList *);

namespace {
namespace fs = std::filesystem;

struct VerificationSummary {
  float maxAbsError;
  double mse;
  int64_t errorCount;
  int64_t totalCount;
  bool passed;
};

struct ErrorItem {
  int64_t index;
  float actual;
  float golden;
  float absDiff;
  int64_t n;
  int64_t c;
  int64_t h;
  int64_t w;
};

bool hasValidNchwShape(const std::vector<int64_t> &shape) {
  if (shape.size() != 4)
    return false;
  return shape[0] > 0 && shape[1] > 0 && shape[2] > 0 && shape[3] > 0;
}

bool endsWith(const std::string &value, const std::string &suffix) {
  if (value.size() < suffix.size())
    return false;
  return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string modelPrefixFromExecutable(const fs::path &exePath) {
  std::string name = exePath.filename().string();
  const std::string suffix = "_zcu102";
  if (endsWith(name, suffix))
    name.resize(name.size() - suffix.size());
  return name.empty() ? "model" : name;
}

std::string resolveDataFile(const fs::path &exeDir, const std::string &modelPrefix,
    const std::string &logicalName) {
  const fs::path prefixed = exeDir / (modelPrefix + "_" + logicalName);
  if (fs::exists(prefixed))
    return prefixed.string();
  const fs::path legacy = exeDir / logicalName;
  if (fs::exists(legacy))
    return legacy.string();
  return prefixed.string();
}

std::vector<float> loadBinaryFloatFile(const std::string &filename, int64_t expectedElements) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: cannot open file: " << filename << std::endl;
    std::exit(1);
  }
  file.seekg(0, std::ios::end);
  const std::streamsize bytes = file.tellg();
  file.seekg(0, std::ios::beg);
  const auto expectedBytes = static_cast<std::streamsize>(expectedElements * sizeof(float));
  if (bytes != expectedBytes) {
    std::cerr << "Error: file size mismatch for " << filename << ", expected: "
              << expectedBytes << ", got: " << bytes << std::endl;
    std::exit(1);
  }
  std::vector<float> data(expectedElements);
  if (!file.read(reinterpret_cast<char *>(data.data()), bytes)) {
    std::cerr << "Error: failed to read file: " << filename << std::endl;
    std::exit(1);
  }
  return data;
}

void printErrorDetails(const std::vector<ErrorItem> &errors, int64_t count, int64_t topK = 10) {
  std::cout << "\\n=== Verification Report ===" << std::endl;
  std::cout << std::setw(12) << "index" << std::setw(8) << "n" << std::setw(8) << "c"
            << std::setw(8) << "h" << std::setw(8) << "w" << std::setw(16) << "actual"
            << std::setw(16) << "golden" << std::setw(16) << "abs diff" << std::endl;

  if (count <= 128) {
    for (const auto &item : errors) {
      std::cout << std::setw(12) << item.index << std::setw(8) << item.n << std::setw(8)
                << item.c << std::setw(8) << item.h << std::setw(8) << item.w
                << std::setw(16) << item.actual << std::setw(16) << item.golden
                << std::setw(16) << item.absDiff << std::endl;
    }
    std::cout << "Displayed all " << count << " elements." << std::endl;
    return;
  }

  const int64_t k = std::min<int64_t>(topK, count);
  std::vector<ErrorItem> sorted = errors;
  std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(),
      [](const ErrorItem &a, const ErrorItem &b) {
        if (a.absDiff == b.absDiff)
          return a.index < b.index;
        return a.absDiff > b.absDiff;
      });

  std::cout << "Top-" << k << " largest absolute errors:" << std::endl;
  for (int64_t i = 0; i < k; ++i) {
    const auto &item = sorted[static_cast<size_t>(i)];
    std::cout << std::setw(12) << item.index << std::setw(8) << item.n << std::setw(8)
              << item.c << std::setw(8) << item.h << std::setw(8) << item.w
              << std::setw(16) << item.actual << std::setw(16) << item.golden
              << std::setw(16) << item.absDiff << std::endl;
  }
}

void printNchwErrorBreakdown(const std::vector<ErrorItem> &errors,
    const std::vector<int64_t> &shape, float threshold, int64_t topK = 10) {
  if (!hasValidNchwShape(shape)) {
    std::cout << "\\n[NCHW Error Breakdown] skipped (output rank != 4)." << std::endl;
    return;
  }

  const int64_t n = shape[0];
  const int64_t c = shape[1];
  const int64_t h = shape[2];
  const int64_t w = shape[3];
  const int64_t hwSize = h * w;

  std::vector<int64_t> channelErrCount(static_cast<size_t>(c), 0);
  std::vector<double> channelAbsSum(static_cast<size_t>(c), 0.0);
  std::vector<float> channelMaxAbs(static_cast<size_t>(c), 0.0f);
  std::vector<int64_t> spatialErrCount(static_cast<size_t>(hwSize), 0);
  std::vector<double> spatialAbsSum(static_cast<size_t>(hwSize), 0.0);

  int64_t thresholdErrCount = 0;
  for (const auto &item : errors) {
    if (item.absDiff <= threshold)
      continue;
    if (item.c < 0 || item.c >= c || item.h < 0 || item.h >= h || item.w < 0 || item.w >= w)
      continue;
    ++thresholdErrCount;
    const size_t cIdx = static_cast<size_t>(item.c);
    const size_t hwIdx = static_cast<size_t>(item.h * w + item.w);
    ++channelErrCount[cIdx];
    channelAbsSum[cIdx] += item.absDiff;
    channelMaxAbs[cIdx] = std::max(channelMaxAbs[cIdx], item.absDiff);
    ++spatialErrCount[hwIdx];
    spatialAbsSum[hwIdx] += item.absDiff;
  }

  std::cout << "\\n=== NCHW Error Breakdown (abs diff > " << threshold << ") ==="
            << std::endl;
  if (thresholdErrCount == 0) {
    std::cout << "No error points above threshold." << std::endl;
    return;
  }

  std::vector<int64_t> channelOrder(static_cast<size_t>(c));
  std::iota(channelOrder.begin(), channelOrder.end(), 0);
  const int64_t topChannelK = std::min<int64_t>(topK, c);
  std::partial_sort(channelOrder.begin(), channelOrder.begin() + topChannelK,
      channelOrder.end(), [&](int64_t a, int64_t b) {
        const int64_t ca = channelErrCount[static_cast<size_t>(a)];
        const int64_t cb = channelErrCount[static_cast<size_t>(b)];
        if (ca == cb)
          return a < b;
        return ca > cb;
      });

  std::cout << "Top-" << topChannelK << " channels by error count:" << std::endl;
  std::cout << std::setw(10) << "channel" << std::setw(14) << "err_count"
            << std::setw(14) << "err_ratio%" << std::setw(16) << "mean_abs"
            << std::setw(16) << "max_abs" << std::endl;
  for (int64_t i = 0; i < topChannelK; ++i) {
    const int64_t ch = channelOrder[static_cast<size_t>(i)];
    const int64_t cnt = channelErrCount[static_cast<size_t>(ch)];
    const double ratio = (thresholdErrCount > 0)
                             ? (100.0 * static_cast<double>(cnt) /
                                   static_cast<double>(thresholdErrCount))
                             : 0.0;
    const double meanAbs = (cnt > 0)
                               ? (channelAbsSum[static_cast<size_t>(ch)] /
                                     static_cast<double>(cnt))
                               : 0.0;
    std::cout << std::setw(10) << ch << std::setw(14) << cnt << std::setw(14)
              << ratio << std::setw(16) << meanAbs << std::setw(16)
              << channelMaxAbs[static_cast<size_t>(ch)] << std::endl;
  }

  auto printModPattern = [&](int modBase) {
    if (c < modBase)
      return;
    std::vector<int64_t> buckets(static_cast<size_t>(modBase), 0);
    for (int64_t ch = 0; ch < c; ++ch)
      buckets[static_cast<size_t>(ch % modBase)] +=
          channelErrCount[static_cast<size_t>(ch)];
    std::cout << "Channel error modulo " << modBase << ":";
    for (int i = 0; i < modBase; ++i)
      std::cout << " [" << i << "]=" << buckets[static_cast<size_t>(i)];
    std::cout << std::endl;
  };
  printModPattern(2);
  printModPattern(3);
  printModPattern(4);
  printModPattern(8);

  std::vector<int64_t> spatialOrder(static_cast<size_t>(hwSize));
  std::iota(spatialOrder.begin(), spatialOrder.end(), 0);
  const int64_t topSpatialK = std::min<int64_t>(topK, hwSize);
  std::partial_sort(spatialOrder.begin(), spatialOrder.begin() + topSpatialK,
      spatialOrder.end(), [&](int64_t a, int64_t b) {
        const int64_t ca = spatialErrCount[static_cast<size_t>(a)];
        const int64_t cb = spatialErrCount[static_cast<size_t>(b)];
        if (ca == cb)
          return a < b;
        return ca > cb;
      });

  std::cout << "Top-" << topSpatialK << " spatial positions by error count:"
            << std::endl;
  std::cout << std::setw(8) << "h" << std::setw(8) << "w" << std::setw(14)
            << "err_count" << std::setw(16) << "mean_abs" << std::setw(16)
            << "affected_ch%" << std::endl;
  for (int64_t i = 0; i < topSpatialK; ++i) {
    const int64_t hw = spatialOrder[static_cast<size_t>(i)];
    const int64_t hh = hw / w;
    const int64_t ww = hw % w;
    const int64_t cnt = spatialErrCount[static_cast<size_t>(hw)];
    const double meanAbs = (cnt > 0)
                               ? (spatialAbsSum[static_cast<size_t>(hw)] /
                                     static_cast<double>(cnt))
                               : 0.0;
    const double chRatio = (n * c > 0)
                               ? (100.0 * static_cast<double>(cnt) /
                                     static_cast<double>(n * c))
                               : 0.0;
    std::cout << std::setw(8) << hh << std::setw(8) << ww << std::setw(14)
              << cnt << std::setw(16) << meanAbs << std::setw(16) << chRatio
              << std::endl;
  }
}

VerificationSummary compareOutputs(const float *actual,
    const std::vector<float> &golden, int64_t count,
    const std::vector<int64_t> &outputShape, float threshold = 0.1f) {
  float maxAbsError = 0.0f;
  double mse = 0.0;
  int64_t errorCount = 0;
  std::vector<ErrorItem> errors;
  errors.reserve(static_cast<size_t>(count));
  const bool hasNchw = hasValidNchwShape(outputShape);
  const int64_t cVal = hasNchw ? outputShape[1] : -1;
  const int64_t hVal = hasNchw ? outputShape[2] : -1;
  const int64_t wVal = hasNchw ? outputShape[3] : -1;
  const int64_t chw = hasNchw ? (cVal * hVal * wVal) : -1;
  const int64_t hw = hasNchw ? (hVal * wVal) : -1;

  for (int64_t i = 0; i < count; ++i) {
    const float diff = std::abs(actual[i] - golden[static_cast<size_t>(i)]);
    maxAbsError = std::max(maxAbsError, diff);
    mse += static_cast<double>(diff) * static_cast<double>(diff);
    if (diff > threshold)
      ++errorCount;
    int64_t nIdx = -1;
    int64_t cIdx = -1;
    int64_t hIdx = -1;
    int64_t wIdx = -1;
    if (hasNchw && chw > 0 && hw > 0) {
      nIdx = i / chw;
      const int64_t rem0 = i % chw;
      cIdx = rem0 / hw;
      const int64_t rem1 = rem0 % hw;
      hIdx = rem1 / wVal;
      wIdx = rem1 % wVal;
    }
    errors.push_back(
        {i, actual[i], golden[static_cast<size_t>(i)], diff, nIdx, cIdx, hIdx, wIdx});
  }
  mse /= static_cast<double>(count);

  printErrorDetails(errors, count);
  printNchwErrorBreakdown(errors, outputShape, threshold);
  std::cout << "Max Absolute Error: " << maxAbsError << std::endl;
  std::cout << "Mean Squared Error: " << mse << std::endl;
  std::cout << "Error Count (错误点数量/总点数量): " << errorCount << "/" << count
            << std::endl;
  const bool passed = (errorCount == 0);
  std::cout << "Threshold Result: " << (passed ? "PASS" : "FAIL") << std::endl;
  return {maxAbsError, mse, errorCount, count, passed};
}

void printFinalSummary(const VerificationSummary &summary) {
  std::cout << "@@MODEL_TEST_RESULT@@ errors=" << summary.errorCount << "/"
            << summary.totalCount << " status="
            << (summary.passed ? "PASS" : "FAIL") << std::endl;
}

void verifyExpectedShape(const int64_t *shape, int64_t rank, const std::vector<int64_t> &expected) {
  bool ok = (rank == static_cast<int64_t>(expected.size()));
  if (ok) {
    for (int64_t i = 0; i < rank; ++i) {
      if (shape[i] != expected[static_cast<size_t>(i)]) {
        ok = false;
        break;
      }
    }
  }
  std::cout << "[ResNet50 Layer Case Shape Check] " << (ok ? "PASS" : "WARNING") << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  const std::vector<int64_t> inputShape = {__INPUT_SHAPE__};
  const std::vector<int64_t> expectedOutputShape = {__OUTPUT_SHAPE__};

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
  OMTensor *inputs[] = {inputTensor};
  OMTensorList *inputList = omTensorListCreate(inputs, 1);
  OMTensorList *outputList = run_main_graph(inputList);
  if (!outputList) {
    std::cerr << "Error: inference failed" << std::endl;
    omTensorListDestroy(inputList);
    return 1;
  }

  OMTensor *outputTensor = omTensorListGetOmtByIndex(outputList, 0);
  float *outputData = reinterpret_cast<float *>(omTensorGetDataPtr(outputTensor));
  const int64_t *outputShape = omTensorGetShape(outputTensor);
  const int64_t outputRank = omTensorGetRank(outputTensor);

  int64_t outputElements = 1;
  for (int64_t i = 0; i < outputRank; ++i)
    outputElements *= outputShape[i];

  std::vector<float> golden = loadBinaryFloatFile(goldenFile, outputElements);
  std::vector<int64_t> outputShapeVec(
      outputShape, outputShape + static_cast<size_t>(outputRank));
  const VerificationSummary summary =
      compareOutputs(outputData, golden, outputElements, outputShapeVec);
  verifyExpectedShape(outputShape, outputRank, expectedOutputShape);

  omTensorListDestroy(inputList);
  omTensorListDestroy(outputList);
  printFinalSummary(summary);
  return 0;
}
"""


def parse_shape(tensor_ty: str) -> List[int]:
    m = re.match(r"tensor<([^>]+)>", tensor_ty.strip())
    if not m:
        raise ValueError(f"Unsupported tensor type format: {tensor_ty}")
    inner = m.group(1)
    parts = inner.split("x")
    if len(parts) < 2:
        raise ValueError(f"Unsupported tensor type: {tensor_ty}")

    dims: List[int] = []
    for d in parts[:-1]:
        token = d.strip()
        if re.fullmatch(r"-?\d+", token):
            dims.append(int(token))
        else:
            dims.append(1)
    return dims


def parse_int_list_attr(attrs: str, key: str, default: List[int]) -> List[int]:
    m = re.search(rf"{key}\s*=\s*\[([^\]]*)\]", attrs)
    if not m:
        return list(default)
    values = [v.strip() for v in m.group(1).split(",") if v.strip()]
    out: List[int] = []
    for v in values:
        if re.fullmatch(r"-?\d+", v):
            out.append(int(v))
    return out if out else list(default)


def parse_int_attr(attrs: str, key: str, default: int) -> int:
    m = re.search(rf"{key}\s*=\s*(-?\d+)", attrs)
    if not m:
        return default
    return int(m.group(1))


def sanitize_name(text: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_").lower()
    return s or "unnamed"


def py_tuple(values: List[int]) -> str:
    if len(values) == 1:
        return f"({values[0]},)"
    return "(" + ", ".join(str(v) for v in values) + ")"


def csv(values: List[int]) -> str:
    return ", ".join(str(v) for v in values)


def shape_token(shape: List[int]) -> str:
    if len(shape) == 4:
        return f"{shape[1]}x{shape[2]}x{shape[3]}"
    if len(shape) == 2:
        return f"{shape[1]}"
    return "x".join(str(v) for v in shape)


def conv_param_token(case: Dict) -> str:
    k = case.get("kernel", [1, 1])
    s = case.get("stride", [1, 1])
    p = case.get("pads", [0, 0, 0, 0])
    d = case.get("dilations", [1, 1])
    g = case.get("group", 1)
    token = f"k{k[0]}x{k[1]}_s{s[0]}x{s[1]}_p{p[0]}x{p[1]}"
    if d != [1, 1]:
        token += f"_d{d[0]}x{d[1]}"
    if g != 1:
        token += f"_g{g}"
    return token


def pool_param_token(case: Dict) -> str:
    k = case.get("kernel", [2, 2])
    s = case.get("stride", [2, 2])
    p = case.get("pads", [0, 0, 0, 0])
    return f"k{k[0]}x{k[1]}_s{s[0]}x{s[1]}_p{p[0]}x{p[1]}"


def case_signature(case: Dict) -> tuple:
    sig = [case["op"], tuple(case["input_shape"]), tuple(case["output_shape"])]
    if case["op"] in {"Conv", "ConvRelu"}:
        sig.extend(
            [
                tuple(case.get("kernel", [])),
                tuple(case.get("stride", [])),
                tuple(case.get("pads", [])),
                tuple(case.get("dilations", [1, 1])),
                int(case.get("group", 1)),
            ]
        )
    elif case["op"] == "MaxPool":
        sig.extend(
            [
                tuple(case.get("kernel", [])),
                tuple(case.get("stride", [])),
                tuple(case.get("pads", [])),
            ]
        )
    elif case["op"] == "ReduceMean":
        sig.extend([tuple(case.get("axes", [2, 3])), int(case.get("keepdims", 1))])
    elif case["op"] == "Flatten":
        sig.append(int(case.get("axis", 1)))
    return tuple(sig)


def build_layer_case_key(case: Dict) -> str:
    op = case["op"]
    inp = shape_token(case["input_shape"])
    out = shape_token(case["output_shape"])
    if op in {"Conv", "ConvRelu"}:
        base = f"{op.lower()}_{inp}_to_{out}_{conv_param_token(case)}"
    elif op == "Add":
        base = f"add_{inp}"
    elif op == "MaxPool":
        base = f"maxpool_{inp}_to_{out}_{pool_param_token(case)}"
    elif op == "ReduceMean":
        axes = "_".join(str(v) for v in case.get("axes", [2, 3]))
        base = f"reducemean_{inp}_to_{out}_axes_{axes}"
    elif op == "Flatten":
        base = f"flatten_{inp}_to_{out}_axis_{case.get('axis', 1)}"
    elif op == "Gemm":
        base = f"gemm_{inp}_to_{out}"
    else:
        base = f"{op.lower()}_{inp}_to_{out}"
    return sanitize_name(base)


def extract_ops_from_mlir(mlir_path: Path) -> List[Dict]:
    content = mlir_path.read_text(encoding="utf-8")
    op_names = "|".join(re.escape(x) for x in SUPPORTED_MLIR_OPS.keys())
    pattern = re.compile(
        rf'^\s*%\d+\s*=\s*"onnx\.({op_names})"\([^\)]*\)\s*(\{{[^\}}]*\}})?\s*:\s*\(([^\)]*)\)\s*->\s*(tensor<[^>]+>)',
        re.MULTILINE,
    )

    ops: List[Dict] = []
    for i, m in enumerate(pattern.finditer(content), start=1):
        mlir_op = m.group(1)
        attrs = m.group(2) or ""
        input_types = m.group(3)
        output_type = m.group(4)

        node_name_m = re.search(r'onnx_node_name\s*=\s*"([^"]+)"', attrs)
        node_name = node_name_m.group(1) if node_name_m else f"{mlir_op}_{i:03d}"
        tensor_types = re.findall(r"tensor<[^>]+>", input_types)
        if not tensor_types:
            continue

        op = SUPPORTED_MLIR_OPS[mlir_op]
        item: Dict = {
            "order": i,
            "node_name": node_name,
            "mlir_op": mlir_op,
            "op": op,
            "input_shape": parse_shape(tensor_types[0]),
            "output_shape": parse_shape(output_type),
        }
        if op == "Conv":
            item["kernel"] = parse_int_list_attr(attrs, "kernel_shape", [3, 3])
            item["stride"] = parse_int_list_attr(attrs, "strides", [1, 1])
            item["pads"] = parse_int_list_attr(attrs, "pads", [0, 0, 0, 0])
            item["dilations"] = parse_int_list_attr(attrs, "dilations", [1, 1])
            item["group"] = parse_int_attr(attrs, "group", 1)
        elif op == "MaxPool":
            item["kernel"] = parse_int_list_attr(attrs, "kernel_shape", [2, 2])
            item["stride"] = parse_int_list_attr(attrs, "strides", [2, 2])
            item["pads"] = parse_int_list_attr(attrs, "pads", [0, 0, 0, 0])
        elif op == "ReduceMean":
            item["axes"] = parse_int_list_attr(attrs, "axes", [2, 3])
            item["keepdims"] = parse_int_attr(attrs, "keepdims", 1)
        elif op == "Flatten":
            item["axis"] = parse_int_attr(attrs, "axis", 1)
        ops.append(item)
    return ops


def build_dedup_layer_cases(ops: List[Dict]) -> List[Dict]:
    dedup: Dict[tuple, Dict] = {}
    i = 0
    while i < len(ops):
        cur = ops[i]
        candidate: Dict

        if cur["op"] == "Relu":
            i += 1
            continue

        if cur["op"] == "Conv" and i + 1 < len(ops):
            nxt = ops[i + 1]
            if nxt["op"] == "Relu" and tuple(nxt["input_shape"]) == tuple(cur["output_shape"]):
                candidate = dict(cur)
                candidate["op"] = "ConvRelu"
                candidate["output_shape"] = list(nxt["output_shape"])
                candidate["node_name"] = f"{cur['node_name']}__{nxt['node_name']}"
                i += 2
            else:
                candidate = dict(cur)
                i += 1
        else:
            candidate = dict(cur)
            i += 1

        sig = case_signature(candidate)
        if sig in dedup:
            continue
        candidate["kind"] = "layer"
        candidate["case_key"] = build_layer_case_key(candidate)
        dedup[sig] = candidate

    cases = list(dedup.values())
    used_names = set()
    for idx, case in enumerate(cases, start=1):
        base_name = f"resnet50_layer_{idx:03d}_{case['case_key']}"
        name = base_name
        suffix = 2
        while name in used_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(name)
        case["index"] = idx
        case["name"] = name
    return cases


def build_residual_block_cases() -> List[Dict]:
    configs = [
        ("stage1_proj", [1, 64, 56, 56], [1, 256, 56, 56], 64, 256, 1, 1),
        ("stage1_identity", [1, 256, 56, 56], [1, 256, 56, 56], 64, 256, 1, 0),
        ("stage2_proj", [1, 256, 56, 56], [1, 512, 28, 28], 128, 512, 2, 1),
        ("stage2_identity", [1, 512, 28, 28], [1, 512, 28, 28], 128, 512, 1, 0),
        ("stage3_proj", [1, 512, 28, 28], [1, 1024, 14, 14], 256, 1024, 2, 1),
        ("stage3_identity", [1, 1024, 14, 14], [1, 1024, 14, 14], 256, 1024, 1, 0),
        ("stage4_proj", [1, 1024, 14, 14], [1, 2048, 7, 7], 512, 2048, 2, 1),
        ("stage4_identity", [1, 2048, 7, 7], [1, 2048, 7, 7], 512, 2048, 1, 0),
    ]
    cases: List[Dict] = []
    for idx, (tag, in_shape, out_shape, mid_c, out_c, stride, use_proj) in enumerate(configs, start=1):
        cases.append(
            {
                "kind": "block",
                "index": idx,
                "name": f"resnet50_block_{idx:02d}_{tag}",
                "case_key": tag,
                "input_shape": in_shape,
                "output_shape": out_shape,
                "mid_channels": mid_c,
                "out_channels": out_c,
                "stride": stride,
                "use_projection": use_proj,
            }
        )
    return cases


def render_model_py(case: Dict) -> str:
    if case["kind"] == "block":
        rendered = RESIDUAL_BLOCK_MODEL_PY_TEMPLATE
        rendered = rendered.replace("__MODEL_NAME__", case["name"])
        rendered = rendered.replace("__INPUT_SHAPE__", py_tuple(case["input_shape"]))
        rendered = rendered.replace("__OUTPUT_SHAPE__", py_tuple(case["output_shape"]))
        rendered = rendered.replace("__MID_CHANNELS__", str(case["mid_channels"]))
        rendered = rendered.replace("__OUT_CHANNELS__", str(case["out_channels"]))
        rendered = rendered.replace("__STRIDE__", str(case["stride"]))
        rendered = rendered.replace("__USE_PROJECTION__", str(case["use_projection"]))
        return rendered

    rendered = SINGLE_LAYER_MODEL_PY_TEMPLATE
    rendered = rendered.replace("__MODEL_NAME__", case["name"])
    rendered = rendered.replace("__OP_TYPE__", case["op"])
    rendered = rendered.replace("__INPUT_SHAPE__", py_tuple(case["input_shape"]))
    rendered = rendered.replace("__OUTPUT_SHAPE__", py_tuple(case["output_shape"]))
    rendered = rendered.replace("__KERNEL__", py_tuple(case.get("kernel", [])))
    rendered = rendered.replace("__STRIDE__", py_tuple(case.get("stride", [])))
    rendered = rendered.replace("__PADS__", py_tuple(case.get("pads", [])))
    rendered = rendered.replace("__DILATIONS__", py_tuple(case.get("dilations", [1, 1])))
    rendered = rendered.replace("__GROUP__", str(case.get("group", 1)))
    rendered = rendered.replace("__AXIS__", str(case.get("axis", 1)))
    rendered = rendered.replace("__AXES__", py_tuple(case.get("axes", [2, 3])))
    rendered = rendered.replace("__KEEPDIMS__", str(case.get("keepdims", 1)))
    return rendered


def render_main_cpp(case: Dict) -> str:
    rendered = MAIN_CPP_TEMPLATE
    rendered = rendered.replace("__INPUT_SHAPE__", csv(case["input_shape"]))
    rendered = rendered.replace("__OUTPUT_SHAPE__", csv(case["output_shape"]))
    return rendered


def write_layer_readme(cases: List[Dict]) -> None:
    counts = Counter(c["op"] for c in cases)
    stats = "\n".join(f"- {k}: {counts[k]}" for k in sorted(counts.keys()))
    readme = MODELS_DIR / "resnet50_layer_cases.README.md"
    text = (
        "# ResNet50 Layer Cases\n\n"
        "These testcases are generated from ResNet50 ONNX-MLIR and deduplicated by\n"
        "(op + shape + attributes). Standalone Relu cases are removed.\n"
        "When a Conv is directly followed by Relu, it is generated as one `ConvRelu` case.\n"
        "They are disabled by default to avoid affecting `make all`.\n\n"
        "Layer op counts:\n"
        f"{stats}\n\n"
        "Enable one case:\n"
        "- `rm -f models/<case_name>/.disabled`\n"
        "- `make <case_name>`\n\n"
        "Or run directly by model name even if disabled:\n"
        "- `make run_one MODEL=<case_name>`\n"
    )
    readme.write_text(text, encoding="utf-8")


def write_block_readme(cases: List[Dict]) -> None:
    block_list = "\n".join(f"- {c['name']}" for c in cases)
    readme = MODELS_DIR / "resnet50_block_cases.README.md"
    text = (
        "# ResNet50 Residual Block Cases\n\n"
        "These testcases cover bottleneck residual blocks (projection + identity variants).\n"
        "They are disabled by default to avoid affecting `make all`.\n\n"
        "Residual block cases:\n"
        f"{block_list}\n\n"
        "Enable one case:\n"
        "- `rm -f models/<case_name>/.disabled`\n"
        "- `make <case_name>`\n\n"
        "Or run directly by model name even if disabled:\n"
        "- `make run_one MODEL=<case_name>`\n"
    )
    readme.write_text(text, encoding="utf-8")


def generate_case_dirs(layer_cases: List[Dict], block_cases: List[Dict]) -> None:
    expected_layer = {c["name"] for c in layer_cases}
    expected_block = {c["name"] for c in block_cases}
    for stale in sorted(MODELS_DIR.glob("resnet50_layer_*/")):
        if stale.name not in expected_layer:
            shutil.rmtree(stale)
    for stale in sorted(MODELS_DIR.glob("resnet50_block_*/")):
        if stale.name not in expected_block:
            shutil.rmtree(stale)

    for case in layer_cases + block_cases:
        model_dir = MODELS_DIR / case["name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.py").write_text(render_model_py(case), encoding="utf-8")
        (model_dir / "main.cpp").write_text(render_main_cpp(case), encoding="utf-8")
        (model_dir / ".disabled").write_text("", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ResNet50 per-layer testcase directories")
    parser.add_argument(
        "--mlir",
        default=str(ROOT / "build" / "resnet50" / "model.onnx.mlir"),
        help="Path to ONNX-MLIR file for resnet50 (default: model_test/build/resnet50/model.onnx.mlir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlir_path = Path(args.mlir).resolve()
    if not mlir_path.exists():
        raise FileNotFoundError(
            f"MLIR file not found: {mlir_path}\n"
            "Please generate it first, e.g. run: cd model_test && make resnet50.mlir"
        )

    ops = extract_ops_from_mlir(mlir_path)
    layer_cases = build_dedup_layer_cases(ops)
    block_cases = build_residual_block_cases()
    if not layer_cases:
        raise RuntimeError(f"No supported layer ops found in: {mlir_path}")

    generate_case_dirs(layer_cases, block_cases)
    write_layer_readme(layer_cases)
    write_block_readme(block_cases)

    counts = Counter(c["op"] for c in layer_cases)
    print(f"Generated {len(layer_cases)} deduplicated ResNet50 layer case directories under {MODELS_DIR}")
    for op in sorted(counts.keys()):
        print(f"  {op}: {counts[op]}")
    print(f"Generated {len(block_cases)} ResNet50 residual block case directories under {MODELS_DIR}")


if __name__ == "__main__":
    main()
