#!/usr/bin/env python3
"""Measure BF16 shared-exponent block-float GEMV error on real Qwen activations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "scripts"))
from gemv_block_float_experiment import (  # noqa: E402
    block_float_bf16,
    ordered_bf16,
    reference_bf16,
    summarize,
    to_bf16,
)
from qwen25_onnx_lib import Qwen25PrefillInputs, parse_providers  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prompt", default="Explain what an NPU compiler does.")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--matmul-index", type=int, default=0)
    parser.add_argument("--output-channels", type=int, default=128)
    parser.add_argument("--act-widths", type=int, nargs="+", default=(16, 24, 27, 32))
    parser.add_argument("--weight-widths", type=int, nargs="+", default=(4, 8))
    parser.add_argument("--providers", default="CPUExecutionProvider")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def tensor_value_info(model: onnx.ModelProto, name: str) -> onnx.ValueInfoProto:
    inferred = onnx.shape_inference.infer_shapes(model)
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if value.name == name:
            return value
    return helper.make_tensor_value_info(name, TensorProto.BFLOAT16, None)


def select_matmul(model: onnx.ModelProto, index: int) -> Tuple[onnx.NodeProto, onnx.TensorProto]:
    initializers = {item.name: item for item in model.graph.initializer}
    candidates = [
        node for node in model.graph.node
        if node.op_type == "MatMul" and len(node.input) == 2 and node.input[1] in initializers
    ]
    if not candidates:
        raise ValueError("no MatMul node with an initializer weight was found")
    if index < 0 or index >= len(candidates):
        raise ValueError(f"--matmul-index must be in [0, {len(candidates) - 1}]")
    node = candidates[index]
    return node, initializers[node.input[1]]


def capture_activation(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    # Keep external-data references intact: the temporary probe is saved beside
    # the original model, so its .data files remain resolvable by ONNX Runtime.
    model = onnx.load(args.onnx_model, load_external_data=False)
    node, initializer = select_matmul(model, args.matmul_index)
    activation_name = node.input[0]
    model.graph.output.append(tensor_value_info(model, activation_name))
    probe_path = args.onnx_model.with_suffix(".block_float_probe.onnx")
    onnx.save(model, probe_path)
    try:
        inputs = Qwen25PrefillInputs(
            args.hf_model, seq_len=args.seq_len, trust_remote_code=args.trust_remote_code
        ).build(args.prompt)
        session = ort.InferenceSession(str(probe_path), providers=parse_providers(args.providers))
        outputs = session.run(None, inputs.feeds)
        activation = to_bf16(outputs[-1])
        # The fixed-shape prefill model pads to seq_len; only real prompt
        # tokens represent Qwen activation data for this experiment.
        if activation.ndim >= 3:
            activation = activation[:, : inputs.real_len, :]
    finally:
        probe_path.unlink(missing_ok=True)

    weight = numpy_helper.to_array(initializer, base_dir=str(args.onnx_model.parent))
    metadata = {
        "matmul_name": node.name,
        "activation_name": activation_name,
        "weight_name": initializer.name,
        "activation_shape": list(activation.shape),
        "weight_shape": list(weight.shape),
        "real_len": inputs.real_len,
        "providers_selected": session.get_providers(),
    }
    return activation, weight, metadata


def quantize_weight_per_output(weight: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
    if weight.ndim != 2:
        raise ValueError(f"expected rank-2 MatMul weight, got {weight.shape}")
    max_value = (1 << (width - 1)) - 1
    scale = np.max(np.abs(weight), axis=0) / max_value
    scale = np.where(scale == 0, 1.0, scale)
    quantized = np.clip(np.rint(weight / scale), -max_value - 1, max_value).astype(np.int16)
    return quantized, scale.astype(np.float64)


def run_experiment(
    activation: np.ndarray, weight: np.ndarray, args: argparse.Namespace
) -> Dict[str, object]:
    flattened = to_bf16(activation.reshape(-1, activation.shape[-1]))
    if weight.shape[0] != flattened.shape[1]:
        raise ValueError("MatMul activation K dimension does not match weight")
    channel_count = min(args.output_channels, weight.shape[1])
    channel_ids = np.linspace(0, weight.shape[1] - 1, channel_count, dtype=int)
    result: Dict[str, object] = {}
    for weight_width in args.weight_widths:
        qweight, scale = quantize_weight_per_output(weight, weight_width)
        tile_size = 64 if weight_width == 4 else 32
        for act_width in args.act_widths:
            samples: List[Dict[str, float]] = []
            span_errors: Dict[str, List[float]] = defaultdict(list)
            zeroed = 0
            nonzero = 0
            for vector in flattened:
                for channel in channel_ids:
                    qweight_column = qweight[:, channel]
                    reference = to_bf16(float(reference_bf16(vector, qweight_column)) * scale[channel])
                    block, tiles = block_float_bf16(vector, qweight_column, act_width, tile_size)
                    block = to_bf16(float(block) * scale[channel])
                    error = abs(float(block) - float(reference))
                    samples.append({
                        "reference": float(reference),
                        "absolute_error": error,
                        "ulp_error": abs(ordered_bf16(block) - ordered_bf16(reference)),
                    })
                    for tile in tiles:
                        span_errors[str(tile["exponent_span"])].append(error)
                        zeroed += tile["rounded_to_zero"]
                        nonzero += tile["nonzero_activations"]
            summary = summarize(samples)
            summary["activation_rounded_to_zero_ratio"] = zeroed / nonzero if nonzero else 0.0
            summary["error_by_tile_exponent_span"] = {
                key: {"samples": len(values), "mean_absolute_error": float(np.mean(values)),
                      "max_absolute_error": float(np.max(values))}
                for key, values in sorted(span_errors.items(), key=lambda item: int(item[0]))
            }
            result[f"A{act_width}W{weight_width}"] = summary
    return result


def main() -> None:
    args = parse_args()
    activation, weight, metadata = capture_activation(args)
    payload = {
        "source": "Qwen2.5 ONNX MatMul activation",
        "reference_format": "BF16",
        "prompt": args.prompt,
        "matmul": metadata,
        "weight_quantization": "symmetric per-output-channel signed integer",
        "results": run_experiment(activation, weight, args),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
