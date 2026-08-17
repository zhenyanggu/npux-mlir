#!/usr/bin/env python3
"""Optimize Qwen2.5 prefill static A8W8 QDQ with SQ, AWQ, and GPTQ.

The emitted model remains standard static QDQ: symmetric int8 activations and
per-output-channel symmetric int8 weights. SmoothQuant and AWQ are evaluated
as alternatives. GPTQ is applied only to the better reparameterized candidate.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

from quantize_qwen25_prefill import (
    ListDataReader,
    build_calibration_reader,
    collect_static_linear_node_names,
    convert_graph_float16_to_float32,
    model_input_names,
    parse_providers,
    quantize_with_calibration_providers,
)
from qwen25_onnx_lib import Qwen25PrefillInputs


LAYER_PATTERN = re.compile(
    r"/model/model/layers\.(?P<layer>\d+)/"
    r"(?P<section>self_attn|mlp)/(?P<projection>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)/MatMul$"
)


@dataclass
class PromptRecord:
    prompt: str
    feeds: Dict[str, np.ndarray]
    real_len: int
    last_token_index: int


@dataclass
class TargetGroup:
    key: str
    kind: str
    layer: Optional[int]
    node_names: List[str]
    input_name: str
    weight_names: List[str]


@dataclass
class Candidate:
    method: str
    scale: Optional[np.ndarray]
    weights: Optional[List[np.ndarray]]
    metrics: Dict[str, float]
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupResult:
    group: TargetGroup
    method: str
    metrics: Dict[str, float]
    params: Dict[str, Any]
    fallback_to_float: bool


class Reservoir:
    """Bounded deterministic reservoir for calibration rows."""

    def __init__(self, capacity: int, width: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.values = np.empty((capacity, width), dtype=np.float16)
        self.count = 0
        self.size = 0

    def add(self, rows: np.ndarray) -> None:
        for row in rows:
            if self.size < self.capacity:
                self.values[self.size] = row
                self.size += 1
            else:
                replacement = int(self.rng.integers(0, self.count + 1))
                if replacement < self.capacity:
                    self.values[replacement] = row
            self.count += 1

    def array(self) -> np.ndarray:
        if self.size == 0:
            raise ValueError("activation reservoir is empty")
        return self.values[: self.size].astype(np.float32)


def parse_csv_floats(text: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-prompts", required=True)
    parser.add_argument("--validation-prompts", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--activation-samples", type=int, default=1024)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--activation-calibration", choices=["MinMax", "Percentile", "Entropy"], default="Percentile")
    parser.add_argument("--activation-percentile", type=float, default=99.95)
    parser.add_argument("--smoothquant-alphas", default="0.3,0.5,0.7")
    parser.add_argument("--awq-alphas", default="0.0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--gptq-damps", default="0.01,0.03,0.1")
    parser.add_argument("--gptq-device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--gptq-block-size", type=int, default=128)
    parser.add_argument("--gptq-output-block", type=int, default=4096)
    parser.add_argument("--disable-gptq", action="store_true")
    parser.add_argument("--layer-nrmse", type=float, default=0.01)
    parser.add_argument("--qk-layer-nrmse", type=float, default=0.005)
    parser.add_argument("--layer-cosine", type=float, default=0.9995)
    parser.add_argument("--qk-layer-cosine", type=float, default=0.9998)
    parser.add_argument("--global-kl", type=float, default=0.005)
    parser.add_argument("--global-top1", type=float, default=0.98)
    parser.add_argument("--top1-margin", type=float, default=0.1)
    parser.add_argument(
        "--global-greedy",
        action="store_true",
        help="Select mixed-INT8 groups one at a time using model-level validation.",
    )
    parser.add_argument(
        "--global-greedy-max-nrmse",
        type=float,
        default=0.05,
        help="Only globally test local candidates at or below this NRMSE.",
    )
    parser.add_argument(
        "--global-greedy-max-groups",
        type=int,
        default=8,
        help="Maximum locally ranked groups to test in the global greedy search.",
    )
    parser.add_argument(
        "--global-search-prompts",
        type=int,
        default=16,
        help="Validation prompt count for each greedy trial; final validation still uses all prompts.",
    )
    parser.add_argument("--max-groups", type=int, default=0, help="Limit groups for a short smoke run")
    parser.add_argument("--keep-probe-model", action="store_true")
    return parser.parse_args()


def load_prompt_file(path: str) -> List[str]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError(f"{path} must contain a JSON string list")
        prompts = [item.strip() for item in payload]
    else:
        prompts = [line.strip() for line in text.splitlines()]
    prompts = [item for item in prompts if item and not item.startswith("#")]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def build_prompt_records(
    hf_model: str,
    prompts: Sequence[str],
    input_names: set[str],
    seq_len: int,
    trust_remote_code: bool,
) -> List[PromptRecord]:
    builder = Qwen25PrefillInputs(
        hf_model=hf_model, seq_len=seq_len, trust_remote_code=trust_remote_code
    )
    records: List[PromptRecord] = []
    for prompt in prompts:
        packed = builder.build(prompt)
        missing = input_names - set(packed.feeds)
        if missing:
            raise ValueError(f"input builder does not provide model inputs: {sorted(missing)}")
        records.append(
            PromptRecord(
                prompt=prompt,
                feeds={name: packed.feeds[name] for name in sorted(input_names)},
                real_len=packed.real_len,
                last_token_index=packed.last_token_index,
            )
        )
    return records


def initializer_map(model: onnx.ModelProto) -> Dict[str, onnx.TensorProto]:
    return {item.name: item for item in model.graph.initializer}


def node_map(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    return {item.name: item for item in model.graph.node if item.name}


def producer_map(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    result: Dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for output in node.output:
            if output:
                result[output] = node
    return result


def collect_targets(model: onnx.ModelProto) -> List[TargetGroup]:
    nodes = node_map(model)
    initializers = initializer_map(model)
    grouped: Dict[str, List[str]] = {}
    kinds: Dict[str, Tuple[str, Optional[int]]] = {}
    for name in collect_static_linear_node_names(model):
        node = nodes[name]
        match = LAYER_PATTERN.match(name)
        if match:
            layer = int(match.group("layer"))
            projection = match.group("projection")
            if projection in {"q_proj", "k_proj", "v_proj"}:
                kind = "qkv"
            elif projection in {"gate_proj", "up_proj"}:
                kind = "gate_up"
            else:
                kind = projection
            key = f"layer.{layer}.{kind}"
            kinds[key] = (kind, layer)
        elif name == "/model/lm_head/MatMul":
            key = "lm_head"
            kinds[key] = ("lm_head", None)
        else:
            raise ValueError(f"unclassified static linear node: {name}")
        grouped.setdefault(key, []).append(name)

    targets: List[TargetGroup] = []
    for key, names in grouped.items():
        group_nodes = [nodes[name] for name in names]
        inputs = {node.input[0] for node in group_nodes}
        if len(inputs) != 1:
            raise ValueError(f"{key} does not have a shared activation input: {sorted(inputs)}")
        weights = []
        for node in group_nodes:
            if len(node.input) < 2 or node.input[1] not in initializers:
                raise ValueError(f"{node.name} does not use an initializer weight")
            if node.op_type != "MatMul":
                raise ValueError(f"{node.name} is {node.op_type}; only MatMul is supported")
            weights.append(node.input[1])
        kind, layer = kinds[key]
        targets.append(
            TargetGroup(
                key=key,
                kind=kind,
                layer=layer,
                node_names=names,
                input_name=next(iter(inputs)),
                weight_names=weights,
            )
        )
    # Process the output head first, then shared-normalization groups before
    # their downstream projections. This also makes overnight logs easier to
    # inspect when a job stops early.
    kind_order = {"lm_head": 0, "qkv": 1, "gate_up": 2, "down_proj": 3, "o_proj": 4}
    return sorted(targets, key=lambda item: (kind_order[item.kind], item.layer if item.layer is not None else -1))


def value_info_map(model: onnx.ModelProto) -> Dict[str, onnx.ValueInfoProto]:
    result: Dict[str, onnx.ValueInfoProto] = {}
    for item in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        result[item.name] = item
    return result


def create_probe_model(source: onnx.ModelProto, targets: Sequence[TargetGroup], path: Path) -> List[str]:
    probe = copy.deepcopy(source)
    try:
        probe = shape_inference.infer_shapes(probe)
    except Exception as error:  # pragma: no cover - exporter-specific shape gaps.
        print(f"[profile] shape inference skipped: {error}")
    infos = value_info_map(probe)
    captured = sorted({target.input_name for target in targets})
    existing = {item.name for item in probe.graph.output}
    for name in captured:
        info = infos.get(name)
        if info is None:
            raise ValueError(f"cannot expose activation {name}: missing type information")
        if name not in existing:
            probe.graph.output.append(copy.deepcopy(info))
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        probe,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=path.name + ".data",
        size_threshold=0,
        convert_attribute=False,
    )
    return captured


def sampled_rows(value: np.ndarray, real_len: int, limit: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0, : max(1, min(real_len, array.shape[1]))]
    elif array.ndim >= 2:
        array = array.reshape(-1, array.shape[-1])
    else:
        raise ValueError(f"expected rank >= 2 activation, got shape={array.shape}")
    if array.ndim != 2:
        array = array.reshape(-1, array.shape[-1])
    if array.shape[0] > limit:
        indices = np.linspace(0, array.shape[0] - 1, num=limit, dtype=np.int64)
        array = array[indices]
    return array.astype(np.float16, copy=False)


def capture_activations(
    probe_path: Path,
    captured: Sequence[str],
    records: Sequence[PromptRecord],
    providers: Sequence[str],
    capacity: int,
    samples_per_prompt: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    session = ort.InferenceSession(str(probe_path), providers=list(providers))
    print(f"[profile] providers selected={session.get_providers()}")
    rng = np.random.default_rng(seed)
    reservoirs: Dict[str, Reservoir] = {}
    for index, record in enumerate(records, start=1):
        outputs = session.run(list(captured), record.feeds)
        for name, value in zip(captured, outputs):
            rows = sampled_rows(value, record.real_len, samples_per_prompt)
            if name not in reservoirs:
                reservoirs[name] = Reservoir(capacity=capacity, width=rows.shape[1], rng=rng)
            reservoirs[name].add(rows)
        if index % 16 == 0 or index == len(records):
            print(f"[profile] captured {index}/{len(records)} prompts")
    return {name: reservoir.array() for name, reservoir in reservoirs.items()}


def tensor_array(initializers: Mapping[str, onnx.TensorProto], name: str) -> np.ndarray:
    return numpy_helper.to_array(initializers[name]).astype(np.float32, copy=True)


def quantize_activation(x: np.ndarray, percentile: float) -> np.ndarray:
    bound = float(np.percentile(np.abs(x), percentile))
    scale = max(bound / 127.0, 1e-8)
    return np.clip(np.rint(x / scale), -127, 127).astype(np.float32) * scale


def quantize_weight_per_channel(weight: np.ndarray) -> np.ndarray:
    if weight.ndim != 2:
        raise ValueError(f"expected rank-2 MatMul weight, got {weight.shape}")
    scale = np.max(np.abs(weight), axis=0, keepdims=True) / 127.0
    scale = np.maximum(scale, 1e-8)
    return np.clip(np.rint(weight / scale), -127, 127).astype(np.float32) * scale


def reconstruction_metrics(x: np.ndarray, weights: Sequence[np.ndarray], percentile: float) -> Dict[str, float]:
    xq = quantize_activation(x, percentile)
    squared_error = 0.0
    squared_reference = 0.0
    dot = 0.0
    norm_reference = 0.0
    norm_candidate = 0.0
    max_abs = 0.0
    for weight in weights:
        reference = x @ weight
        candidate = xq @ quantize_weight_per_channel(weight)
        difference = candidate - reference
        squared_error += float(np.sum(difference * difference, dtype=np.float64))
        squared_reference += float(np.sum(reference * reference, dtype=np.float64))
        dot += float(np.sum(reference * candidate, dtype=np.float64))
        norm_reference += float(np.sum(reference * reference, dtype=np.float64))
        norm_candidate += float(np.sum(candidate * candidate, dtype=np.float64))
        max_abs = max(max_abs, float(np.max(np.abs(difference))))
    return {
        "nrmse": math.sqrt(squared_error / max(squared_reference, 1e-12)),
        "cosine": dot / max(math.sqrt(norm_reference * norm_candidate), 1e-12),
        "max_abs_error": max_abs,
    }


def combined_weight_stat(weights: Sequence[np.ndarray], reducer: str) -> np.ndarray:
    rows = []
    for weight in weights:
        if reducer == "max":
            rows.append(np.max(np.abs(weight), axis=1))
        elif reducer == "mean":
            rows.append(np.mean(np.abs(weight), axis=1))
        else:
            raise ValueError(f"unsupported reducer: {reducer}")
    return np.maximum(np.max(np.stack(rows, axis=0), axis=0), 1e-8)


def normalize_scale(scale: np.ndarray) -> np.ndarray:
    safe = np.clip(scale.astype(np.float32), 1e-4, 1e4)
    return safe / float(np.exp(np.mean(np.log(safe))))


def smoothquant_scale(x: np.ndarray, weights: Sequence[np.ndarray], alpha: float) -> np.ndarray:
    activation = np.maximum(np.percentile(np.abs(x), 99.9, axis=0), 1e-8)
    weight = combined_weight_stat(weights, "max")
    return normalize_scale(np.power(activation, alpha) / np.power(weight, 1.0 - alpha))


def awq_scale(x: np.ndarray, weights: Sequence[np.ndarray], alpha: float) -> np.ndarray:
    # Activation-aware scaling search. This is intentionally evaluated as an
    # alternative to SmoothQuant, not composed with it.
    activation = np.maximum(np.mean(np.abs(x), axis=0), 1e-8)
    weight = combined_weight_stat(weights, "mean")
    return normalize_scale(np.power(activation, alpha) / np.power(weight, 1.0 - alpha))


def transformed_weights(weights: Sequence[np.ndarray], scale: Optional[np.ndarray]) -> List[np.ndarray]:
    if scale is None:
        return [item.copy() for item in weights]
    return [item * scale[:, None] for item in weights]


def transformed_input(x: np.ndarray, scale: Optional[np.ndarray]) -> np.ndarray:
    return x if scale is None else x / scale[None, :]


def is_accepted(target: TargetGroup, metrics: Mapping[str, float], args: argparse.Namespace) -> bool:
    qk = target.kind == "qkv"
    nrmse_limit = args.qk_layer_nrmse if qk else args.layer_nrmse
    cosine_limit = args.qk_layer_cosine if qk else args.layer_cosine
    return metrics["nrmse"] <= nrmse_limit and metrics["cosine"] >= cosine_limit


def resolve_torch_device(requested: str) -> str:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("GPTQ requires torch; pass --disable-gptq to continue without it") from error
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--gptq-device requests CUDA but torch.cuda.is_available() is false")
    return requested


def gptq_round_weight(
    weight: np.ndarray,
    x: np.ndarray,
    damp: float,
    device: str,
    block_size: int,
    output_block: int,
) -> np.ndarray:
    """GPTQ row-wise rounding with one symmetric int8 scale per output channel."""
    import torch

    if weight.ndim != 2 or x.ndim != 2 or weight.shape[0] != x.shape[1]:
        raise ValueError(f"GPTQ shape mismatch: x={x.shape}, weight={weight.shape}")
    torch_device = torch.device(device)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=torch_device)
    hessian = (x_tensor.transpose(0, 1) @ x_tensor) / max(x_tensor.shape[0], 1)
    diagonal = torch.diag(hessian)
    hessian += torch.eye(hessian.shape[0], device=torch_device) * (
        float(damp) * torch.mean(diagonal).clamp_min(1e-8)
    )
    try:
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
        inverse = torch.linalg.cholesky(inverse, upper=True)
    except RuntimeError as error:
        raise RuntimeError(f"GPTQ Hessian factorization failed: {error}") from error

    in_features, out_features = weight.shape
    result = np.empty_like(weight, dtype=np.float32)
    for output_start in range(0, out_features, output_block):
        output_end = min(output_start + output_block, out_features)
        values = torch.as_tensor(
            weight[:, output_start:output_end].transpose(1, 0),
            dtype=torch.float32,
            device=torch_device,
        )
        scales = torch.max(torch.abs(values), dim=1, keepdim=True).values.div(127.0).clamp_min(1e-8)
        quantized = torch.empty_like(values)
        for start in range(0, in_features, block_size):
            end = min(start + block_size, in_features)
            work = values[:, start:end].clone()
            errors = torch.zeros_like(work)
            local_inverse = inverse[start:end, start:end]
            for column in range(end - start):
                diagonal_value = local_inverse[column, column].clamp_min(1e-8)
                q = torch.clamp(torch.round(work[:, column : column + 1] / scales), -127, 127)
                quantized[:, start + column : start + column + 1] = q
                error = (work[:, column : column + 1] - q * scales) / diagonal_value
                errors[:, column : column + 1] = error
                work[:, column:] -= error @ local_inverse[column : column + 1, column:]
            if end < in_features:
                values[:, end:] -= errors @ inverse[start:end, end:]
        dequantized = (quantized * scales).transpose(1, 0).contiguous().cpu().numpy()
        result[:, output_start:output_end] = dequantized
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def gptq_candidate(
    x: np.ndarray,
    weights: Sequence[np.ndarray],
    percentile: float,
    damps: Sequence[float],
    device: str,
    block_size: int,
    output_block: int,
) -> Candidate:
    best: Optional[Candidate] = None
    for damp in damps:
        rounded = [
            gptq_round_weight(
                weight=item,
                x=x,
                damp=damp,
                device=device,
                block_size=block_size,
                output_block=output_block,
            )
            for item in weights
        ]
        # The final ORT pass requantizes these dequantized values to standard
        # per-channel QDQ. The local simulation uses them directly.
        xq = quantize_activation(x, percentile)
        squared_error = 0.0
        squared_reference = 0.0
        dot = 0.0
        norm_reference = 0.0
        norm_candidate = 0.0
        max_abs = 0.0
        for original, candidate_weight in zip(weights, rounded):
            reference = x @ original
            candidate_output = xq @ candidate_weight
            difference = candidate_output - reference
            squared_error += float(np.sum(difference * difference, dtype=np.float64))
            squared_reference += float(np.sum(reference * reference, dtype=np.float64))
            dot += float(np.sum(reference * candidate_output, dtype=np.float64))
            norm_reference += float(np.sum(reference * reference, dtype=np.float64))
            norm_candidate += float(np.sum(candidate_output * candidate_output, dtype=np.float64))
            max_abs = max(max_abs, float(np.max(np.abs(difference))))
        metrics = {
            "nrmse": math.sqrt(squared_error / max(squared_reference, 1e-12)),
            "cosine": dot / max(math.sqrt(norm_reference * norm_candidate), 1e-12),
            "max_abs_error": max_abs,
        }
        candidate = Candidate(
            method="gptq",
            scale=None,
            weights=rounded,
            metrics=metrics,
            params={"damp": damp, "device": device},
        )
        if best is None or candidate.metrics["nrmse"] < best.metrics["nrmse"]:
            best = candidate
    assert best is not None
    return best


def fold_initializer_for_scale(model: onnx.ModelProto, input_name: str, width: int) -> Optional[str]:
    producer = producer_map(model).get(input_name)
    initializers = initializer_map(model)
    if producer is None or producer.op_type != "Mul":
        return None
    candidates = [name for name in producer.input if name in initializers]
    if len(candidates) != 1:
        return None
    array = tensor_array(initializers, candidates[0])
    return candidates[0] if array.size == width else None


def replace_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    for tensor in model.graph.initializer:
        if tensor.name == name:
            tensor.CopyFrom(numpy_helper.from_array(array.astype(np.float32), name=name))
            return
    raise KeyError(f"initializer not found: {name}")


def apply_candidate(
    model: onnx.ModelProto,
    target: TargetGroup,
    candidate: Candidate,
    fold_name: Optional[str],
) -> None:
    current_initializers = initializer_map(model)
    scale = candidate.scale
    if scale is not None:
        if fold_name is None:
            raise ValueError(f"{target.key}: cannot apply an unfused channel scale")
        gamma = tensor_array(current_initializers, fold_name).reshape(-1)
        if gamma.size != scale.size:
            raise ValueError(f"{target.key}: RMSNorm scale size mismatch")
        replace_initializer(model, fold_name, (gamma / scale).reshape(current_initializers[fold_name].dims))

    for index, weight_name in enumerate(target.weight_names):
        if candidate.weights is not None:
            final_weight = candidate.weights[index]
        else:
            final_weight = tensor_array(current_initializers, weight_name)
            if scale is not None:
                final_weight = final_weight * scale[:, None]
        replace_initializer(model, weight_name, final_weight)


def optimize_target(
    model: onnx.ModelProto,
    target: TargetGroup,
    activation: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Candidate, bool]:
    initializers = initializer_map(model)
    weights = [tensor_array(initializers, name) for name in target.weight_names]
    if any(weight.shape[0] != activation.shape[1] for weight in weights):
        raise ValueError(
            f"{target.key}: activation width={activation.shape[1]} does not match "
            f"weights={[item.shape for item in weights]}"
        )
    fold_name = fold_initializer_for_scale(model, target.input_name, activation.shape[1])
    supports_fused_scale = fold_name is not None and target.kind in {"qkv", "gate_up", "lm_head"}
    base = Candidate(
        method="baseline",
        scale=None,
        weights=None,
        metrics=reconstruction_metrics(activation, weights, args.activation_percentile),
        params={},
    )

    if supports_fused_scale:
        smooth_candidates: List[Candidate] = []
        for alpha in parse_csv_floats(args.smoothquant_alphas):
            scale = smoothquant_scale(activation, weights, alpha)
            candidate = Candidate(
                method="smoothquant",
                scale=scale,
                weights=None,
                metrics=reconstruction_metrics(
                    transformed_input(activation, scale),
                    transformed_weights(weights, scale),
                    args.activation_percentile,
                ),
                params={"alpha": alpha, "fold_initializer": fold_name},
            )
            smooth_candidates.append(candidate)
        smooth = min(smooth_candidates, key=lambda item: item.metrics["nrmse"])
        if not args.global_greedy and is_accepted(target, smooth.metrics, args):
            return smooth, False

        awq_candidates: List[Candidate] = []
        for alpha in parse_csv_floats(args.awq_alphas):
            scale = awq_scale(activation, weights, alpha)
            awq_candidates.append(
                Candidate(
                    method="awq",
                    scale=scale,
                    weights=None,
                    metrics=reconstruction_metrics(
                        transformed_input(activation, scale),
                        transformed_weights(weights, scale),
                        args.activation_percentile,
                    ),
                    params={"alpha": alpha, "fold_initializer": fold_name},
                )
            )
        awq = min(awq_candidates, key=lambda item: item.metrics["nrmse"])
        if not args.global_greedy and is_accepted(target, awq.metrics, args):
            return awq, False

        winner = min([base, smooth, awq], key=lambda item: item.metrics["nrmse"])
    else:
        winner = base
        # o_proj and down_proj have no preceding RMSNorm scale that can be
        # folded safely. Keep baseline A8W8 when it already meets the local
        # error budget and reserve GPTQ for the difficult projections only.
        if not args.global_greedy and is_accepted(target, winner.metrics, args):
            return winner, False

    # Global greedy mode evaluates base/SmoothQuant/AWQ candidates in the
    # actual QDQ graph. Keep GPTQ as the second-stage rescue path instead of
    # retaining full rounded weights for every one of the 97 groups in RAM.
    if args.global_greedy:
        return winner, True

    if args.disable_gptq:
        return winner, not is_accepted(target, winner.metrics, args)

    device = resolve_torch_device(args.gptq_device)
    input_for_gptq = transformed_input(activation, winner.scale)
    weights_for_gptq = transformed_weights(weights, winner.scale)
    gptq = gptq_candidate(
        x=input_for_gptq,
        weights=weights_for_gptq,
        percentile=args.activation_percentile,
        damps=parse_csv_floats(args.gptq_damps),
        device=device,
        block_size=args.gptq_block_size,
        output_block=args.gptq_output_block,
    )
    gptq.method = f"{winner.method}_gptq" if winner.method != "baseline" else "gptq"
    gptq.scale = winner.scale
    gptq.params.update(winner.params)
    if is_accepted(target, gptq.metrics, args):
        return gptq, False
    # A failed GPTQ search must not replace a better baseline/SQ/AWQ
    # diagnostic result. The caller keeps this group in FP32 either way.
    return min([winner, gptq], key=lambda item: item.metrics["nrmse"]), True


def save_model(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # save_model externalizes tensors in-place. Keep the in-memory candidate
    # independent of temporary trial files that global greedy later removes.
    serializable_model = copy.deepcopy(model)
    onnx.save_model(
        serializable_model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=path.name + ".data",
        size_threshold=0,
        convert_attribute=False,
    )
    onnx.checker.check_model(str(path))


def quantize_model(
    source_model: Path,
    output_model: Path,
    hf_model: str,
    prompts: Sequence[str],
    input_names: set[str],
    args: argparse.Namespace,
    nodes_to_quantize: Sequence[str],
    providers: Sequence[str],
) -> None:
    if not nodes_to_quantize:
        # ORT treats an empty list as no node filter, which would quantize
        # every eligible MatMul and violate an all-FP32 fallback policy.
        raise ValueError("refusing to call ORT quantization with an empty node list")
    reader = build_calibration_reader(
        hf_model=hf_model,
        input_names=input_names,
        prompts=prompts,
        seq_len=args.seq_len,
        trust_remote_code=args.trust_remote_code,
    )
    calibration_method = getattr(CalibrationMethod, args.activation_calibration)
    options: Dict[str, Any] = {
        "WeightSymmetric": True,
        "ActivationSymmetric": True,
        "MatMulConstBOnly": True,
    }
    if args.activation_calibration == "Percentile":
        options["CalibPercentile"] = args.activation_percentile
    quantize_with_calibration_providers(
        providers=providers,
        model_input=str(source_model),
        model_output=str(output_model),
        calibration_data_reader=reader,
        calibrate_method=calibration_method,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm", "Softmax"],
        nodes_to_quantize=list(nodes_to_quantize),
        extra_options=options,
        use_external_data_format=True,
    )
    onnx.checker.check_model(str(output_model))


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def evaluate_models(
    reference_model: Path,
    candidate_model: Path,
    records: Sequence[PromptRecord],
    providers: Sequence[str],
    top1_margin: float,
) -> Dict[str, Any]:
    reference = ort.InferenceSession(str(reference_model), providers=list(providers))
    candidate = ort.InferenceSession(str(candidate_model), providers=list(providers))
    kl_values: List[float] = []
    top1_checked = 0
    top1_matches = 0
    max_abs = 0.0
    mean_abs: List[float] = []
    details = []
    for record in records:
        reference_logits = reference.run(None, record.feeds)[0][:, record.last_token_index, :].astype(np.float32)
        candidate_logits = candidate.run(None, record.feeds)[0][:, record.last_token_index, :].astype(np.float32)
        reference_probability = softmax(reference_logits)
        candidate_probability = np.clip(softmax(candidate_logits), 1e-12, 1.0)
        kl = float(np.sum(reference_probability * (np.log(reference_probability + 1e-12) - np.log(candidate_probability))))
        order = np.argpartition(reference_logits[0], -2)[-2:]
        order = order[np.argsort(reference_logits[0, order])][::-1]
        margin = float(reference_logits[0, order[0]] - reference_logits[0, order[1]])
        match = bool(np.argmax(reference_logits) == np.argmax(candidate_logits))
        if margin >= top1_margin:
            top1_checked += 1
            top1_matches += int(match)
        difference = np.abs(reference_logits - candidate_logits)
        max_abs = max(max_abs, float(np.max(difference)))
        mean_abs.append(float(np.mean(difference)))
        kl_values.append(kl)
        details.append({"prompt": record.prompt, "kl": kl, "top1_match": match, "top1_margin": margin})
    return {
        "providers": {"reference": reference.get_providers(), "candidate": candidate.get_providers()},
        "prompt_count": len(records),
        "mean_logit_kl": float(np.mean(kl_values)),
        "p99_logit_kl": float(np.percentile(kl_values, 99)),
        "top1_checked": top1_checked,
        "top1_matches": top1_matches,
        "top1_agreement": top1_matches / max(top1_checked, 1),
        "mean_abs_diff": float(np.mean(mean_abs)),
        "max_abs_diff": max_abs,
        "records": details,
    }


def passes_global_gate(evaluation: Mapping[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        evaluation["mean_logit_kl"] <= args.global_kl
        and evaluation["top1_agreement"] >= args.global_top1
    )


def globally_select_candidates(
    base_model: onnx.ModelProto,
    candidates: Sequence[Tuple[TargetGroup, Candidate]],
    output_dir: Path,
    hf_model: str,
    calibration_prompts: Sequence[str],
    validation_records: Sequence[PromptRecord],
    activations: Mapping[str, np.ndarray],
    input_names: set[str],
    input_model: Path,
    providers: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[onnx.ModelProto, List[str], Dict[str, Candidate], List[Dict[str, Any]]]:
    ranked = [
        (target, candidate)
        for target, candidate in candidates
        if candidate.weights is None and candidate.metrics["nrmse"] <= args.global_greedy_max_nrmse
    ]
    ranked.sort(key=lambda item: (item[1].metrics["nrmse"], item[0].key))
    if args.global_greedy_max_groups > 0:
        ranked = ranked[: args.global_greedy_max_groups]
    search_records = validation_records[: args.global_search_prompts]
    if not search_records:
        raise ValueError("global greedy search requires at least one validation prompt")

    selected_model = copy.deepcopy(base_model)
    selected_nodes: List[str] = []
    selected_keys: List[str] = []
    selected_candidates: Dict[str, Candidate] = {}
    trace: List[Dict[str, Any]] = []
    search_root = output_dir / "global_search_tmp"
    search_root.mkdir(parents=True, exist_ok=True)
    try:
        for index, (target, candidate) in enumerate(ranked, start=1):
            trial_dir = search_root / f"{index:03d}_{target.key.replace('.', '_')}"
            trial_source = trial_dir / "source.onnx"
            trial_output = trial_dir / "candidate.onnx"
            trial = copy.deepcopy(selected_model)
            fold_name = fold_initializer_for_scale(trial, target.input_name, candidate.scale.size) if candidate.scale is not None else None
            apply_candidate(trial, target, candidate, fold_name)
            trial_nodes = selected_nodes + target.node_names
            print(f"[global] {index}/{len(ranked)} test {target.key} ({candidate.method})")
            save_model(trial, trial_source)
            quantize_model(
                source_model=trial_source,
                output_model=trial_output,
                hf_model=hf_model,
                prompts=calibration_prompts,
                input_names=input_names,
                args=args,
                nodes_to_quantize=trial_nodes,
                providers=providers,
            )
            evaluation = evaluate_models(
                reference_model=input_model,
                candidate_model=trial_output,
                records=search_records,
                providers=providers,
                top1_margin=args.top1_margin,
            )
            accepted = passes_global_gate(evaluation, args)
            trace.append(
                {
                    "key": target.key,
                    "method": candidate.method,
                    "local_metrics": candidate.metrics,
                    "accepted": accepted,
                    "global_metrics": {
                        key: evaluation[key]
                        for key in ("mean_logit_kl", "top1_agreement", "top1_checked", "mean_abs_diff")
                    },
                }
            )
            if accepted:
                selected_model = trial
                selected_nodes = trial_nodes
                selected_keys.append(target.key)
                selected_candidates[target.key] = candidate
                print(f"[global] keep {target.key}")
            else:
                print(f"[global] reject {target.key}")
            shutil.rmtree(trial_dir, ignore_errors=True)
            if accepted or args.disable_gptq:
                continue

            initializers = initializer_map(selected_model)
            weights = [tensor_array(initializers, name) for name in target.weight_names]
            rescue = gptq_candidate(
                x=transformed_input(activations[target.input_name], candidate.scale),
                weights=transformed_weights(weights, candidate.scale),
                percentile=args.activation_percentile,
                damps=parse_csv_floats(args.gptq_damps),
                device=resolve_torch_device(args.gptq_device),
                block_size=args.gptq_block_size,
                output_block=args.gptq_output_block,
            )
            rescue.method = f"{candidate.method}_gptq" if candidate.method != "baseline" else "gptq"
            rescue.scale = candidate.scale
            rescue.params.update(candidate.params)
            if rescue.metrics["nrmse"] >= candidate.metrics["nrmse"]:
                trace.append(
                    {
                        "key": target.key,
                        "method": rescue.method,
                        "local_metrics": rescue.metrics,
                        "accepted": False,
                        "reason": "GPTQ did not improve local NRMSE",
                    }
                )
                continue

            rescue_dir = search_root / f"{index:03d}_{target.key.replace('.', '_')}_gptq"
            rescue_source = rescue_dir / "source.onnx"
            rescue_output = rescue_dir / "candidate.onnx"
            rescue_trial = copy.deepcopy(selected_model)
            rescue_fold = fold_initializer_for_scale(
                rescue_trial, target.input_name, rescue.scale.size
            ) if rescue.scale is not None else None
            apply_candidate(rescue_trial, target, rescue, rescue_fold)
            print(f"[global] {index}/{len(ranked)} GPTQ rescue {target.key}")
            save_model(rescue_trial, rescue_source)
            quantize_model(
                source_model=rescue_source,
                output_model=rescue_output,
                hf_model=hf_model,
                prompts=calibration_prompts,
                input_names=input_names,
                args=args,
                nodes_to_quantize=trial_nodes,
                providers=providers,
            )
            rescue_evaluation = evaluate_models(
                reference_model=input_model,
                candidate_model=rescue_output,
                records=search_records,
                providers=providers,
                top1_margin=args.top1_margin,
            )
            rescue_accepted = passes_global_gate(rescue_evaluation, args)
            trace.append(
                {
                    "key": target.key,
                    "method": rescue.method,
                    "local_metrics": rescue.metrics,
                    "accepted": rescue_accepted,
                    "global_metrics": {
                        key: rescue_evaluation[key]
                        for key in ("mean_logit_kl", "top1_agreement", "top1_checked", "mean_abs_diff")
                    },
                }
            )
            if rescue_accepted:
                selected_model = rescue_trial
                selected_nodes = trial_nodes
                selected_keys.append(target.key)
                selected_candidates[target.key] = rescue
                print(f"[global] keep GPTQ rescue {target.key}")
            else:
                print(f"[global] reject GPTQ rescue {target.key}")
            shutil.rmtree(rescue_dir, ignore_errors=True)
    finally:
        shutil.rmtree(search_root, ignore_errors=True)
    return selected_model, selected_keys, selected_candidates, trace


def group_json(result: GroupResult) -> Dict[str, Any]:
    return {
        "key": result.group.key,
        "kind": result.group.kind,
        "layer": result.group.layer,
        "node_names": result.group.node_names,
        "input_name": result.group.input_name,
        "weight_names": result.group.weight_names,
        "method": result.method,
        "metrics": result.metrics,
        "params": result.params,
        "fallback_to_float": result.fallback_to_float,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_model = Path(args.input_model).resolve()
    providers = parse_providers(args.providers)
    calibration_prompts = load_prompt_file(args.calibration_prompts)
    validation_prompts = load_prompt_file(args.validation_prompts)
    print(f"[config] calibration prompts={len(calibration_prompts)} validation prompts={len(validation_prompts)}")
    print(f"[config] requested providers={providers}")

    source = onnx.load(str(input_model), load_external_data=True)
    targets = collect_targets(source)
    if args.max_groups:
        targets = targets[: args.max_groups]
    print(f"[graph] optimizing {len(targets)} groups from {len(collect_static_linear_node_names(source))} static MatMuls")
    input_names = model_input_names(source)
    calibration_records = build_prompt_records(
        args.hf_model, calibration_prompts, input_names, args.seq_len, args.trust_remote_code
    )
    validation_records = build_prompt_records(
        args.hf_model, validation_prompts, input_names, args.seq_len, args.trust_remote_code
    )

    probe_path = output_dir / "activation_probe.onnx"
    captured = create_probe_model(source, targets, probe_path)
    activations = capture_activations(
        probe_path=probe_path,
        captured=captured,
        records=calibration_records,
        providers=providers,
        capacity=args.activation_samples,
        samples_per_prompt=args.samples_per_prompt,
        seed=args.seed,
    )
    if not args.keep_probe_model:
        probe_data = probe_path.with_name(probe_path.name + ".data")
        probe_path.unlink(missing_ok=True)
        probe_data.unlink(missing_ok=True)

    # ORT static quantization operates on float32 QDQ sources. Convert once
    # after profiling so all candidate statistics retain the original graph.
    convert_graph_float16_to_float32(source.graph)
    all_static = collect_static_linear_node_names(source)
    target_node_names = {name for target in targets for name in target.node_names}
    results: List[GroupResult] = []
    fallback_nodes: set[str] = set()
    candidate_by_key: Dict[str, Candidate] = {}
    for index, target in enumerate(targets, start=1):
        print(f"[optimize] {index}/{len(targets)} {target.key}")
        candidate, fallback = optimize_target(source, target, activations[target.input_name], args)
        fold_name = fold_initializer_for_scale(source, target.input_name, activations[target.input_name].shape[1])
        if candidate.scale is not None:
            candidate.params = dict(candidate.params)
            candidate.params["channel_scale"] = [float(value) for value in candidate.scale]
        candidate_by_key[target.key] = candidate
        if args.global_greedy:
            results.append(
                GroupResult(
                    group=target,
                    method="fp32",
                    metrics=candidate.metrics,
                    params=candidate.params,
                    fallback_to_float=True,
                )
            )
            continue
        if fallback:
            fallback_nodes.update(target.node_names)
            print(f"[optimize] {target.key}: fallback=float, nrmse={candidate.metrics['nrmse']:.6f}")
        else:
            apply_candidate(source, target, candidate, fold_name)
            print(
                f"[optimize] {target.key}: {candidate.method}, "
                f"nrmse={candidate.metrics['nrmse']:.6f}, cosine={candidate.metrics['cosine']:.6f}"
            )
        results.append(
            GroupResult(
                group=target,
                method="fp32" if fallback else candidate.method,
                metrics=candidate.metrics,
                params=candidate.params,
                fallback_to_float=fallback,
            )
        )

    global_trace: List[Dict[str, Any]] = []
    if args.global_greedy:
        source, selected_keys, selected_candidates, global_trace = globally_select_candidates(
            base_model=source,
            candidates=[(target, candidate_by_key[target.key]) for target in targets],
            output_dir=output_dir,
            hf_model=args.hf_model,
            calibration_prompts=calibration_prompts,
            validation_records=validation_records,
            activations=activations,
            input_names=input_names,
            input_model=input_model,
            providers=providers,
            args=args,
        )
        selected_key_set = set(selected_keys)
        selected_node_names = {
            name for target in targets if target.key in selected_key_set for name in target.node_names
        }
        fallback_nodes = set(all_static) - selected_node_names
        for result in results:
            if result.group.key in selected_key_set:
                candidate = selected_candidates[result.group.key]
                result.method = candidate.method
                result.metrics = candidate.metrics
                result.params = candidate.params
                result.fallback_to_float = False
    else:
        # max-groups is a smoke-test mode: unprofiled static MatMuls must
        # never be quantized merely because they were absent from targets.
        fallback_nodes.update(set(all_static) - target_node_names)

    optimized_source = output_dir / "qwen25_prefill_optimized_source_fp32.onnx"
    save_model(source, optimized_source)
    final_nodes = [name for name in all_static if name not in fallback_nodes]
    output_model = output_dir / "qwen25_prefill_static_int8_sota.onnx"
    policy = {
        "input_model": str(input_model),
        "optimized_source_model": str(optimized_source),
        "output_model": str(output_model) if final_nodes else None,
        "calibration_prompt_count": len(calibration_prompts),
        "validation_prompt_count": len(validation_prompts),
        "activation_calibration": args.activation_calibration,
        "activation_percentile": args.activation_percentile,
        "int8_format": "static symmetric A8 + symmetric per-output-channel W8",
        "quantized_nodes": final_nodes,
        "fallback_nodes": sorted(fallback_nodes),
        "groups": [group_json(item) for item in results],
    }
    if args.global_greedy:
        policy["global_greedy"] = {
            "max_nrmse": args.global_greedy_max_nrmse,
            "max_groups": args.global_greedy_max_groups,
            "search_prompt_count": len(validation_records[: args.global_search_prompts]),
            "trace": global_trace,
        }
    if not final_nodes:
        evaluation = {
            "accepted": False,
            "reason": "all target groups failed local acceptance; no INT8 model was emitted",
        }
        policy["quantization_status"] = "no_quantizable_nodes"
        (output_dir / "layer_profile.json").write_text(
            json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "quant_policy.json").write_text(
            json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "final_eval.json").write_text(
            json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise RuntimeError(
            "all target groups were kept in FP32; refusing to let ORT quantize "
            "an empty node selection"
        )
    quantize_model(
        source_model=optimized_source,
        output_model=output_model,
        hf_model=args.hf_model,
        prompts=calibration_prompts,
        input_names=input_names,
        args=args,
        nodes_to_quantize=final_nodes,
        providers=providers,
    )
    evaluation = evaluate_models(
        reference_model=input_model,
        candidate_model=output_model,
        records=validation_records,
        providers=providers,
        top1_margin=args.top1_margin,
    )
    evaluation["accepted"] = passes_global_gate(evaluation, args)
    policy["quantization_status"] = "quantized"
    (output_dir / "layer_profile.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "quant_policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "final_eval.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"policy": str(output_dir / 'layer_profile.json'), "evaluation": evaluation}, indent=2))
    if not evaluation["accepted"]:
        raise RuntimeError(
            "final model-level accuracy gate failed; inspect final_eval.json and "
            "layer_profile.json before using the emitted model"
        )


if __name__ == "__main__":
    main()
