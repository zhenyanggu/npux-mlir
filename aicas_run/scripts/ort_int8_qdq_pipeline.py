#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from tqdm import tqdm
from transformers import AutoProcessor

from smolvlm2_onnx_lib import (
    OCRBENCH_SCORE_KEYS,
    SmolVLM2OnnxRunner,
    ensure_dir,
    evaluate_prediction,
    load_json,
    save_json,
)

try:
    import onnx  # type: ignore
except Exception:  # pragma: no cover - runtime dependency in env
    onnx = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AICAS_ROOT = os.path.dirname(SCRIPT_DIR)
if AICAS_ROOT not in sys.path:
    sys.path.insert(0, AICAS_ROOT)

DEFAULT_INPUT_JSON = os.path.join(AICAS_ROOT, "FullTest.json")
DEFAULT_IMAGE_ROOT = os.path.join(AICAS_ROOT, "data")
DEFAULT_MODEL_DIR = os.path.join(AICAS_ROOT, "models")
DEFAULT_ASSET_DIR = AICAS_ROOT
DEFAULT_RESULTS_DIR = os.path.join(AICAS_ROOT, "results")
DEFAULT_DEV_JSON = os.path.join(DEFAULT_RESULTS_DIR, "dev_260.json")
DEFAULT_CALIB_JSON = os.path.join(DEFAULT_RESULTS_DIR, "calib_520.json")
DEFAULT_SPLIT_SUMMARY_JSON = os.path.join(DEFAULT_RESULTS_DIR, "int8_qdq_split_summary.json")
DEFAULT_QUANT_OUT_DIR = os.path.join(DEFAULT_RESULTS_DIR, "int8_qdq_models")
DEFAULT_QUANT_SUMMARY_JSON = os.path.join(DEFAULT_RESULTS_DIR, "int8_qdq_quant_summary.json")
DEFAULT_EVAL_OUT_DIR = os.path.join(DEFAULT_RESULTS_DIR, "int8_qdq_eval")
DEFAULT_REPORT_PATH = os.path.join(DEFAULT_RESULTS_DIR, "int8_qdq_fulltest_report.md")

DEFAULT_PROVIDER_TEXT = "CUDAExecutionProvider,CPUExecutionProvider"
DEFAULT_SEED = 20260329
DEFAULT_DEV_PER_TYPE = 20
DEFAULT_CALIB_PER_TYPE = 40
DEFAULT_PASS_THRESHOLD = 0.05
DEFAULT_FLUSH_EVERY = 20
DEFAULT_CALIB_PROGRESS_EVERY = 10

EXCLUDED_FROM_OFFICIAL_TYPES = {"Chinese", "Occluded", "Semantic Text Recognition"}
NON_FP32_FLOAT_INPUT_TYPES = {"FLOAT16", "DOUBLE", "BFLOAT16"}
DECODER_REQUIRED_CUSTOM_OPS = {
    "com.microsoft::MultiHeadAttention",
    "com.microsoft::RotaryEmbedding",
    "com.microsoft::SkipSimplifiedLayerNormalization",
}


QUANT_CONFIGS: List[Dict[str, Any]] = [
    {
        "config_id": "cfg1",
        "name": "MinMax + exclude MatMul/Gemm output quant",
        "calibration_method": "MinMax",
        "percentile": None,
        "exclude_output_quant_ops": ["MatMul", "Gemm"],
    },
    {
        "config_id": "cfg2",
        "name": "Percentile(99.99) + exclude MatMul/Gemm output quant",
        "calibration_method": "Percentile",
        "percentile": 99.99,
        "exclude_output_quant_ops": ["MatMul", "Gemm"],
    },
    {
        "config_id": "cfg3",
        "name": "Entropy + exclude MatMul/Gemm output quant",
        "calibration_method": "Entropy",
        "percentile": None,
        "exclude_output_quant_ops": ["MatMul", "Gemm"],
    },
    {
        "config_id": "cfg4",
        "name": "MinMax + keep MatMul/Gemm output quant",
        "calibration_method": "MinMax",
        "percentile": None,
        "exclude_output_quant_ops": [],
    },
]
QUANT_CONFIG_BY_ID = {item["config_id"]: item for item in QUANT_CONFIGS}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_provider_text(provider_text: str) -> List[str]:
    providers = [item.strip() for item in provider_text.split(",") if item.strip()]
    return providers if providers else ["CPUExecutionProvider"]


def resolve_runtime_providers(provider_text: str) -> Dict[str, Any]:
    requested = parse_provider_text(provider_text)
    available = list(ort.get_available_providers())
    selected = [name for name in requested if name in available]
    if not selected:
        if "CPUExecutionProvider" in available:
            selected = ["CPUExecutionProvider"]
        else:
            selected = available[:]
    if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in selected:
        selected.append("CPUExecutionProvider")
    return {
        "requested": requested,
        "available": available,
        "selected": selected,
    }


def resolve_existing_path(
    model_dir: str,
    explicit_path: str,
    candidates: Sequence[str],
    label: str,
) -> str:
    tried: List[str] = []
    if explicit_path:
        path = explicit_path if os.path.isabs(explicit_path) else os.path.join(model_dir, explicit_path)
        path = os.path.abspath(path)
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"{label} model not found: {path}")
    for candidate in candidates:
        path = os.path.abspath(os.path.join(model_dir, candidate))
        tried.append(path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{label} model not found in candidates: {tried}")


@dataclass
class ModelPaths:
    vision: str
    embed: str
    decoder: str


def resolve_quant_source_model_paths(args: argparse.Namespace) -> ModelPaths:
    model_dir = os.path.abspath(args.model_dir)
    return ModelPaths(
        vision=resolve_existing_path(
            model_dir=model_dir,
            explicit_path=args.quant_source_vision_model,
            candidates=[
                "models/vision_encoder.onnx",
                "vision_encoder.onnx",
            ],
            label="quant-source vision",
        ),
        embed=resolve_existing_path(
            model_dir=model_dir,
            explicit_path=args.quant_source_embed_model,
            candidates=[
                "models/embed_tokens.onnx",
                "embed_tokens.onnx",
            ],
            label="quant-source embed",
        ),
        decoder=resolve_existing_path(
            model_dir=model_dir,
            explicit_path=args.quant_source_decoder_model,
            candidates=[
                "models/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
            ],
            label="quant-source decoder",
        ),
    )


def resolve_fp32_model_paths(args: argparse.Namespace) -> ModelPaths:
    model_dir = os.path.abspath(args.model_dir)
    return ModelPaths(
        vision=resolve_existing_path(
            model_dir=model_dir,
            explicit_path=args.vision_model,
            candidates=[
                "models/vision_encoder.onnx",
                "vision_encoder.onnx",
            ],
            label="vision",
        ),
        embed=resolve_existing_path(
            model_dir=model_dir,
            explicit_path=args.embed_model,
            candidates=[
                "models/embed_tokens.onnx",
                "embed_tokens.onnx",
            ],
            label="embed",
        ),
        decoder=resolve_existing_path(
            model_dir=model_dir,
            explicit_path=args.decoder_model,
            candidates=[
                "models/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
            ],
            label="decoder",
        ),
    )


def resolve_asset_dir(asset_dir_arg: str, model_dir: str) -> str:
    candidates: List[str] = []
    if asset_dir_arg:
        candidates.append(os.path.abspath(asset_dir_arg))
    candidates.append(os.path.abspath(model_dir))
    parent_dir = os.path.abspath(os.path.dirname(model_dir))
    if parent_dir not in candidates:
        candidates.append(parent_dir)
    if os.path.abspath(AICAS_ROOT) not in candidates:
        candidates.append(os.path.abspath(AICAS_ROOT))

    required = ["config.json", "processor_config.json", "tokenizer.json"]
    for candidate in candidates:
        if all(os.path.exists(os.path.join(candidate, name)) for name in required):
            return candidate
    raise FileNotFoundError(
        "Cannot find HF asset directory containing config/tokenizer files. "
        f"Tried: {candidates}"
    )


def resolve_int8_model_paths(quant_out_dir: str, config_id: str, embed_path: str) -> ModelPaths:
    model_dir = os.path.abspath(os.path.join(quant_out_dir, config_id))
    vision = os.path.join(model_dir, "vision_encoder_int8_qdq.onnx")
    decoder = os.path.join(model_dir, "decoder_model_merged_int8_qdq.onnx")
    if not os.path.exists(vision):
        raise FileNotFoundError(vision)
    if not os.path.exists(decoder):
        raise FileNotFoundError(decoder)
    return ModelPaths(
        vision=os.path.abspath(vision),
        embed=os.path.abspath(embed_path),
        decoder=os.path.abspath(decoder),
    )


def parse_config_ids(config_text: str) -> List[str]:
    ids = [item.strip() for item in config_text.split(",") if item.strip()]
    if not ids:
        raise ValueError("No config id provided.")
    for config_id in ids:
        if config_id not in QUANT_CONFIG_BY_ID:
            raise ValueError(f"Unknown config id: {config_id}")
    return ids


def parse_components(components_text: str) -> List[str]:
    components = [item.strip() for item in components_text.split(",") if item.strip()]
    if not components:
        raise ValueError("No component provided.")
    valid = {"vision", "decoder"}
    for component in components:
        if component not in valid:
            raise ValueError(f"Unsupported component: {component}")
    return components


def subset_calib_records(
    records: List[Dict[str, Any]],
    calib_limit: int,
    calib_per_type_override: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if calib_per_type_override > 0:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in records:
            grouped[str(item.get("type", ""))].append(item)
        rng = random.Random(seed)
        selected: List[Dict[str, Any]] = []
        for item_type in sorted(grouped.keys()):
            candidates = grouped[item_type]
            if len(candidates) < calib_per_type_override:
                raise ValueError(
                    f"Type '{item_type}' samples not enough for calib-per-type={calib_per_type_override}: "
                    f"found {len(candidates)}"
                )
            selected.extend(rng.sample(candidates, calib_per_type_override))
        return selected

    if calib_limit > 0 and len(records) > calib_limit:
        return records[:calib_limit]
    return records


def select_records_by_type(
    records: List[Dict[str, Any]],
    dev_per_type: int,
    calib_per_type: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, item in enumerate(records):
        grouped[str(item.get("type", ""))].append((idx, item))

    rng = random.Random(seed)
    ordered_types = sorted(grouped.keys())
    dev: List[Dict[str, Any]] = []
    calib: List[Dict[str, Any]] = []
    dev_indices: set = set()
    calib_indices: set = set()
    per_type_summary: List[Dict[str, Any]] = []

    for item_type in ordered_types:
        candidates = grouped[item_type]
        required = dev_per_type + calib_per_type
        if len(candidates) < required:
            raise ValueError(
                f"Type '{item_type}' samples not enough: need {required}, found {len(candidates)}"
            )
        chosen = rng.sample(candidates, required)
        dev_items = chosen[:dev_per_type]
        calib_items = chosen[dev_per_type:]
        dev.extend(item for _, item in dev_items)
        calib.extend(item for _, item in calib_items)
        dev_indices.update(idx for idx, _ in dev_items)
        calib_indices.update(idx for idx, _ in calib_items)
        per_type_summary.append(
            {
                "type": item_type,
                "total": len(candidates),
                "dev_count": len(dev_items),
                "calib_count": len(calib_items),
            }
        )

    overlap = dev_indices & calib_indices
    if overlap:
        raise ValueError(f"Dev and calib split overlap (sample row index): {len(overlap)}")

    summary = {
        "seed": seed,
        "num_types": len(ordered_types),
        "dev_total": len(dev),
        "calib_total": len(calib),
        "dev_per_type": dev_per_type,
        "calib_per_type": calib_per_type,
        "per_type": per_type_summary,
    }
    return dev, calib, summary


def cmd_prepare(args: argparse.Namespace) -> int:
    input_json = os.path.abspath(args.input_json)
    dev_json = os.path.abspath(args.dev_json_out)
    calib_json = os.path.abspath(args.calib_json_out)
    split_summary = os.path.abspath(args.split_summary_json)

    if args.resume and os.path.exists(dev_json) and os.path.exists(calib_json):
        print(f"[prepare] resume enabled, keep existing split files:\n  {dev_json}\n  {calib_json}")
        return 0

    records = load_json(input_json)
    if not isinstance(records, list):
        raise TypeError(f"Input json must be list, got {type(records).__name__}")

    dev, calib, summary = select_records_by_type(
        records=records,
        dev_per_type=args.dev_per_type,
        calib_per_type=args.calib_per_type,
        seed=args.seed,
    )
    ensure_dir(os.path.dirname(dev_json))
    ensure_dir(os.path.dirname(calib_json))
    save_json(dev, dev_json)
    save_json(calib, calib_json)
    save_json(
        {
            "generated_at": now_text(),
            "input_json": input_json,
            "dev_json": dev_json,
            "calib_json": calib_json,
            "summary": summary,
        },
        split_summary,
    )
    print(f"[prepare] dev split saved: {dev_json} ({len(dev)})")
    print(f"[prepare] calib split saved: {calib_json} ({len(calib)})")
    print(f"[prepare] split summary: {split_summary}")
    return 0


class StreamingDataReader(CalibrationDataReader):
    def __init__(
        self,
        runner: SmolVLM2OnnxRunner,
        calib_records: List[Dict[str, Any]],
        image_root: str,
        mode: str,
        progress_every: int,
    ) -> None:
        self.runner = runner
        self.calib_records = calib_records
        self.image_root = image_root
        self.mode = mode
        self.progress_every = max(1, progress_every)
        self.index = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.index >= len(self.calib_records):
            return None
        record = self.calib_records[self.index]
        self.index += 1
        if self.index == 1 or self.index % self.progress_every == 0 or self.index == len(self.calib_records):
            print(
                f"[calib:{self.mode}] {self.index}/{len(self.calib_records)} "
                f"id={record.get('id', '')}"
            )
        image_path = os.path.join(self.image_root, record["image_path"])
        if self.mode == "vision":
            prepared = self.runner.prepare_inputs(image_path=image_path, question=record["question"])
            return {
                "pixel_values": prepared["pixel_values"].astype(np.float32, copy=False),
                "pixel_attention_mask": prepared["pixel_attention_mask"].astype(bool, copy=False),
            }
        if self.mode == "decoder":
            state = self.runner.prefill(image_path=image_path, question=record["question"])
            return {
                "inputs_embeds": state["merged_embeds"].astype(
                    self.runner.decoder_inputs_embeds_dtype, copy=False
                ),
                "attention_mask": state["attention_mask"].astype(
                    self.runner.decoder_attention_mask_dtype, copy=False
                ),
                "position_ids": state["position_ids"].astype(
                    self.runner.decoder_position_ids_dtype, copy=False
                ),
                **self.runner.zero_past_key_values(batch_size=1),
            }
        raise ValueError(f"Unsupported calibration reader mode: {self.mode}")

    def rewind(self) -> None:
        self.index = 0


class VisionOnlyDataReader(CalibrationDataReader):
    def __init__(
        self,
        asset_dir: str,
        calib_records: List[Dict[str, Any]],
        image_root: str,
        progress_every: int,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(asset_dir)
        self.calib_records = calib_records
        self.image_root = image_root
        self.progress_every = max(1, progress_every)
        self.index = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.index >= len(self.calib_records):
            return None
        record = self.calib_records[self.index]
        self.index += 1
        if self.index == 1 or self.index % self.progress_every == 0 or self.index == len(self.calib_records):
            print(
                f"[calib:vision] {self.index}/{len(self.calib_records)} "
                f"id={record.get('id', '')}"
            )
        from PIL import Image

        image_path = os.path.join(self.image_root, record["image_path"])
        image = Image.open(image_path).convert("RGB")
        try:
            prepared = self.processor.image_processor.preprocess(
                images=[image],
                return_tensors="np",
            )
        finally:
            image.close()
        return {
            "pixel_values": prepared["pixel_values"].astype(np.float32, copy=False),
            "pixel_attention_mask": prepared["pixel_attention_mask"].astype(bool, copy=False),
        }

    def rewind(self) -> None:
        self.index = 0


def build_vision_reader(
    asset_dir: str,
    calib_records: List[Dict[str, Any]],
    image_root: str,
    progress_every: int,
) -> VisionOnlyDataReader:
    return VisionOnlyDataReader(
        asset_dir=asset_dir,
        calib_records=calib_records,
        image_root=image_root,
        progress_every=progress_every,
    )


def build_decoder_reader(
    runner: SmolVLM2OnnxRunner,
    calib_records: List[Dict[str, Any]],
    image_root: str,
    progress_every: int,
) -> StreamingDataReader:
    return StreamingDataReader(
        runner=runner,
        calib_records=calib_records,
        image_root=image_root,
        mode="decoder",
        progress_every=progress_every,
    )


def ensure_onnx_importable() -> None:
    if onnx is None:
        raise RuntimeError(
            "The 'onnx' package is required for quantization coverage stats. "
            "Please activate the project environment first."
        )


def collect_constant_outputs(graph: Any) -> set:
    names = {item.name for item in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for out in node.output:
                if out:
                    names.add(out)
    return names


def graph_input_dtypes(graph: Any) -> Dict[str, str]:
    ensure_onnx_importable()
    result: Dict[str, str] = {}
    for item in graph.input:
        tensor_type = item.type.tensor_type
        result[item.name] = str(onnx.TensorProto.DataType.Name(tensor_type.elem_type))  # type: ignore[attr-defined]
    return result


def is_static_linear_op(node: Any, const_outputs: set) -> bool:
    if node.op_type not in {"MatMul", "Gemm"}:
        return False
    if len(node.input) < 2:
        return False
    return bool(node.input[1]) and node.input[1] in const_outputs


def is_qdq_wrapped(node: Any, producer: Dict[str, Any]) -> bool:
    def from_dequantize(tensor_name: str) -> bool:
        op = producer.get(tensor_name)
        return op is not None and op.op_type == "DequantizeLinear"

    if node.op_type in {"MatMul", "Gemm"}:
        if len(node.input) < 2:
            return False
        return from_dequantize(node.input[0]) and from_dequantize(node.input[1])
    return False


def inspect_linear_ops(
    model_path: str,
    target_node_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    ensure_onnx_importable()
    model = onnx.load(model_path)  # type: ignore[attr-defined]
    graph = model.graph
    producer: Dict[str, Any] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    const_outputs = collect_constant_outputs(graph)

    matmul_total = 0
    matmul_const_b = 0
    matmul_dynamic = 0
    matmul_qdq_wrapped = 0
    gemm_total = 0
    gemm_const_b = 0
    gemm_dynamic = 0
    gemm_qdq_wrapped = 0
    static_linear_node_names: List[str] = []
    unnamed_static_linear = 0
    custom_ops: Counter[str] = Counter()
    linear_node_names: set = set()

    for node in graph.node:
        domain = node.domain or "ai.onnx"
        if node.domain or node.op_type in {
            "MultiHeadAttention",
            "RotaryEmbedding",
            "SimplifiedLayerNormalization",
            "SkipSimplifiedLayerNormalization",
        }:
            custom_ops[f"{domain}::{node.op_type}"] += 1

        if node.op_type == "MatMul":
            matmul_total += 1
            if is_static_linear_op(node, const_outputs):
                matmul_const_b += 1
                if node.name:
                    static_linear_node_names.append(node.name)
                    linear_node_names.add(node.name)
                else:
                    unnamed_static_linear += 1
            else:
                matmul_dynamic += 1
                if node.name:
                    linear_node_names.add(node.name)
            if is_qdq_wrapped(node, producer):
                matmul_qdq_wrapped += 1
        elif node.op_type == "Gemm":
            gemm_total += 1
            if is_static_linear_op(node, const_outputs):
                gemm_const_b += 1
                if node.name:
                    static_linear_node_names.append(node.name)
                    linear_node_names.add(node.name)
                else:
                    unnamed_static_linear += 1
            else:
                gemm_dynamic += 1
                if node.name:
                    linear_node_names.add(node.name)
            if is_qdq_wrapped(node, producer):
                gemm_qdq_wrapped += 1

    effective_target_names = (
        set(item for item in target_node_names if item)
        if target_node_names is not None
        else set(static_linear_node_names)
    )
    target_linear_qdq_wrapped = 0
    for node in graph.node:
        if node.op_type not in {"MatMul", "Gemm"} or not node.name:
            continue
        if node.name in effective_target_names and is_qdq_wrapped(node, producer):
            target_linear_qdq_wrapped += 1

    return {
        "model_path": os.path.abspath(model_path),
        "graph_input_dtypes": graph_input_dtypes(graph),
        "custom_ops": dict(custom_ops),
        "matmul_total": matmul_total,
        "matmul_const_b": matmul_const_b,
        "matmul_dynamic": matmul_dynamic,
        "matmul_qdq_wrapped": matmul_qdq_wrapped,
        "gemm_total": gemm_total,
        "gemm_const_b": gemm_const_b,
        "gemm_dynamic": gemm_dynamic,
        "gemm_qdq_wrapped": gemm_qdq_wrapped,
        "static_linear_total": len(static_linear_node_names),
        "target_linear_total": len(effective_target_names),
        "target_linear_qdq_wrapped": target_linear_qdq_wrapped,
        "target_linear_all_in_qdq_set": target_linear_qdq_wrapped == len(effective_target_names),
        "target_linear_missing_in_model": sorted(effective_target_names - linear_node_names),
        "unnamed_static_linear": unnamed_static_linear,
        "_static_linear_node_names": static_linear_node_names,
    }


def quantize_one_model(
    model_input_path: str,
    model_output_path: str,
    reader: CalibrationDataReader,
    config: Dict[str, Any],
    nodes_to_quantize: Sequence[str],
) -> None:
    method_name = str(config["calibration_method"])
    calibrate_method = getattr(CalibrationMethod, method_name)
    extra_options: Dict[str, Any] = {
        "WeightSymmetric": True,
        "ActivationSymmetric": True,
        "MatMulConstBOnly": True,
    }
    exclude_ops = list(config.get("exclude_output_quant_ops", []))
    if exclude_ops:
        extra_options["OpTypesToExcludeOutputQuantization"] = exclude_ops
    percentile = config.get("percentile")
    if percentile is not None:
        extra_options["CalibPercentile"] = float(percentile)

    quantize_static(
        model_input=model_input_path,
        model_output=model_output_path,
        calibration_data_reader=reader,
        calibrate_method=calibrate_method,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
        nodes_to_quantize=list(nodes_to_quantize),
        extra_options=extra_options,
    )


def validate_fp32_quant_source_inputs(label: str, stats: Dict[str, Any]) -> None:
    bad_inputs = {
        name: dtype
        for name, dtype in stats.get("graph_input_dtypes", {}).items()
        if dtype in NON_FP32_FLOAT_INPUT_TYPES
    }
    if bad_inputs:
        raise ValueError(
            f"{label} quant-source model must keep fp32 floating inputs, "
            f"but found non-fp32 inputs: {bad_inputs}"
        )


def validate_quant_source_models(source_paths: ModelPaths, source_stats: Dict[str, Dict[str, Any]]) -> None:
    validate_fp32_quant_source_inputs("vision", source_stats["vision"])
    validate_fp32_quant_source_inputs("decoder", source_stats["decoder"])
    decoder_custom_ops = source_stats["decoder"].get("custom_ops", {})
    missing_ops = [name for name in sorted(DECODER_REQUIRED_CUSTOM_OPS) if decoder_custom_ops.get(name, 0) <= 0]
    if missing_ops:
        raise ValueError(
            "Decoder quant-source model looks rewritten or custom ops were stripped. "
            f"Expected to keep {missing_ops}, got path={source_paths.decoder}"
        )
    for label in ("vision", "decoder"):
        unnamed = int(source_stats[label].get("unnamed_static_linear", 0))
        if unnamed > 0:
            raise ValueError(
                f"{label} quant-source model has {unnamed} unnamed static MatMul/Gemm nodes; "
                "cannot safely restrict quantization to static linear ops."
            )


def cmd_quantize(args: argparse.Namespace) -> int:
    source_paths = resolve_quant_source_model_paths(args)
    asset_dir = resolve_asset_dir(args.asset_dir, args.model_dir)
    calib_json = os.path.abspath(args.calib_json)
    image_root = os.path.abspath(args.image_root)
    quant_out_dir = os.path.abspath(args.quant_out_dir)
    quant_summary = os.path.abspath(args.quant_summary_json)
    config_ids = parse_config_ids(args.configs)

    calib_records = load_json(calib_json)
    if not isinstance(calib_records, list):
        raise TypeError(f"Calibration json must be list, got {type(calib_records).__name__}")
    calib_records = subset_calib_records(
        records=calib_records,
        calib_limit=args.calib_limit,
        calib_per_type_override=args.calib_per_type_override,
        seed=args.seed,
    )
    components = parse_components(args.components)

    provider_info = resolve_runtime_providers(args.providers)
    print(f"[quantize] providers requested={provider_info['requested']}")
    print(f"[quantize] providers selected={provider_info['selected']}")
    print(f"[quantize] calib samples used={len(calib_records)}")
    print(f"[quantize] components={components}")

    runner: Optional[SmolVLM2OnnxRunner] = None
    if "decoder" in components:
        runner = SmolVLM2OnnxRunner(
            model_dir=asset_dir,
            vision_model_path=source_paths.vision,
            embed_model_path=source_paths.embed,
            decoder_model_path=source_paths.decoder,
            providers=provider_info["selected"],
        )

    ensure_dir(quant_out_dir)
    source_stats = {
        "vision": inspect_linear_ops(source_paths.vision),
        "decoder": inspect_linear_ops(source_paths.decoder),
    }
    validate_quant_source_models(source_paths, source_stats)
    vision_target_nodes = list(source_stats["vision"].pop("_static_linear_node_names", []))
    decoder_target_nodes = list(source_stats["decoder"].pop("_static_linear_node_names", []))
    source_stats["combined"] = {
        "matmul_total": source_stats["vision"]["matmul_total"] + source_stats["decoder"]["matmul_total"],
        "matmul_const_b": source_stats["vision"]["matmul_const_b"] + source_stats["decoder"]["matmul_const_b"],
        "matmul_dynamic": source_stats["vision"]["matmul_dynamic"] + source_stats["decoder"]["matmul_dynamic"],
        "gemm_total": source_stats["vision"]["gemm_total"] + source_stats["decoder"]["gemm_total"],
        "static_linear_total": (
            source_stats["vision"]["static_linear_total"] + source_stats["decoder"]["static_linear_total"]
        ),
    }
    print("[quantize] target policy=only static MatMul/Gemm with constant weight input")
    print(f"[quantize] source vision path={source_paths.vision}")
    print(f"[quantize] source decoder path={source_paths.decoder}")
    print(f"[quantize] source decoder custom ops={source_stats['decoder']['custom_ops']}")

    config_summaries: List[Dict[str, Any]] = []
    for config_id in config_ids:
        config = dict(QUANT_CONFIG_BY_ID[config_id])
        cfg_dir = os.path.join(quant_out_dir, config_id)
        ensure_dir(cfg_dir)
        vision_out = os.path.join(cfg_dir, "vision_encoder_int8_qdq.onnx")
        decoder_out = os.path.join(cfg_dir, "decoder_model_merged_int8_qdq.onnx")

        if args.resume and os.path.exists(vision_out) and os.path.exists(decoder_out):
            print(f"[quantize] {config_id} exists, skip quantization due to --resume")
        else:
            if "vision" in components:
                print(f"[quantize] {config_id}: quantizing vision")
                quantize_one_model(
                    model_input_path=source_paths.vision,
                    model_output_path=vision_out,
                    reader=build_vision_reader(
                        asset_dir=asset_dir,
                        calib_records=calib_records,
                        image_root=image_root,
                        progress_every=args.calib_progress_every,
                    ),
                    config=config,
                    nodes_to_quantize=vision_target_nodes,
                )

            if "decoder" in components:
                if runner is None:
                    raise RuntimeError("Decoder quantization requires initialized runner.")
                print(f"[quantize] {config_id}: quantizing decoder")
                quantize_one_model(
                    model_input_path=source_paths.decoder,
                    model_output_path=decoder_out,
                    reader=build_decoder_reader(
                        runner=runner,
                        calib_records=calib_records,
                        image_root=image_root,
                        progress_every=args.calib_progress_every,
                    ),
                    config=config,
                    nodes_to_quantize=decoder_target_nodes,
                )
        if "vision" in components:
            if not os.path.exists(vision_out):
                raise FileNotFoundError(f"vision output missing after quantization: {vision_out}")
            vision_stats = inspect_linear_ops(vision_out, target_node_names=vision_target_nodes)
        else:
            vision_stats = {
                **source_stats["vision"],
                "model_path": os.path.abspath(vision_out),
                "matmul_qdq_wrapped": 0,
                "gemm_qdq_wrapped": 0,
                "target_linear_qdq_wrapped": 0,
                "target_linear_all_in_qdq_set": False,
                "skipped": True,
            }

        if "decoder" in components:
            if not os.path.exists(decoder_out):
                raise FileNotFoundError(f"decoder output missing after quantization: {decoder_out}")
            decoder_stats = inspect_linear_ops(decoder_out, target_node_names=decoder_target_nodes)
        else:
            decoder_stats = {
                **source_stats["decoder"],
                "model_path": os.path.abspath(decoder_out),
                "matmul_qdq_wrapped": 0,
                "gemm_qdq_wrapped": 0,
                "target_linear_qdq_wrapped": 0,
                "target_linear_all_in_qdq_set": False,
                "skipped": True,
            }

        combined_target = 0
        combined_wrapped = 0
        if "vision" in components:
            combined_target += source_stats["vision"]["target_linear_total"]
            combined_wrapped += vision_stats["target_linear_qdq_wrapped"]
        if "decoder" in components:
            combined_target += source_stats["decoder"]["target_linear_total"]
            combined_wrapped += decoder_stats["target_linear_qdq_wrapped"]
        config_summaries.append(
            {
                "config": config,
                "output_dir": os.path.abspath(cfg_dir),
                "model_paths": {
                    "vision": os.path.abspath(vision_out),
                    "decoder": os.path.abspath(decoder_out),
                    "embed": source_paths.embed,
                },
                "coverage": {
                    "vision": vision_stats,
                    "decoder": decoder_stats,
                    "combined": {
                        "target_linear_total": combined_target,
                        "target_linear_qdq_wrapped": combined_wrapped,
                        "all_target_in_qdq_set": combined_wrapped == combined_target,
                    },
                },
            }
        )

    payload = {
        "generated_at": now_text(),
        "calib_json": calib_json,
        "image_root": image_root,
        "providers": provider_info,
        "asset_dir": asset_dir,
        "quant_source_model_paths": source_paths.__dict__,
        "target_policy": "Only quantize static MatMul/Gemm whose second input is constant; decoder source must preserve custom attention ops.",
        "source_coverage": source_stats,
        "quant_configs": config_summaries,
    }
    save_json(payload, quant_summary)
    print(f"[quantize] summary saved: {quant_summary}")
    print(
        "[quantize] source linear totals:",
        f"vision={source_stats['vision']['matmul_total']},",
        f"decoder={source_stats['decoder']['matmul_total']},",
        f"combined={source_stats['combined']['matmul_total']},",
        f"dynamic_matmul={source_stats['combined']['matmul_dynamic']},",
        f"static_targets={source_stats['combined']['static_linear_total']}",
    )
    return 0


@dataclass
class RunningScore:
    official_correct_by_type: Counter
    official_total_by_type: Counter
    all_correct_by_type: Counter
    all_total_by_type: Counter
    correct_by_dataset: Counter
    total_by_dataset: Counter
    overall_correct: int
    overall_total: int
    error_count: int


def empty_running_score() -> RunningScore:
    return RunningScore(
        official_correct_by_type=Counter(),
        official_total_by_type=Counter(),
        all_correct_by_type=Counter(),
        all_total_by_type=Counter(),
        correct_by_dataset=Counter(),
        total_by_dataset=Counter(),
        overall_correct=0,
        overall_total=0,
        error_count=0,
    )


def update_running_score(score: RunningScore, record: Dict[str, Any]) -> None:
    result = int(record.get("result", 0))
    item_type = str(record.get("type", ""))
    dataset = str(record.get("dataset_name", ""))
    score.overall_total += 1
    score.overall_correct += result
    if record.get("error"):
        score.error_count += 1
    if item_type:
        score.all_total_by_type[item_type] += 1
        score.all_correct_by_type[item_type] += result
    if dataset:
        score.total_by_dataset[dataset] += 1
        score.correct_by_dataset[dataset] += result
    if item_type in OCRBENCH_SCORE_KEYS:
        score.official_total_by_type[item_type] += 1
        score.official_correct_by_type[item_type] += result


def build_summary(score: RunningScore) -> Dict[str, Any]:
    official_score_by_type = {key: int(score.official_correct_by_type.get(key, 0)) for key in OCRBENCH_SCORE_KEYS}
    official_total_by_type = {key: int(score.official_total_by_type.get(key, 0)) for key in OCRBENCH_SCORE_KEYS}
    final_score = int(sum(official_score_by_type.values()))
    final_total = int(sum(official_total_by_type.values()))
    official_accuracy = (final_score / final_total) if final_total else 0.0
    overall_accuracy = (score.overall_correct / score.overall_total) if score.overall_total else 0.0

    type_names = sorted(score.all_total_by_type.keys())
    by_type = [
        {
            "type": name,
            "correct": int(score.all_correct_by_type[name]),
            "total": int(score.all_total_by_type[name]),
            "accuracy": (
                float(score.all_correct_by_type[name]) / float(score.all_total_by_type[name])
                if score.all_total_by_type[name]
                else 0.0
            ),
            "in_official_score": name in OCRBENCH_SCORE_KEYS,
        }
        for name in type_names
    ]

    dataset_names = sorted(score.total_by_dataset.keys())
    by_dataset = [
        {
            "dataset": name,
            "correct": int(score.correct_by_dataset[name]),
            "total": int(score.total_by_dataset[name]),
            "accuracy": (
                float(score.correct_by_dataset[name]) / float(score.total_by_dataset[name])
                if score.total_by_dataset[name]
                else 0.0
            ),
        }
        for name in dataset_names
    ]

    excluded_types = sorted(set(type_names) - set(OCRBENCH_SCORE_KEYS))
    return {
        "official": {
            "score_by_type": official_score_by_type,
            "total_by_type": official_total_by_type,
            "final_score": final_score,
            "final_total": final_total,
            "accuracy": official_accuracy,
        },
        "overall": {
            "correct": int(score.overall_correct),
            "total": int(score.overall_total),
            "accuracy": overall_accuracy,
            "error_count": int(score.error_count),
        },
        "excluded_from_official_types": excluded_types,
        "by_type": by_type,
        "by_dataset": by_dataset,
    }


def record_key(item: Dict[str, Any], index: int) -> str:
    if item.get("id") is not None:
        return str(item.get("id"))
    return f"__idx_{index}"


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            records.append(json.loads(text))
    return records


def run_eval_once(
    records: List[Dict[str, Any]],
    image_root: str,
    output_json: str,
    progress_jsonl: str,
    runner: SmolVLM2OnnxRunner,
    max_new_tokens: int,
    resume: bool,
    flush_every: int,
) -> Dict[str, Any]:
    ordered_keys = [record_key(item, idx) for idx, item in enumerate(records)]
    results_by_key: Dict[str, Dict[str, Any]] = {}

    if not resume:
        if os.path.exists(progress_jsonl):
            os.remove(progress_jsonl)
    else:
        for item in load_jsonl_records(progress_jsonl):
            key = record_key(item, 0)
            results_by_key[key] = item
        if not results_by_key and os.path.exists(output_json):
            old = load_json(output_json)
            for item in old.get("records", []):
                key = record_key(item, 0)
                results_by_key[key] = item

    score = empty_running_score()
    for key in ordered_keys:
        if key in results_by_key:
            update_running_score(score, results_by_key[key])

    pending = 0
    for idx, item in enumerate(tqdm(records, desc=f"eval-{os.path.basename(output_json)}")):
        key = record_key(item, idx)
        if key in results_by_key:
            continue
        rec = dict(item)
        image_path = os.path.join(image_root, item["image_path"])
        try:
            prediction, _ = runner.decode_tokens(
                image_path=image_path,
                question=item["question"],
                max_new_tokens=max_new_tokens,
            )
            rec["predict"] = prediction
            rec["result"] = evaluate_prediction(rec, prediction)
        except Exception as exc:  # pragma: no cover - runtime handling
            rec["predict"] = f"ERROR: {exc}"
            rec["result"] = 0
            rec["error"] = f"{type(exc).__name__}: {exc}"

        append_jsonl(progress_jsonl, rec)
        results_by_key[key] = rec
        update_running_score(score, rec)
        pending += 1

        if pending >= flush_every:
            pending = 0
            save_json(
                {
                    "summary": build_summary(score),
                    "records": [results_by_key[k] for k in ordered_keys if k in results_by_key],
                },
                output_json,
            )

    payload = {
        "summary": build_summary(score),
        "records": [results_by_key[k] for k in ordered_keys if k in results_by_key],
    }
    save_json(payload, output_json)
    return payload


def make_eval_output_name(dataset_tag: str, model_mode: str, config_id: str) -> str:
    if model_mode == "fp32":
        return f"{dataset_tag}_fp32.json"
    return f"{dataset_tag}_int8_{config_id}.json"


def cmd_eval(args: argparse.Namespace) -> int:
    input_json = os.path.abspath(args.input_json)
    image_root = os.path.abspath(args.image_root)
    eval_out_dir = os.path.abspath(args.eval_out_dir)
    ensure_dir(eval_out_dir)

    fp32_paths = resolve_fp32_model_paths(args)
    source_paths = resolve_quant_source_model_paths(args)
    asset_dir = resolve_asset_dir(args.asset_dir, args.model_dir)
    if args.model_mode == "fp32":
        model_paths = fp32_paths
    else:
        if not args.config_id:
            raise ValueError("--config-id is required for int8 mode.")
        model_paths = resolve_int8_model_paths(args.quant_out_dir, args.config_id, source_paths.embed)

    dataset_tag = args.dataset_tag or os.path.splitext(os.path.basename(input_json))[0]
    output_json = (
        os.path.abspath(args.output_json)
        if args.output_json
        else os.path.join(
            eval_out_dir,
            make_eval_output_name(
                dataset_tag=dataset_tag,
                model_mode=args.model_mode,
                config_id=args.config_id or "",
            ),
        )
    )
    progress_jsonl = (
        os.path.abspath(args.progress_jsonl)
        if args.progress_jsonl
        else f"{output_json}.progress.jsonl"
    )

    provider_info = resolve_runtime_providers(args.providers)
    print(f"[eval] providers requested={provider_info['requested']}")
    print(f"[eval] providers selected={provider_info['selected']}")

    records = load_json(input_json)
    if not isinstance(records, list):
        raise TypeError(f"Input json must be list, got {type(records).__name__}")
    if args.offset > 0:
        records = records[args.offset:]
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[eval] items={len(records)}, image_root={image_root}")
    print(f"[eval] output={output_json}")
    print(f"[eval] progress={progress_jsonl}")

    runner = SmolVLM2OnnxRunner(
        model_dir=asset_dir,
        vision_model_path=model_paths.vision,
        embed_model_path=model_paths.embed,
        decoder_model_path=model_paths.decoder,
        providers=provider_info["selected"],
    )
    payload = run_eval_once(
        records=records,
        image_root=image_root,
        output_json=output_json,
        progress_jsonl=progress_jsonl,
        runner=runner,
        max_new_tokens=args.max_new_tokens,
        resume=args.resume,
        flush_every=max(1, args.flush_every),
    )
    summary = payload["summary"]
    print(
        "[eval] official:",
        f"{summary['official']['final_score']}/{summary['official']['final_total']}",
        f"({summary['official']['accuracy']:.6f})",
    )
    print(
        "[eval] overall:",
        f"{summary['overall']['correct']}/{summary['overall']['total']}",
        f"({summary['overall']['accuracy']:.6f})",
    )
    print(f"[eval] saved {output_json}")
    return 0


def load_eval_payload(path: str) -> Dict[str, Any]:
    payload = load_json(path)
    if "summary" not in payload:
        raise ValueError(f"Missing summary in {path}")
    return payload


def find_type_metric(by_type: Iterable[Dict[str, Any]], item_type: str) -> Dict[str, Any]:
    for item in by_type:
        if item.get("type") == item_type:
            return item
    return {"type": item_type, "correct": 0, "total": 0, "accuracy": 0.0, "in_official_score": False}


def find_dataset_metric(by_dataset: Iterable[Dict[str, Any]], dataset: str) -> Dict[str, Any]:
    for item in by_dataset:
        if item.get("dataset") == dataset:
            return item
    return {"dataset": dataset, "correct": 0, "total": 0, "accuracy": 0.0}


def build_report_markdown(
    dev_fp32_payload: Dict[str, Any],
    dev_int8_payloads: List[Dict[str, Any]],
    full_fp32_payload: Dict[str, Any],
    full_int8_payload: Dict[str, Any],
    selected_config_id: str,
    pass_threshold: float,
    quant_summary: Dict[str, Any],
    paths: Dict[str, str],
) -> str:
    full_fp32_official = float(full_fp32_payload["summary"]["official"]["accuracy"])
    full_int8_official = float(full_int8_payload["summary"]["official"]["accuracy"])
    official_drop = full_fp32_official - full_int8_official
    pass_ok = official_drop < pass_threshold

    lines: List[str] = []
    lines.append("# AICAS FullTest ORT QDQ INT8 报告")
    lines.append("")
    lines.append(f"- 生成时间: {now_text()}")
    lines.append(f"- 输入集: `{paths['input_json']}`")
    lines.append(f"- 图片根目录: `{paths['image_root']}`")
    lines.append(f"- 量化模型目录: `{paths['quant_out_dir']}`")
    lines.append(f"- 评测结果目录: `{paths['eval_out_dir']}`")
    lines.append(f"- 量化策略: `{quant_summary.get('target_policy', 'Only static MatMul/Gemm')}`")
    lines.append("")
    lines.append("## 量化覆盖统计")
    lines.append("")
    source_coverage = quant_summary.get("source_coverage", {})
    lines.append(
        "- 原始线性层统计: "
        f"vision={source_coverage.get('vision', {}).get('matmul_total', 0)}, "
        f"decoder={source_coverage.get('decoder', {}).get('matmul_total', 0)}, "
        f"combined={source_coverage.get('combined', {}).get('matmul_total', 0)}, "
        f"dynamic_matmul={source_coverage.get('combined', {}).get('matmul_dynamic', 0)}, "
        f"static_target={source_coverage.get('combined', {}).get('static_linear_total', 0)}"
    )
    lines.append(
        "- Decoder custom op 保留统计: "
        f"{source_coverage.get('decoder', {}).get('custom_ops', {})}"
    )
    lines.append("")
    lines.append("| 配置 | 方法 | 静态线性目标数 | QDQ包裹数 | 全覆盖 |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for item in quant_summary.get("quant_configs", []):
        cfg = item["config"]
        cov = item["coverage"]["combined"]
        lines.append(
            f"| {cfg['config_id']} | {cfg['name']} | "
            f"{cov.get('target_linear_total', 0)} | {cov.get('target_linear_qdq_wrapped', 0)} | "
            f"{'是' if cov['all_target_in_qdq_set'] else '否'} |"
        )
    lines.append("")
    lines.append("## Dev_260 Sweep")
    lines.append("")
    lines.append("| 配置 | 官方分数 | 官方总数 | 官方准确率 | 全记录准确率 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for item in dev_int8_payloads:
        summary = item["summary"]
        cfg = item.get("_config_id", "")
        lines.append(
            f"| {cfg} | {summary['official']['final_score']} | {summary['official']['final_total']} | "
            f"{summary['official']['accuracy']:.6f} | {summary['overall']['accuracy']:.6f} |"
        )
    dev_fp32_summary = dev_fp32_payload["summary"]
    lines.append(
        f"| fp32 | {dev_fp32_summary['official']['final_score']} | {dev_fp32_summary['official']['final_total']} | "
        f"{dev_fp32_summary['official']['accuracy']:.6f} | {dev_fp32_summary['overall']['accuracy']:.6f} |"
    )
    lines.append("")
    lines.append(f"- Dev 最优配置: `{selected_config_id}`")
    lines.append("")
    lines.append("## FullTest 结论")
    lines.append("")
    lines.append("| 模式 | 官方分数 | 官方总数 | 官方准确率 | 全记录准确率 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, payload in [("fp32", full_fp32_payload), (f"int8({selected_config_id})", full_int8_payload)]:
        summary = payload["summary"]
        lines.append(
            f"| {name} | {summary['official']['final_score']} | {summary['official']['final_total']} | "
            f"{summary['official']['accuracy']:.6f} | {summary['overall']['accuracy']:.6f} |"
        )
    lines.append("")
    lines.append(f"- 绝对百分点下降: `{official_drop * 100.0:.4f}`")
    lines.append(f"- 通过阈值: `< {pass_threshold * 100.0:.2f}` 百分点")
    lines.append(f"- 是否通过: `{'PASS' if pass_ok else 'FAIL'}`")
    lines.append("")

    excluded = sorted(
        set(full_fp32_payload["summary"]["excluded_from_official_types"])
        | set(full_int8_payload["summary"]["excluded_from_official_types"])
    )
    lines.append(
        "- 官方 Final Score 不计入的类型: "
        + (", ".join(excluded) if excluded else "无")
    )
    if excluded and EXCLUDED_FROM_OFFICIAL_TYPES.issubset(set(excluded)):
        lines.append(
            "- 补充说明: Chinese / Occluded / Semantic Text Recognition 在官方分数中不计入。"
        )
    lines.append("")

    lines.append("## FullTest 分类型结果")
    lines.append("")
    fp32_types = full_fp32_payload["summary"]["by_type"]
    int8_types = full_int8_payload["summary"]["by_type"]
    all_types = sorted({item["type"] for item in fp32_types} | {item["type"] for item in int8_types})
    lines.append("| Type | FP32(正确/总数) | FP32准确率 | INT8(正确/总数) | INT8准确率 | 官方计分 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for item_type in all_types:
        a = find_type_metric(fp32_types, item_type)
        b = find_type_metric(int8_types, item_type)
        lines.append(
            f"| {item_type} | {a['correct']}/{a['total']} | {a['accuracy']:.6f} | "
            f"{b['correct']}/{b['total']} | {b['accuracy']:.6f} | "
            f"{'是' if a.get('in_official_score') or b.get('in_official_score') else '否'} |"
        )
    lines.append("")

    lines.append("## FullTest 分数据集结果")
    lines.append("")
    fp32_ds = full_fp32_payload["summary"]["by_dataset"]
    int8_ds = full_int8_payload["summary"]["by_dataset"]
    all_ds = sorted({item["dataset"] for item in fp32_ds} | {item["dataset"] for item in int8_ds})
    lines.append("| Dataset | FP32(正确/总数) | FP32准确率 | INT8(正确/总数) | INT8准确率 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name in all_ds:
        a = find_dataset_metric(fp32_ds, name)
        b = find_dataset_metric(int8_ds, name)
        lines.append(
            f"| {name} | {a['correct']}/{a['total']} | {a['accuracy']:.6f} | "
            f"{b['correct']}/{b['total']} | {b['accuracy']:.6f} |"
        )
    lines.append("")

    if not pass_ok:
        lines.append("## 未通过时的候选回退方向")
        lines.append("")
        lines.append("- 保持“仅静态 MatMul/Gemm”策略，先微调校准集与校准方法参数。")
        lines.append("- 如仍不达标，再缩小静态线性层量化范围，而不是重新量化 attention。")
        lines.append("- 最后再评估是否需要进入局部 rewrite 路线。")
        lines.append("")

    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    report_path = os.path.abspath(args.report_path)
    quant_summary_path = os.path.abspath(args.quant_summary_json)
    dev_fp32_path = os.path.abspath(args.dev_fp32_json)
    full_fp32_path = os.path.abspath(args.full_fp32_json)
    full_int8_path = os.path.abspath(args.full_int8_json)
    selected_config_id = args.selected_config_id

    dev_int8_paths = [os.path.abspath(item) for item in args.dev_int8_jsons.split(",") if item.strip()]
    if not dev_int8_paths:
        raise ValueError("--dev-int8-jsons is required")
    dev_int8_payloads = []
    for path in dev_int8_paths:
        payload = load_eval_payload(path)
        basename = os.path.basename(path)
        cfg = ""
        for config_id in QUANT_CONFIG_BY_ID:
            if f"_{config_id}" in basename:
                cfg = config_id
                break
        payload["_config_id"] = cfg
        dev_int8_payloads.append(payload)

    quant_summary = load_json(quant_summary_path)
    dev_fp32_payload = load_eval_payload(dev_fp32_path)
    full_fp32_payload = load_eval_payload(full_fp32_path)
    full_int8_payload = load_eval_payload(full_int8_path)

    md = build_report_markdown(
        dev_fp32_payload=dev_fp32_payload,
        dev_int8_payloads=dev_int8_payloads,
        full_fp32_payload=full_fp32_payload,
        full_int8_payload=full_int8_payload,
        selected_config_id=selected_config_id,
        pass_threshold=args.pass_threshold,
        quant_summary=quant_summary,
        paths={
            "input_json": os.path.abspath(args.input_json),
            "image_root": os.path.abspath(args.image_root),
            "quant_out_dir": os.path.abspath(args.quant_out_dir),
            "eval_out_dir": os.path.abspath(args.eval_out_dir),
        },
    )
    ensure_dir(os.path.dirname(report_path) or ".")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[report] saved {report_path}")
    return 0


def choose_best_config(dev_int8_payloads: List[Dict[str, Any]]) -> str:
    if not dev_int8_payloads:
        raise ValueError("No dev int8 payloads to choose best config.")

    def key_fn(item: Dict[str, Any]) -> Tuple[float, float]:
        summary = item["summary"]
        return (
            float(summary["official"]["accuracy"]),
            float(summary["overall"]["accuracy"]),
        )

    best = max(dev_int8_payloads, key=key_fn)
    return str(best["_config_id"])


def run_eval_for_mode(
    args: argparse.Namespace,
    input_json: str,
    dataset_tag: str,
    model_mode: str,
    config_id: str,
) -> Tuple[str, Dict[str, Any]]:
    output_json = os.path.join(
        os.path.abspath(args.eval_out_dir),
        make_eval_output_name(dataset_tag=dataset_tag, model_mode=model_mode, config_id=config_id),
    )
    eval_args = argparse.Namespace(**vars(args))
    eval_args.input_json = input_json
    eval_args.dataset_tag = dataset_tag
    eval_args.model_mode = model_mode
    eval_args.config_id = config_id
    eval_args.output_json = output_json
    eval_args.progress_jsonl = f"{output_json}.progress.jsonl"
    cmd_eval(eval_args)
    return output_json, load_eval_payload(output_json)


def cmd_run_all(args: argparse.Namespace) -> int:
    ensure_dir(os.path.abspath(args.eval_out_dir))
    ensure_dir(os.path.abspath(args.quant_out_dir))

    prepare_args = argparse.Namespace(**vars(args))
    prepare_args.dev_json_out = args.dev_json_out
    prepare_args.calib_json_out = args.calib_json_out
    prepare_args.split_summary_json = args.split_summary_json
    prepare_args.dev_per_type = args.dev_per_type
    prepare_args.calib_per_type = args.calib_per_type
    cmd_prepare(prepare_args)

    quant_args = argparse.Namespace(**vars(args))
    quant_args.calib_json = args.calib_json_out
    quant_args.quant_summary_json = args.quant_summary_json
    cmd_quantize(quant_args)

    dev_fp32_path, dev_fp32_payload = run_eval_for_mode(
        args=args,
        input_json=os.path.abspath(args.dev_json_out),
        dataset_tag="dev_260",
        model_mode="fp32",
        config_id="",
    )

    config_ids = parse_config_ids(args.configs)
    dev_int8_paths: List[str] = []
    dev_int8_payloads: List[Dict[str, Any]] = []
    for config_id in config_ids:
        path, payload = run_eval_for_mode(
            args=args,
            input_json=os.path.abspath(args.dev_json_out),
            dataset_tag="dev_260",
            model_mode="int8",
            config_id=config_id,
        )
        payload["_config_id"] = config_id
        dev_int8_paths.append(path)
        dev_int8_payloads.append(payload)

    best_config_id = choose_best_config(dev_int8_payloads)
    print(f"[run-all] best config on dev_260: {best_config_id}")

    full_fp32_path, full_fp32_payload = run_eval_for_mode(
        args=args,
        input_json=os.path.abspath(args.input_json),
        dataset_tag="fulltest",
        model_mode="fp32",
        config_id="",
    )
    full_int8_path, full_int8_payload = run_eval_for_mode(
        args=args,
        input_json=os.path.abspath(args.input_json),
        dataset_tag="fulltest",
        model_mode="int8",
        config_id=best_config_id,
    )

    report_args = argparse.Namespace(**vars(args))
    report_args.dev_fp32_json = dev_fp32_path
    report_args.dev_int8_jsons = ",".join(dev_int8_paths)
    report_args.full_fp32_json = full_fp32_path
    report_args.full_int8_json = full_int8_path
    report_args.selected_config_id = best_config_id
    cmd_report(report_args)

    fp32_official = float(full_fp32_payload["summary"]["official"]["accuracy"])
    int8_official = float(full_int8_payload["summary"]["official"]["accuracy"])
    drop = fp32_official - int8_official
    print(
        "[run-all] FullTest official:",
        f"fp32={fp32_official:.6f}",
        f"int8={int8_official:.6f}",
        f"drop={drop:.6f}",
    )
    if drop >= args.pass_threshold:
        print(
            "[run-all] FAIL: drop not less than threshold. "
            "Stopped after report generation; no extra static-target fallback executed."
        )
        return 2
    print("[run-all] PASS")
    return 0


def add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--asset-dir", default=DEFAULT_ASSET_DIR)
    parser.add_argument("--vision-model", default="")
    parser.add_argument("--embed-model", default="")
    parser.add_argument("--decoder-model", default="")
    parser.add_argument("--quant-source-vision-model", default="")
    parser.add_argument("--quant-source-embed-model", default="")
    parser.add_argument("--quant-source-decoder-model", default="")


def add_shared_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AICAS FullTest ORT QDQ INT8 pipeline (prepare / quantize / eval / report / run-all)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare")
    add_shared_data_args(p_prepare)
    p_prepare.add_argument("--dev-json-out", default=DEFAULT_DEV_JSON)
    p_prepare.add_argument("--calib-json-out", default=DEFAULT_CALIB_JSON)
    p_prepare.add_argument("--split-summary-json", default=DEFAULT_SPLIT_SUMMARY_JSON)
    p_prepare.add_argument("--dev-per-type", type=int, default=DEFAULT_DEV_PER_TYPE)
    p_prepare.add_argument("--calib-per-type", type=int, default=DEFAULT_CALIB_PER_TYPE)

    p_quant = sub.add_parser("quantize")
    add_shared_data_args(p_quant)
    add_shared_model_args(p_quant)
    p_quant.add_argument("--calib-json", default=DEFAULT_CALIB_JSON)
    p_quant.add_argument("--quant-out-dir", default=DEFAULT_QUANT_OUT_DIR)
    p_quant.add_argument("--quant-summary-json", default=DEFAULT_QUANT_SUMMARY_JSON)
    p_quant.add_argument("--providers", default=DEFAULT_PROVIDER_TEXT)
    p_quant.add_argument("--configs", default="cfg1,cfg2,cfg3,cfg4")
    p_quant.add_argument("--components", default="vision,decoder")
    p_quant.add_argument("--calib-limit", type=int, default=0)
    p_quant.add_argument("--calib-per-type-override", type=int, default=0)
    p_quant.add_argument("--calib-progress-every", type=int, default=DEFAULT_CALIB_PROGRESS_EVERY)

    p_eval = sub.add_parser("eval")
    add_shared_data_args(p_eval)
    add_shared_model_args(p_eval)
    p_eval.add_argument("--quant-out-dir", default=DEFAULT_QUANT_OUT_DIR)
    p_eval.add_argument("--eval-out-dir", default=DEFAULT_EVAL_OUT_DIR)
    p_eval.add_argument("--output-json", default="")
    p_eval.add_argument("--progress-jsonl", default="")
    p_eval.add_argument("--dataset-tag", default="")
    p_eval.add_argument("--model-mode", choices=["fp32", "int8"], required=True)
    p_eval.add_argument("--config-id", default="")
    p_eval.add_argument("--providers", default=DEFAULT_PROVIDER_TEXT)
    p_eval.add_argument("--max-new-tokens", type=int, default=100)
    p_eval.add_argument("--flush-every", type=int, default=DEFAULT_FLUSH_EVERY)
    p_eval.add_argument("--offset", type=int, default=0)
    p_eval.add_argument("--limit", type=int, default=0)

    p_report = sub.add_parser("report")
    add_shared_data_args(p_report)
    p_report.add_argument("--quant-out-dir", default=DEFAULT_QUANT_OUT_DIR)
    p_report.add_argument("--eval-out-dir", default=DEFAULT_EVAL_OUT_DIR)
    p_report.add_argument("--quant-summary-json", default=DEFAULT_QUANT_SUMMARY_JSON)
    p_report.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    p_report.add_argument("--dev-fp32-json", required=True)
    p_report.add_argument("--dev-int8-jsons", required=True)
    p_report.add_argument("--full-fp32-json", required=True)
    p_report.add_argument("--full-int8-json", required=True)
    p_report.add_argument("--selected-config-id", required=True)
    p_report.add_argument("--pass-threshold", type=float, default=DEFAULT_PASS_THRESHOLD)

    p_run = sub.add_parser("run-all")
    add_shared_data_args(p_run)
    add_shared_model_args(p_run)
    p_run.add_argument("--dev-json-out", default=DEFAULT_DEV_JSON)
    p_run.add_argument("--calib-json-out", default=DEFAULT_CALIB_JSON)
    p_run.add_argument("--split-summary-json", default=DEFAULT_SPLIT_SUMMARY_JSON)
    p_run.add_argument("--dev-per-type", type=int, default=DEFAULT_DEV_PER_TYPE)
    p_run.add_argument("--calib-per-type", type=int, default=DEFAULT_CALIB_PER_TYPE)
    p_run.add_argument("--quant-out-dir", default=DEFAULT_QUANT_OUT_DIR)
    p_run.add_argument("--quant-summary-json", default=DEFAULT_QUANT_SUMMARY_JSON)
    p_run.add_argument("--eval-out-dir", default=DEFAULT_EVAL_OUT_DIR)
    p_run.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    p_run.add_argument("--providers", default=DEFAULT_PROVIDER_TEXT)
    p_run.add_argument("--configs", default="cfg1,cfg2,cfg3,cfg4")
    p_run.add_argument("--components", default="vision,decoder")
    p_run.add_argument("--calib-limit", type=int, default=0)
    p_run.add_argument("--calib-per-type-override", type=int, default=0)
    p_run.add_argument("--calib-progress-every", type=int, default=DEFAULT_CALIB_PROGRESS_EVERY)
    p_run.add_argument("--max-new-tokens", type=int, default=100)
    p_run.add_argument("--flush-every", type=int, default=DEFAULT_FLUSH_EVERY)
    p_run.add_argument("--pass-threshold", type=float, default=DEFAULT_PASS_THRESHOLD)
    p_run.add_argument("--offset", type=int, default=0)
    p_run.add_argument("--limit", type=int, default=0)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "quantize":
        return cmd_quantize(args)
    if args.command == "eval":
        return cmd_eval(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "run-all":
        return cmd_run_all(args)
    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
