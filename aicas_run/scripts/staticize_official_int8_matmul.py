#!/usr/bin/env python3
import argparse
import copy
import math
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from transformers import AutoProcessor

from smolvlm2_onnx_lib import SmolVLM2OnnxRunner, ensure_dir, load_json, save_json


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AICAS_ROOT = os.path.dirname(SCRIPT_DIR)
if AICAS_ROOT not in sys.path:
    sys.path.insert(0, AICAS_ROOT)
DEFAULT_IMAGE_ROOT = os.path.join(AICAS_ROOT, "data")
DEFAULT_MODEL_DIR = AICAS_ROOT


@dataclass(frozen=True)
class QuantizerSpec:
    source_input_name: str
    quantized_output_name: str
    scale_name: str
    zero_point_name: str
    original_node_name: str


@dataclass(frozen=True)
class MatMulRewriteSpec:
    matmul_node_name: str
    activation_quantizer_name: str
    activation_quantized_name: str
    activation_scale_name: str
    activation_zero_point_name: str
    weight_quantized_name: str
    weight_scale_name: str
    weight_zero_point_name: str
    output_float_name: str
    qlinear_output_name: str
    cast_node_name: str
    scale_mul_node_name: str
    output_mul_node_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze official dynamic-int8 MatMulInteger chains into static "
            "QuantizeLinear -> QLinearMatMul -> DequantizeLinear."
        )
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--asset-dir", default="")
    parser.add_argument("--calib-json", required=True)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--components", default="vision,decoder")
    parser.add_argument("--providers", default="CPUExecutionProvider")
    parser.add_argument("--calib-limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--vision-model", default="")
    parser.add_argument("--embed-model", default="")
    parser.add_argument("--decoder-model", default="")
    parser.add_argument("--prep-vision-model", default="")
    parser.add_argument("--prep-embed-model", default="")
    parser.add_argument("--prep-decoder-model", default="")
    parser.add_argument("--vision-probe-model", default="")
    parser.add_argument("--decoder-probe-model", default="")
    return parser.parse_args()


def parse_components(text: str) -> List[str]:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if not parts:
        raise ValueError("No component provided.")
    valid = {"vision", "decoder"}
    for part in parts:
        if part not in valid:
            raise ValueError(f"Unsupported component: {part}")
    return parts


def parse_providers(text: str) -> List[str]:
    providers = [item.strip() for item in text.split(",") if item.strip()]
    return providers if providers else ["CPUExecutionProvider"]


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


def collect_model_maps(model: onnx.ModelProto) -> Tuple[Dict[str, onnx.NodeProto], Dict[str, List[onnx.NodeProto]]]:
    producer: Dict[str, onnx.NodeProto] = {}
    consumers: Dict[str, List[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for output_name in node.output:
            if output_name:
                producer[output_name] = node
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)
    return producer, consumers


def ensure_single_consumer(
    consumers: Dict[str, List[onnx.NodeProto]],
    tensor_name: str,
    expected_op_type: str,
    owner: str,
) -> onnx.NodeProto:
    users = consumers.get(tensor_name, [])
    if len(users) != 1 or users[0].op_type != expected_op_type:
        raise ValueError(
            f"{owner}: expect {tensor_name} to have single {expected_op_type} consumer, "
            f"got {[node.op_type for node in users]}"
        )
    return users[0]


def extract_weight_scale_name(
    scale_mul_node: onnx.NodeProto,
    activation_scale_name: str,
    owner: str,
) -> str:
    candidates = [name for name in scale_mul_node.input if name and name != activation_scale_name]
    if len(candidates) != 1:
        raise ValueError(
            f"{owner}: cannot extract weight scale from {scale_mul_node.name}, inputs={list(scale_mul_node.input)}"
        )
    return candidates[0]


def extract_rewrite_specs(model: onnx.ModelProto) -> Tuple[Dict[str, QuantizerSpec], List[MatMulRewriteSpec]]:
    producer, consumers = collect_model_maps(model)
    quantizers: Dict[str, QuantizerSpec] = {}
    rewrites: List[MatMulRewriteSpec] = []

    for node in model.graph.node:
        if node.op_type != "MatMulInteger":
            continue
        if len(node.input) != 4 or len(node.output) != 1:
            raise ValueError(f"{node.name}: unexpected MatMulInteger arity.")

        dynamic_quant = producer.get(node.input[0])
        if dynamic_quant is None or dynamic_quant.op_type != "DynamicQuantizeLinear":
            raise ValueError(f"{node.name}: input A is not from DynamicQuantizeLinear.")
        if len(dynamic_quant.input) != 1 or len(dynamic_quant.output) != 3:
            raise ValueError(f"{node.name}: unexpected DynamicQuantizeLinear shape.")

        cast_node = ensure_single_consumer(consumers, node.output[0], "Cast", node.name)
        if len(cast_node.output) != 1:
            raise ValueError(f"{node.name}: unexpected Cast outputs.")
        output_mul = ensure_single_consumer(consumers, cast_node.output[0], "Mul", node.name)
        if len(output_mul.output) != 1:
            raise ValueError(f"{node.name}: unexpected output Mul outputs.")

        scale_mul_input = next(name for name in output_mul.input if name != cast_node.output[0])
        scale_mul = producer.get(scale_mul_input)
        if scale_mul is None or scale_mul.op_type != "Mul":
            raise ValueError(f"{node.name}: output scale chain is not Mul.")

        quantizer = QuantizerSpec(
            source_input_name=dynamic_quant.input[0],
            quantized_output_name=dynamic_quant.output[0],
            scale_name=dynamic_quant.output[1],
            zero_point_name=dynamic_quant.output[2],
            original_node_name=dynamic_quant.name,
        )
        quantizers[dynamic_quant.name] = quantizer

        rewrites.append(
            MatMulRewriteSpec(
                matmul_node_name=node.name,
                activation_quantizer_name=dynamic_quant.name,
                activation_quantized_name=dynamic_quant.output[0],
                activation_scale_name=dynamic_quant.output[1],
                activation_zero_point_name=dynamic_quant.output[2],
                weight_quantized_name=node.input[1],
                weight_scale_name=extract_weight_scale_name(scale_mul, dynamic_quant.output[1], node.name),
                weight_zero_point_name=node.input[3],
                output_float_name=output_mul.output[0],
                qlinear_output_name=node.output[0],
                cast_node_name=cast_node.name,
                scale_mul_node_name=scale_mul.name,
                output_mul_node_name=output_mul.name,
            )
        )
    return quantizers, rewrites


def make_reduce_output_name(prefix: str, idx: int, kind: str) -> str:
    return f"__staticize_{prefix}_{idx}_{kind}"


def build_reduce_model(
    model: onnx.ModelProto,
    tensor_names: Sequence[str],
    prefix: str,
) -> Tuple[onnx.ModelProto, Dict[str, Tuple[str, str]]]:
    reduced_model = copy.deepcopy(model)
    output_map: Dict[str, Tuple[str, str]] = {}
    existing_outputs = {item.name for item in reduced_model.graph.output}
    nodes_to_add: List[onnx.NodeProto] = []
    outputs_to_add: List[onnx.ValueInfoProto] = []

    for idx, tensor_name in enumerate(sorted(set(tensor_names))):
        min_name = make_reduce_output_name(prefix, idx, "min")
        max_name = make_reduce_output_name(prefix, idx, "max")
        if min_name in existing_outputs or max_name in existing_outputs:
            raise ValueError(f"Temporary reduce output name collision for {tensor_name}.")
        nodes_to_add.append(
            helper.make_node(
                "ReduceMin",
                [tensor_name],
                [min_name],
                keepdims=0,
                name=f"{min_name}_node",
            )
        )
        nodes_to_add.append(
            helper.make_node(
                "ReduceMax",
                [tensor_name],
                [max_name],
                keepdims=0,
                name=f"{max_name}_node",
            )
        )
        outputs_to_add.append(helper.make_tensor_value_info(min_name, TensorProto.FLOAT, []))
        outputs_to_add.append(helper.make_tensor_value_info(max_name, TensorProto.FLOAT, []))
        output_map[tensor_name] = (min_name, max_name)

    reduced_model.graph.node.extend(nodes_to_add)
    reduced_model.graph.output.extend(outputs_to_add)
    return reduced_model, output_map


def to_scalar_float(value: Any) -> float:
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def compute_uint8_quant_params(min_value: float, max_value: float) -> Tuple[float, int]:
    min_value = min(min_value, 0.0)
    max_value = max(max_value, 0.0)
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        raise ValueError(f"Invalid quant range: min={min_value}, max={max_value}")
    if max_value - min_value < 1e-12:
        return 1.0, 0
    scale = float((max_value - min_value) / 255.0)
    zero_point = int(round(-min_value / scale))
    zero_point = min(255, max(0, zero_point))
    return scale, zero_point


def make_float_initializer(name: str, value: float) -> onnx.TensorProto:
    array = np.asarray(value, dtype=np.float32)
    return numpy_helper.from_array(array, name=name)


def make_uint8_initializer(name: str, value: int) -> onnx.TensorProto:
    array = np.asarray(value, dtype=np.uint8)
    return numpy_helper.from_array(array, name=name)


def build_vision_inputs(
    processor: AutoProcessor,
    image_path: str,
) -> Dict[str, np.ndarray]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    try:
        prepared = processor.image_processor.preprocess(
            images=[image],
            return_tensors="np",
        )
    finally:
        image.close()
    return {
        "pixel_values": prepared["pixel_values"].astype(np.float32, copy=False),
        "pixel_attention_mask": prepared["pixel_attention_mask"].astype(bool, copy=False),
    }


def build_decoder_inputs(
    runner: SmolVLM2OnnxRunner,
    image_path: str,
    question: str,
) -> Dict[str, np.ndarray]:
    state = runner.build_prompt_state(image_path=image_path, question=question)
    return {
        "inputs_embeds": state["merged_embeds"].astype(runner.decoder_inputs_embeds_dtype, copy=False),
        "attention_mask": state["attention_mask"].astype(runner.decoder_attention_mask_dtype, copy=False),
        "position_ids": state["position_ids"].astype(runner.decoder_position_ids_dtype, copy=False),
        **runner.zero_past_key_values(batch_size=1),
    }


def run_reduce_calibration(
    session: ort.InferenceSession,
    output_map: Dict[str, Tuple[str, str]],
    records: Sequence[Dict[str, Any]],
    feed_builder: Any,
    image_root: str,
    progress_every: int,
    component: str,
) -> Dict[str, Dict[str, float]]:
    output_names: List[str] = []
    output_to_tensor: Dict[str, Tuple[str, str]] = {}
    for tensor_name, (min_name, max_name) in output_map.items():
        output_names.extend([min_name, max_name])
        output_to_tensor[min_name] = (tensor_name, "min")
        output_to_tensor[max_name] = (tensor_name, "max")

    stats = {
        tensor_name: {"min": float("inf"), "max": float("-inf")}
        for tensor_name in output_map.keys()
    }

    total = len(records)
    for index, item in enumerate(records, start=1):
        if index == 1 or index % max(1, progress_every) == 0 or index == total:
            print(f"[calib:{component}] {index}/{total} id={item.get('id', '')}")
        image_path = os.path.join(image_root, item["image_path"])
        feed = feed_builder(image_path=image_path, question=item.get("question", ""))
        values = session.run(output_names, feed)
        for output_name, value in zip(output_names, values):
            tensor_name, kind = output_to_tensor[output_name]
            scalar = to_scalar_float(value)
            if kind == "min":
                stats[tensor_name]["min"] = min(stats[tensor_name]["min"], scalar)
            else:
                stats[tensor_name]["max"] = max(stats[tensor_name]["max"], scalar)
    return stats


def rewrite_model(
    model: onnx.ModelProto,
    quantizers: Dict[str, QuantizerSpec],
    rewrites: Sequence[MatMulRewriteSpec],
    tensor_stats: Dict[str, Dict[str, float]],
) -> Tuple[onnx.ModelProto, Dict[str, Any]]:
    rewritten = copy.deepcopy(model)
    quantizer_scale_inits: Dict[str, onnx.TensorProto] = {}
    quantizer_zero_point_inits: Dict[str, onnx.TensorProto] = {}
    output_scale_inits: Dict[str, onnx.TensorProto] = {}
    output_zero_point_inits: Dict[str, onnx.TensorProto] = {}

    quantizer_nodes: Dict[str, onnx.NodeProto] = {}
    qlinear_nodes: Dict[str, List[onnx.NodeProto]] = {}
    removed_node_names = set()
    summary_items: List[Dict[str, Any]] = []

    for quantizer in quantizers.values():
        stat = tensor_stats[quantizer.source_input_name]
        scale, zero_point = compute_uint8_quant_params(stat["min"], stat["max"])
        quantizer_scale_inits[quantizer.scale_name] = make_float_initializer(quantizer.scale_name, scale)
        quantizer_zero_point_inits[quantizer.zero_point_name] = make_uint8_initializer(quantizer.zero_point_name, zero_point)
        quantizer_nodes[quantizer.original_node_name] = helper.make_node(
            "QuantizeLinear",
            [quantizer.source_input_name, quantizer.scale_name, quantizer.zero_point_name],
            [quantizer.quantized_output_name],
            name=f"{quantizer.original_node_name}_static",
        )
        removed_node_names.add(quantizer.original_node_name)

    for rewrite in rewrites:
        stat = tensor_stats[rewrite.output_float_name]
        y_scale, y_zero_point = compute_uint8_quant_params(stat["min"], stat["max"])
        y_scale_name = f"{rewrite.matmul_node_name}_static_y_scale"
        y_zero_name = f"{rewrite.matmul_node_name}_static_y_zero_point"
        output_scale_inits[y_scale_name] = make_float_initializer(y_scale_name, y_scale)
        output_zero_point_inits[y_zero_name] = make_uint8_initializer(y_zero_name, y_zero_point)
        qlinear_node = helper.make_node(
            "QLinearMatMul",
            [
                rewrite.activation_quantized_name,
                rewrite.activation_scale_name,
                rewrite.activation_zero_point_name,
                rewrite.weight_quantized_name,
                rewrite.weight_scale_name,
                rewrite.weight_zero_point_name,
                y_scale_name,
                y_zero_name,
            ],
            [rewrite.qlinear_output_name],
            name=f"{rewrite.matmul_node_name}_static_qlinearmatmul",
        )
        dequant_node = helper.make_node(
            "DequantizeLinear",
            [rewrite.qlinear_output_name, y_scale_name, y_zero_name],
            [rewrite.output_float_name],
            name=f"{rewrite.matmul_node_name}_static_dequant",
        )
        qlinear_nodes[rewrite.matmul_node_name] = [qlinear_node, dequant_node]
        removed_node_names.update(
            {
                rewrite.matmul_node_name,
                rewrite.cast_node_name,
                rewrite.scale_mul_node_name,
                rewrite.output_mul_node_name,
            }
        )
        summary_items.append(
            {
                "matmul_node": rewrite.matmul_node_name,
                "activation_input": rewrite.activation_quantized_name,
                "output_float": rewrite.output_float_name,
                "output_scale_name": y_scale_name,
                "output_zero_point_name": y_zero_name,
                "output_min": stat["min"],
                "output_max": stat["max"],
                "output_scale": y_scale,
                "output_zero_point": y_zero_point,
            }
        )

    new_nodes: List[onnx.NodeProto] = []
    inserted_quantizers: set[str] = set()
    for node in rewritten.graph.node:
        if node.name in quantizer_nodes:
            if node.name not in inserted_quantizers:
                new_nodes.append(quantizer_nodes[node.name])
                inserted_quantizers.add(node.name)
            continue
        if node.name in qlinear_nodes:
            new_nodes.extend(qlinear_nodes[node.name])
            continue
        if node.name in removed_node_names:
            continue
        new_nodes.append(node)

    del rewritten.graph.node[:]
    rewritten.graph.node.extend(new_nodes)

    initializers_by_name: Dict[str, onnx.TensorProto] = {
        item.name: item for item in rewritten.graph.initializer
    }
    initializers_by_name.update(quantizer_scale_inits)
    initializers_by_name.update(quantizer_zero_point_inits)
    initializers_by_name.update(output_scale_inits)
    initializers_by_name.update(output_zero_point_inits)
    del rewritten.graph.initializer[:]
    rewritten.graph.initializer.extend(initializers_by_name.values())

    return rewritten, {
        "rewritten_matmul_count": len(rewrites),
        "rewritten_quantizer_count": len(quantizers),
        "items": summary_items,
    }


def subset_records(records: Sequence[Dict[str, Any]], calib_limit: int) -> List[Dict[str, Any]]:
    if calib_limit > 0:
        return list(records[:calib_limit])
    return list(records)


def save_model(model: onnx.ModelProto, path: str, providers: Optional[Sequence[str]] = None) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    onnx.save(model, path)
    onnx.load(path)
    if providers:
        try:
            ort.InferenceSession(path, providers=list(providers))
        except Exception as err:
            print(f"[warn] ORT load check skipped for {path}: {err}")


def staticize_component(
    component: str,
    source_model_path: str,
    probe_model_path: str,
    calib_records: Sequence[Dict[str, Any]],
    image_root: str,
    out_dir: str,
    providers: Sequence[str],
    progress_every: int,
    asset_dir: str,
    prep_vision_model_path: str,
    prep_embed_model_path: str,
    prep_decoder_model_path: str,
) -> Dict[str, Any]:
    source_model = onnx.load(source_model_path)
    quantizers, rewrites = extract_rewrite_specs(source_model)
    if not rewrites:
        raise ValueError(f"{component}: no MatMulInteger rewrite targets found in {source_model_path}")

    target_tensor_names = [item.source_input_name for item in quantizers.values()]
    target_tensor_names.extend(item.output_float_name for item in rewrites)
    probe_model = onnx.load(probe_model_path)
    reduce_model, output_map = build_reduce_model(probe_model, target_tensor_names, component)

    ensure_dir(out_dir)
    with tempfile.TemporaryDirectory(dir=out_dir) as temp_dir:
        reduce_model_path = os.path.join(temp_dir, f"{component}_reduce_probe.onnx")
        onnx.save(reduce_model, reduce_model_path)
        session = ort.InferenceSession(reduce_model_path, providers=list(providers))

        if component == "vision":
            processor = AutoProcessor.from_pretrained(asset_dir)

            def feed_builder(image_path: str, question: str) -> Dict[str, np.ndarray]:
                return build_vision_inputs(processor=processor, image_path=image_path)

        elif component == "decoder":
            runner = SmolVLM2OnnxRunner(
                model_dir=asset_dir,
                vision_model_path=prep_vision_model_path,
                embed_model_path=prep_embed_model_path,
                decoder_model_path=prep_decoder_model_path,
                providers=providers,
            )

            def feed_builder(image_path: str, question: str) -> Dict[str, np.ndarray]:
                return build_decoder_inputs(runner=runner, image_path=image_path, question=question)

        else:
            raise ValueError(f"Unsupported component: {component}")

        tensor_stats = run_reduce_calibration(
            session=session,
            output_map=output_map,
            records=calib_records,
            feed_builder=feed_builder,
            image_root=image_root,
            progress_every=progress_every,
            component=component,
        )

    rewritten_model, rewrite_summary = rewrite_model(
        model=source_model,
        quantizers=quantizers,
        rewrites=rewrites,
        tensor_stats=tensor_stats,
    )

    output_model_name = (
        "vision_encoder_int8_static_qlinearmatmul.onnx"
        if component == "vision"
        else "decoder_model_merged_int8_static_qlinearmatmul.onnx"
    )
    output_model_path = os.path.join(out_dir, output_model_name)
    save_model(rewritten_model, output_model_path, providers=providers)

    summary = {
        "component": component,
        "source_model": os.path.abspath(source_model_path),
        "probe_model": os.path.abspath(probe_model_path),
        "output_model": os.path.abspath(output_model_path),
        "calib_count": len(calib_records),
        "quantizer_count": len(quantizers),
        "matmul_rewrite_count": len(rewrites),
        "quantizers": [
            {
                "source_input": item.source_input_name,
                "scale_name": item.scale_name,
                "zero_point_name": item.zero_point_name,
                "source_min": tensor_stats[item.source_input_name]["min"],
                "source_max": tensor_stats[item.source_input_name]["max"],
            }
            for item in quantizers.values()
        ],
        "rewrites": rewrite_summary["items"],
    }
    summary_path = os.path.join(out_dir, f"{component}_static_qlinearmatmul_summary.json")
    save_json(summary, summary_path)
    print(f"[{component}] saved model: {output_model_path}")
    print(f"[{component}] saved summary: {summary_path}")
    return summary


def main() -> int:
    args = parse_args()
    components = parse_components(args.components)
    providers = parse_providers(args.providers)
    ensure_dir(args.out_dir)

    asset_dir = resolve_asset_dir(args.asset_dir, args.model_dir)
    vision_model_path = resolve_existing_path(
        model_dir=args.model_dir,
        explicit_path=args.vision_model,
        candidates=["models/vision_encoder_int8.onnx", "vision_encoder_int8.onnx"],
        label="vision int8",
    )
    embed_model_path = resolve_existing_path(
        model_dir=args.model_dir,
        explicit_path=args.embed_model,
        candidates=["models/embed_tokens_int8.onnx", "embed_tokens_int8.onnx"],
        label="embed int8",
    )
    decoder_model_path = resolve_existing_path(
        model_dir=args.model_dir,
        explicit_path=args.decoder_model,
        candidates=["models/decoder_model_merged_int8.onnx", "decoder_model_merged_int8.onnx"],
        label="decoder int8",
    )
    vision_probe_model_path = resolve_existing_path(
        model_dir=args.model_dir,
        explicit_path=args.vision_probe_model,
        candidates=["models/vision_encoder_int8.onnx", "vision_encoder_int8.onnx", "models/vision_encoder.onnx"],
        label="vision probe",
    )
    decoder_probe_model_path = resolve_existing_path(
        model_dir=args.model_dir,
        explicit_path=args.decoder_probe_model,
        candidates=["models/decoder_model_merged_int8.onnx", "decoder_model_merged_int8.onnx"],
        label="decoder probe",
    )
    prep_vision_model_path = resolve_existing_path(
        model_dir=args.model_dir,
        explicit_path=args.prep_vision_model,
        candidates=["models/vision_encoder.onnx", "vision_encoder.onnx", "models/vision_encoder_int8.onnx"],
        label="prep vision",
    )
    prep_embed_model_path = resolve_existing_path(
        model_dir=args.model_dir,
        explicit_path=args.prep_embed_model,
        candidates=["models/embed_tokens.onnx", "embed_tokens.onnx", "models/embed_tokens_int8.onnx"],
        label="prep embed",
    )
    prep_decoder_model_path = resolve_existing_path(
        model_dir=args.model_dir,
        explicit_path=args.prep_decoder_model,
        candidates=["models/decoder_model_merged_int8.onnx", "decoder_model_merged_int8.onnx", "models/decoder_model_merged.onnx"],
        label="prep decoder",
    )

    calib_records = subset_records(load_json(args.calib_json), args.calib_limit)
    if not isinstance(calib_records, list) or not calib_records:
        raise ValueError("Calibration json must be a non-empty list.")

    all_summaries: Dict[str, Any] = {
        "asset_dir": os.path.abspath(asset_dir),
        "model_dir": os.path.abspath(args.model_dir),
        "image_root": os.path.abspath(args.image_root),
        "calib_json": os.path.abspath(args.calib_json),
        "providers": providers,
        "components": {},
    }

    for component in components:
        component_model_path = vision_model_path if component == "vision" else decoder_model_path
        component_probe_model_path = (
            vision_probe_model_path if component == "vision" else decoder_probe_model_path
        )
        all_summaries["components"][component] = staticize_component(
            component=component,
            source_model_path=component_model_path,
            probe_model_path=component_probe_model_path,
            calib_records=calib_records,
            image_root=args.image_root,
            out_dir=args.out_dir,
            providers=providers,
            progress_every=args.progress_every,
            asset_dir=asset_dir,
            prep_vision_model_path=prep_vision_model_path,
            prep_embed_model_path=prep_embed_model_path,
            prep_decoder_model_path=prep_decoder_model_path,
        )

    summary_path = os.path.join(args.out_dir, "static_qlinearmatmul_overall_summary.json")
    save_json(all_summaries, summary_path)
    print(f"[done] saved overall summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
