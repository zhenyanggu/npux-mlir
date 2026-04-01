#!/usr/bin/env python3
import argparse
import copy
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from transformers import AutoProcessor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AICAS_ROOT = os.path.dirname(SCRIPT_DIR)
if AICAS_ROOT not in sys.path:
    sys.path.insert(0, AICAS_ROOT)

from smolvlm2_onnx_lib import SmolVLM2OnnxRunner, ensure_dir, load_json, save_json


DEFAULT_IMAGE_ROOT = os.path.join(AICAS_ROOT, "data")
DEFAULT_MODEL_DIR = AICAS_ROOT


@dataclass(frozen=True)
class QuantizerSpec:
    node_name: str
    source_input_name: str
    quantized_output_name: str
    scale_name: str
    zero_point_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze official DynamicQuantizeLinear nodes into static "
            "QuantizeLinear + constant scale/zero-point, while preserving "
            "the original MatMulInteger/ConvInteger computation graph."
        )
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--asset-dir", default="")
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--calib-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--components", default="vision,decoder")
    parser.add_argument("--providers", default="CPUExecutionProvider")
    parser.add_argument("--calib-limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--vision-model", default="")
    parser.add_argument("--decoder-model", default="")
    parser.add_argument("--prep-vision-model", default="")
    parser.add_argument("--prep-embed-model", default="")
    parser.add_argument("--prep-decoder-model", default="")
    parser.add_argument("--vision-probe-model", default="")
    parser.add_argument("--decoder-probe-model", default="")
    return parser.parse_args()


def parse_components(text: str) -> List[str]:
    items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise ValueError("No component provided.")
    valid = {"vision", "decoder"}
    for item in items:
        if item not in valid:
            raise ValueError(f"Unsupported component: {item}")
    return items


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


def collect_quantizers(model: onnx.ModelProto) -> List[QuantizerSpec]:
    result: List[QuantizerSpec] = []
    for node in model.graph.node:
        if node.op_type != "DynamicQuantizeLinear":
            continue
        if len(node.input) != 1 or len(node.output) != 3:
            raise ValueError(f"{node.name}: unexpected DynamicQuantizeLinear shape.")
        result.append(
            QuantizerSpec(
                node_name=node.name,
                source_input_name=node.input[0],
                quantized_output_name=node.output[0],
                scale_name=node.output[1],
                zero_point_name=node.output[2],
            )
        )
    return result


def build_reduce_model(
    model: onnx.ModelProto,
    tensor_names: Sequence[str],
    prefix: str,
) -> Tuple[onnx.ModelProto, Dict[str, Tuple[str, str]]]:
    reduced_model = copy.deepcopy(model)
    output_map: Dict[str, Tuple[str, str]] = {}
    nodes_to_add: List[onnx.NodeProto] = []
    outputs_to_add: List[onnx.ValueInfoProto] = []

    for index, tensor_name in enumerate(sorted(set(tensor_names))):
        min_name = f"__freeze_dql_{prefix}_{index}_min"
        max_name = f"__freeze_dql_{prefix}_{index}_max"
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


def build_vision_inputs(
    processor: AutoProcessor,
    image_path: str,
) -> Dict[str, np.ndarray]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    try:
        prepared = processor.image_processor.preprocess(images=[image], return_tensors="np")
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


def to_scalar_float(value: Any) -> float:
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def run_calibration(
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


def compute_uint8_quant_params(min_value: float, max_value: float) -> Tuple[float, int]:
    min_value = min(min_value, 0.0)
    max_value = max(max_value, 0.0)
    if max_value - min_value < 1e-12:
        return 1.0, 0
    scale = float((max_value - min_value) / 255.0)
    zero_point = int(round(-min_value / scale))
    zero_point = min(255, max(0, zero_point))
    return scale, zero_point


def rewrite_quantizers(
    model: onnx.ModelProto,
    quantizers: Sequence[QuantizerSpec],
    tensor_stats: Dict[str, Dict[str, float]],
) -> Tuple[onnx.ModelProto, Dict[str, Any]]:
    rewritten = copy.deepcopy(model)
    replacement_nodes: Dict[str, onnx.NodeProto] = {}
    added_initializers: Dict[str, onnx.TensorProto] = {}
    summary_items: List[Dict[str, Any]] = []

    for quantizer in quantizers:
        stats = tensor_stats[quantizer.source_input_name]
        scale, zero_point = compute_uint8_quant_params(stats["min"], stats["max"])
        added_initializers[quantizer.scale_name] = numpy_helper.from_array(
            np.asarray(scale, dtype=np.float32), name=quantizer.scale_name
        )
        added_initializers[quantizer.zero_point_name] = numpy_helper.from_array(
            np.asarray(zero_point, dtype=np.uint8), name=quantizer.zero_point_name
        )
        replacement_nodes[quantizer.node_name] = helper.make_node(
            "QuantizeLinear",
            [quantizer.source_input_name, quantizer.scale_name, quantizer.zero_point_name],
            [quantizer.quantized_output_name],
            name=f"{quantizer.node_name}_static",
        )
        summary_items.append(
            {
                "quantizer": quantizer.node_name,
                "source_input": quantizer.source_input_name,
                "scale_name": quantizer.scale_name,
                "zero_point_name": quantizer.zero_point_name,
                "source_min": stats["min"],
                "source_max": stats["max"],
                "scale": scale,
                "zero_point": zero_point,
            }
        )

    new_nodes: List[onnx.NodeProto] = []
    for node in rewritten.graph.node:
        replacement = replacement_nodes.get(node.name)
        new_nodes.append(replacement if replacement is not None else node)
    del rewritten.graph.node[:]
    rewritten.graph.node.extend(new_nodes)

    initializers = {item.name: item for item in rewritten.graph.initializer}
    initializers.update(added_initializers)
    del rewritten.graph.initializer[:]
    rewritten.graph.initializer.extend(initializers.values())

    return rewritten, {
        "quantizer_count": len(summary_items),
        "items": summary_items,
    }


def save_model(model: onnx.ModelProto, path: str, providers: Optional[Sequence[str]] = None) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    onnx.save(model, path)
    onnx.load(path)
    if providers:
        try:
            ort.InferenceSession(path, providers=list(providers))
        except Exception as err:
            print(f"[warn] ORT load check skipped for {path}: {err}")


def subset_records(records: Sequence[Dict[str, Any]], calib_limit: int) -> List[Dict[str, Any]]:
    if calib_limit > 0:
        return list(records[:calib_limit])
    return list(records)


def freeze_component(
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
    quantizers = collect_quantizers(source_model)
    if not quantizers:
        raise ValueError(f"{component}: no DynamicQuantizeLinear found in {source_model_path}")

    probe_model = onnx.load(probe_model_path)
    reduce_model, output_map = build_reduce_model(
        probe_model,
        [item.source_input_name for item in quantizers],
        component,
    )

    ensure_dir(out_dir)
    with tempfile.TemporaryDirectory(dir=out_dir) as temp_dir:
        reduce_model_path = os.path.join(temp_dir, f"{component}_freeze_dql_probe.onnx")
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

        tensor_stats = run_calibration(
            session=session,
            output_map=output_map,
            records=calib_records,
            feed_builder=feed_builder,
            image_root=image_root,
            progress_every=progress_every,
            component=component,
        )

    rewritten_model, rewrite_summary = rewrite_quantizers(
        model=source_model,
        quantizers=quantizers,
        tensor_stats=tensor_stats,
    )

    output_name = (
        "vision_encoder_int8_frozen_dql.onnx"
        if component == "vision"
        else "decoder_model_merged_int8_frozen_dql.onnx"
    )
    output_model_path = os.path.join(out_dir, output_name)
    save_model(rewritten_model, output_model_path, providers=providers)

    summary = {
        "component": component,
        "source_model": os.path.abspath(source_model_path),
        "probe_model": os.path.abspath(probe_model_path),
        "output_model": os.path.abspath(output_model_path),
        "calib_count": len(calib_records),
        "quantizer_count": rewrite_summary["quantizer_count"],
        "quantizers": rewrite_summary["items"],
    }
    summary_path = os.path.join(out_dir, f"{component}_frozen_dql_summary.json")
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
        candidates=[
            "models/vision_encoder_int8_cuda.onnx",
            "models/vision_encoder_int8.onnx",
            "vision_encoder_int8_cuda.onnx",
            "vision_encoder_int8.onnx",
        ],
        label="vision int8",
    )
    decoder_model_path = resolve_existing_path(
        model_dir=args.model_dir,
        explicit_path=args.decoder_model,
        candidates=["models/decoder_model_merged_int8.onnx", "decoder_model_merged_int8.onnx"],
        label="decoder int8",
    )
    prep_vision_model_path = resolve_existing_path(
        model_dir=args.model_dir,
        explicit_path=args.prep_vision_model,
        candidates=["models/vision_encoder.onnx", "vision_encoder.onnx", "models/vision_encoder_int8_cuda.onnx"],
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
        candidates=["models/decoder_model_merged_int8.onnx", "decoder_model_merged_int8.onnx"],
        label="prep decoder",
    )
    vision_probe_model_path = resolve_existing_path(
        model_dir=args.model_dir,
        explicit_path=args.vision_probe_model,
        candidates=["models/vision_encoder.onnx", "vision_encoder.onnx", "models/vision_encoder_int8_cuda.onnx"],
        label="vision probe",
    )
    decoder_probe_model_path = resolve_existing_path(
        model_dir=args.model_dir,
        explicit_path=args.decoder_probe_model,
        candidates=["models/decoder_model_merged_int8.onnx", "decoder_model_merged_int8.onnx"],
        label="decoder probe",
    )

    calib_records = subset_records(load_json(args.calib_json), args.calib_limit)
    if not isinstance(calib_records, list) or not calib_records:
        raise ValueError("Calibration json must be a non-empty list.")

    overall = {
        "asset_dir": os.path.abspath(asset_dir),
        "model_dir": os.path.abspath(args.model_dir),
        "image_root": os.path.abspath(args.image_root),
        "calib_json": os.path.abspath(args.calib_json),
        "providers": providers,
        "components": {},
    }

    for component in components:
        source_model_path = vision_model_path if component == "vision" else decoder_model_path
        probe_model_path = vision_probe_model_path if component == "vision" else decoder_probe_model_path
        overall["components"][component] = freeze_component(
            component=component,
            source_model_path=source_model_path,
            probe_model_path=probe_model_path,
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

    summary_path = os.path.join(args.out_dir, "frozen_dql_overall_summary.json")
    save_json(overall, summary_path)
    print(f"[done] saved overall summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
