import argparse
import os
from typing import Dict, List

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from smolvlm2_onnx_lib import SmolVLM2OnnxRunner, ensure_dir, load_json


NON_FP32_FLOAT_INPUT_TYPES = {"FLOAT16", "DOUBLE", "BFLOAT16"}
DECODER_REQUIRED_CUSTOM_OPS = {
    "com.microsoft::MultiHeadAttention",
    "com.microsoft::RotaryEmbedding",
    "com.microsoft::SkipSimplifiedLayerNormalization",
}


class ListDataReader(CalibrationDataReader):
    def __init__(self, items: List[Dict[str, np.ndarray]]) -> None:
        self.items = items
        self.index = 0

    def get_next(self) -> Dict[str, np.ndarray] | None:
        if self.index >= len(self.items):
            return None
        item = self.items[self.index]
        self.index += 1
        return item

    def rewind(self) -> None:
        self.index = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--calib-json", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def resolve_existing_path(model_dir: str, candidates: List[str], label: str) -> str:
    tried: List[str] = []
    for candidate in candidates:
        path = os.path.abspath(os.path.join(model_dir, candidate))
        tried.append(path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{label} model not found in candidates: {tried}")


def collect_constant_outputs(graph: onnx.GraphProto) -> set[str]:
    names = {item.name for item in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for out in node.output:
                if out:
                    names.add(out)
    return names


def graph_input_dtypes(graph: onnx.GraphProto) -> Dict[str, str]:
    return {
        item.name: str(onnx.TensorProto.DataType.Name(item.type.tensor_type.elem_type))
        for item in graph.input
    }


def collect_static_linear_node_names(model_path: str) -> List[str]:
    model = onnx.load(model_path)
    const_outputs = collect_constant_outputs(model.graph)
    names: List[str] = []
    unnamed = 0
    for node in model.graph.node:
        if node.op_type not in {"MatMul", "Gemm"}:
            continue
        if len(node.input) < 2 or node.input[1] not in const_outputs:
            continue
        if node.name:
            names.append(node.name)
        else:
            unnamed += 1
    if unnamed:
        raise ValueError(
            f"{model_path} has {unnamed} unnamed static MatMul/Gemm nodes; "
            "cannot safely quantize only static linear ops."
        )
    return names


def validate_fp32_quant_source_inputs(label: str, model_path: str) -> None:
    input_dtypes = graph_input_dtypes(onnx.load(model_path).graph)
    bad_inputs = {
        name: dtype
        for name, dtype in input_dtypes.items()
        if dtype in NON_FP32_FLOAT_INPUT_TYPES
    }
    if bad_inputs:
        raise ValueError(
            f"{label} quant-source model must keep fp32 floating inputs, "
            f"but found non-fp32 inputs in {model_path}: {bad_inputs}"
        )


def validate_decoder_custom_ops(model_path: str) -> None:
    model = onnx.load(model_path)
    custom_ops = {
        f"{(node.domain or 'ai.onnx')}::{node.op_type}"
        for node in model.graph.node
        if node.domain or node.op_type in {
            "MultiHeadAttention",
            "RotaryEmbedding",
            "SimplifiedLayerNormalization",
            "SkipSimplifiedLayerNormalization",
        }
    }
    missing = [name for name in sorted(DECODER_REQUIRED_CUSTOM_OPS) if name not in custom_ops]
    if missing:
        raise ValueError(
            "Decoder quant-source model looks rewritten or custom ops were stripped. "
            f"Expected to keep {missing}, got path={model_path}"
        )


def build_vision_reader(runner: SmolVLM2OnnxRunner, calib_records: List[dict], image_root: str) -> ListDataReader:
    items: List[Dict[str, np.ndarray]] = []
    for item in calib_records:
        prepared = runner.prepare_inputs(
            image_path=os.path.join(image_root, item["image_path"]),
            question=item["question"],
        )
        items.append(
            {
                "pixel_values": prepared["pixel_values"].astype(np.float32, copy=False),
                "pixel_attention_mask": prepared["pixel_attention_mask"].astype(bool, copy=False),
            }
        )
    return ListDataReader(items)


def build_decoder_reader(runner: SmolVLM2OnnxRunner, calib_records: List[dict], image_root: str) -> ListDataReader:
    items: List[Dict[str, np.ndarray]] = []
    for item in calib_records:
        image_path = os.path.join(image_root, item["image_path"])
        state = runner.prefill(image_path=image_path, question=item["question"])
        items.append(
            {
                "inputs_embeds": state["merged_embeds"].astype(
                    runner.decoder_inputs_embeds_dtype, copy=False
                ),
                "attention_mask": state["attention_mask"].astype(
                    runner.decoder_attention_mask_dtype, copy=False
                ),
                "position_ids": state["position_ids"].astype(
                    runner.decoder_position_ids_dtype, copy=False
                ),
                **runner.zero_past_key_values(batch_size=1),
            }
        )
    return ListDataReader(items)


def quantize_model(
    model_input_path: str,
    model_output_path: str,
    reader: CalibrationDataReader,
    nodes_to_quantize: List[str],
) -> None:
    quantize_static(
        model_input=model_input_path,
        model_output=model_output_path,
        calibration_data_reader=reader,
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
        nodes_to_quantize=nodes_to_quantize,
        extra_options={
            "WeightSymmetric": True,
            "ActivationSymmetric": True,
            "MatMulConstBOnly": True,
        },
    )


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    calib_records = load_json(args.calib_json)
    image_root = os.path.dirname(args.calib_json)

    vision_input = resolve_existing_path(
        model_dir=args.model_dir,
        candidates=[
            "models/vision_encoder.onnx",
            "vision_encoder.onnx",
        ],
        label="vision",
    )
    embed_input = resolve_existing_path(
        model_dir=args.model_dir,
        candidates=[
            "models/embed_tokens.onnx",
            "embed_tokens.onnx",
        ],
        label="embed",
    )
    decoder_input = resolve_existing_path(
        model_dir=args.model_dir,
        candidates=[
            "models/decoder_model_merged.onnx",
            "decoder_model_merged.onnx",
        ],
        label="decoder",
    )
    validate_fp32_quant_source_inputs("vision", vision_input)
    validate_fp32_quant_source_inputs("decoder", decoder_input)
    validate_decoder_custom_ops(decoder_input)
    vision_nodes_to_quantize = collect_static_linear_node_names(vision_input)
    decoder_nodes_to_quantize = collect_static_linear_node_names(decoder_input)
    runner = SmolVLM2OnnxRunner(
        model_dir=args.model_dir,
        vision_model_path=vision_input,
        embed_model_path=embed_input,
        decoder_model_path=decoder_input,
    )

    vision_out = os.path.join(args.out_dir, "vision_encoder_int8_sym.onnx")
    decoder_out = os.path.join(args.out_dir, "decoder_model_merged_int8_sym.onnx")

    print("quantizing vision encoder...")
    quantize_model(
        model_input_path=vision_input,
        model_output_path=vision_out,
        reader=build_vision_reader(runner=runner, calib_records=calib_records, image_root=image_root),
        nodes_to_quantize=vision_nodes_to_quantize,
    )

    print("quantizing decoder...")
    quantize_model(
        model_input_path=decoder_input,
        model_output_path=decoder_out,
        reader=build_decoder_reader(runner=runner, calib_records=calib_records, image_root=image_root),
        nodes_to_quantize=decoder_nodes_to_quantize,
    )

    print(f"saved {vision_out}")
    print(f"saved {decoder_out}")


if __name__ == "__main__":
    main()
