#!/usr/bin/env python3
"""Statically quantize Qwen2.5 fixed-shape prefill linear layers to QDQ."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from qwen25_onnx_lib import DEFAULT_PROMPTS, Qwen25PrefillInputs


class ListDataReader(CalibrationDataReader):
    def __init__(self, items: Sequence[Dict[str, np.ndarray]]) -> None:
        self._items = list(items)
        self._index = 0

    def get_next(self) -> Dict[str, np.ndarray] | None:
        if self._index >= len(self._items):
            return None
        item = self._items[self._index]
        self._index += 1
        return item

    def rewind(self) -> None:
        self._index = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--fp32-quant-source",
        default="",
        help="Optional path for the generated FP32 ONNX quantization source.",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument(
        "--calibration-providers",
        default="CUDAExecutionProvider,CPUExecutionProvider",
        help="Comma-separated ONNX Runtime providers used only for calibration runs.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def constant_outputs(graph: onnx.GraphProto) -> set[str]:
    names = {item.name for item in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            names.update(name for name in node.output if name)
    return names


def collect_static_linear_node_names(model: onnx.ModelProto) -> List[str]:
    constants = constant_outputs(model.graph)
    names: List[str] = []
    unnamed = 0
    for node in model.graph.node:
        if node.op_type not in {"MatMul", "Gemm"}:
            continue
        if len(node.input) < 2 or node.input[1] not in constants:
            continue
        if node.name:
            names.append(node.name)
        else:
            unnamed += 1
    if unnamed:
        raise ValueError(
            f"model contains {unnamed} unnamed static MatMul/Gemm nodes; "
            "refusing to select them by an unstable generated name"
        )
    if not names:
        raise ValueError("no static MatMul/Gemm nodes found")
    return names


def model_input_names(model: onnx.ModelProto) -> set[str]:
    initializer_names = {item.name for item in model.graph.initializer}
    return {item.name for item in model.graph.input if item.name not in initializer_names}


def convert_tensor_float16_to_float32(tensor: onnx.TensorProto) -> None:
    if tensor.data_type != onnx.TensorProto.FLOAT16:
        return
    converted = numpy_helper.from_array(
        numpy_helper.to_array(tensor).astype(np.float32), name=tensor.name
    )
    tensor.CopyFrom(converted)


def convert_graph_float16_to_float32(graph: onnx.GraphProto) -> None:
    for initializer in graph.initializer:
        convert_tensor_float16_to_float32(initializer)
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            tensor_type.elem_type = onnx.TensorProto.FLOAT
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                convert_tensor_float16_to_float32(attribute.t)
            elif attribute.type == onnx.AttributeProto.TENSORS:
                for tensor in attribute.tensors:
                    convert_tensor_float16_to_float32(tensor)
            elif attribute.type == onnx.AttributeProto.GRAPH:
                convert_graph_float16_to_float32(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    convert_graph_float16_to_float32(subgraph)
            elif (
                attribute.name == "to"
                and attribute.type == onnx.AttributeProto.INT
                and attribute.i == onnx.TensorProto.FLOAT16
            ):
                attribute.i = onnx.TensorProto.FLOAT


def build_fp32_quant_source(source_model: onnx.ModelProto, output_path: Path) -> Path:
    convert_graph_float16_to_float32(source_model.graph)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        source_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.name + ".data",
        size_threshold=0,
        convert_attribute=False,
    )
    onnx.checker.check_model(str(output_path))
    return output_path


def build_calibration_reader(
    hf_model: str,
    input_names: set[str],
    prompts: Sequence[str],
    seq_len: int,
    trust_remote_code: bool,
) -> ListDataReader:
    input_builder = Qwen25PrefillInputs(
        hf_model=hf_model, seq_len=seq_len, trust_remote_code=trust_remote_code
    )
    items: List[Dict[str, np.ndarray]] = []
    for prompt in prompts:
        feeds = input_builder.build(prompt).feeds
        missing = input_names - set(feeds)
        if missing:
            raise ValueError(f"model inputs not provided by Qwen prefill builder: {sorted(missing)}")
        items.append({name: feeds[name] for name in sorted(input_names)})
    return ListDataReader(items)


def parse_providers(text: str) -> List[str]:
    providers = [item.strip() for item in text.split(",") if item.strip()]
    if not providers:
        raise ValueError("at least one calibration provider is required")
    unavailable = set(providers) - set(ort.get_available_providers())
    if unavailable:
        raise ValueError(
            "requested calibration providers are unavailable: "
            f"{sorted(unavailable)}; available={ort.get_available_providers()}"
        )
    return providers


def quantize_with_calibration_providers(
    providers: Sequence[str], **kwargs: object
) -> None:
    quantize_module = importlib.import_module("onnxruntime.quantization.quantize")
    original_create_calibrator = quantize_module.create_calibrator

    def create_calibrator_with_providers(*args: object, **inner_kwargs: object) -> object:
        calibrator = original_create_calibrator(*args, **inner_kwargs)
        calibrator.set_execution_providers(list(providers))
        return calibrator

    quantize_module.create_calibrator = create_calibrator_with_providers
    try:
        quantize_static(**kwargs)
    finally:
        quantize_module.create_calibrator = original_create_calibrator


def count_quantized_graph_nodes(model_path: Path) -> Dict[str, int]:
    model = onnx.load(str(model_path), load_external_data=False)
    counts: Dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return {
        "node_count": len(model.graph.node),
        "qlinear_matmul_count": counts.get("QLinearMatMul", 0),
        "quantize_linear_count": counts.get("QuantizeLinear", 0),
        "dequantize_linear_count": counts.get("DequantizeLinear", 0),
        "matmul_count": counts.get("MatMul", 0),
        "gemm_count": counts.get("Gemm", 0),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_model).resolve()
    output_path = Path(args.output_model).resolve()
    summary_path = Path(args.summary_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    source = onnx.load(str(input_path), load_external_data=False)
    nodes_to_quantize = collect_static_linear_node_names(source)
    prompts = args.prompt or DEFAULT_PROMPTS
    calibration_providers = parse_providers(args.calibration_providers)
    reader = build_calibration_reader(
        hf_model=args.hf_model,
        input_names=model_input_names(source),
        prompts=prompts,
        seq_len=args.seq_len,
        trust_remote_code=args.trust_remote_code,
    )

    contains_float16 = any(
        initializer.data_type == onnx.TensorProto.FLOAT16
        for initializer in source.graph.initializer
    )
    quantization_source = input_path
    if contains_float16:
        quantization_source = (
            Path(args.fp32_quant_source).resolve()
            if args.fp32_quant_source
            else output_path.with_name(output_path.stem + "_quant_source_fp32.onnx")
        )
        build_fp32_quant_source(source, quantization_source)

    quantize_with_calibration_providers(
        providers=calibration_providers,
        model_input=str(quantization_source),
        model_output=str(output_path),
        calibration_data_reader=reader,
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        # ORT 1.19 adjusts every Softmax range while building a QDQ model,
        # even when Softmax is absent from nodes_to_quantize. Include it in
        # calibration instrumentation so that adjustment has a collected
        # range; nodes_to_quantize still limits emitted QDQ to static linear
        # layers only.
        op_types_to_quantize=["MatMul", "Gemm", "Softmax"],
        nodes_to_quantize=nodes_to_quantize,
        extra_options={
            "WeightSymmetric": True,
            "ActivationSymmetric": True,
            "MatMulConstBOnly": True,
        },
        use_external_data_format=True,
    )

    onnx.checker.check_model(str(output_path))
    payload = {
        "input_model": str(input_path),
        "quantization_source_model": str(quantization_source),
        "source_float16_converted_to_float32": contains_float16,
        "output_model": str(output_path),
        "hf_model": args.hf_model,
        "seq_len": args.seq_len,
        "calibration_prompts": list(prompts),
        "calibration_providers_requested": calibration_providers,
        "static_linear_nodes_requested": len(nodes_to_quantize),
        "static_linear_node_names": nodes_to_quantize,
        "quantized_graph": count_quantized_graph_nodes(output_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
