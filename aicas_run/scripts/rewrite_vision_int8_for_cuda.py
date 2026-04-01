#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List

import onnx
from onnx import helper


TARGET_CONV_NAME = "/vision_model/embeddings/patch_embedding/Conv_quant"
TARGET_CAST_NAME = "/vision_model/embeddings/patch_embedding/Conv_output_0_output_quantized_cast"
TARGET_SCALES_MUL_NAME = "/vision_model/embeddings/patch_embedding/Conv_quant_scales_mul"
TARGET_OUTPUT_SCALE_MUL_NAME = "/vision_model/embeddings/patch_embedding/Conv_quant_output_scale_mul"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite the official int8 vision encoder so the patch embedding "
            "ConvInteger path becomes DequantizeLinear + Conv, which ORT CUDA can load."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def find_node_by_name(nodes: List[onnx.NodeProto], name: str) -> onnx.NodeProto:
    for node in nodes:
        if node.name == name:
            return node
    raise ValueError(f"node not found: {name}")


def main() -> int:
    args = parse_args()
    model = onnx.load(args.input)
    nodes = list(model.graph.node)

    conv_node = find_node_by_name(nodes, TARGET_CONV_NAME)
    find_node_by_name(nodes, TARGET_CAST_NAME)
    find_node_by_name(nodes, TARGET_SCALES_MUL_NAME)
    find_node_by_name(nodes, TARGET_OUTPUT_SCALE_MUL_NAME)

    conv_attrs: Dict[str, object] = {
        attr.name: helper.get_attribute_value(attr) for attr in conv_node.attribute
    }

    dq_input_out = "/vision_model/embeddings/patch_embedding/Conv_input_dequant_output"
    dq_weight_out = "/vision_model/embeddings/patch_embedding/Conv_weight_dequant_output"
    conv_out = "/vision_model/embeddings/patch_embedding/Conv_output_0quant_scaled_output"

    new_nodes = [
        helper.make_node(
            "DequantizeLinear",
            inputs=[
                "/GatherND_output_0_quantized",
                "/GatherND_output_0_scale",
                "/GatherND_output_0_zero_point",
            ],
            outputs=[dq_input_out],
            name="/vision_model/embeddings/patch_embedding/Conv_input_dequant",
        ),
        helper.make_node(
            "DequantizeLinear",
            inputs=[
                "vision_model.embeddings.patch_embedding.weight_quantized",
                "vision_model.embeddings.patch_embedding.weight_scale",
                "vision_model.embeddings.patch_embedding.weight_zero_point",
            ],
            outputs=[dq_weight_out],
            name="/vision_model/embeddings/patch_embedding/Conv_weight_dequant",
        ),
        helper.make_node(
            "Conv",
            inputs=[dq_input_out, dq_weight_out],
            outputs=[conv_out],
            name="/vision_model/embeddings/patch_embedding/Conv_dequant",
            **conv_attrs,
        ),
    ]

    remove_names = {
        TARGET_CONV_NAME,
        TARGET_CAST_NAME,
        TARGET_SCALES_MUL_NAME,
        TARGET_OUTPUT_SCALE_MUL_NAME,
    }
    conv_index = next(i for i, node in enumerate(nodes) if node.name == TARGET_CONV_NAME)
    kept_nodes = [node for node in nodes if node.name not in remove_names]

    rebuilt_nodes: List[onnx.NodeProto] = []
    inserted = False
    for node in kept_nodes:
        if not inserted and len(rebuilt_nodes) == conv_index:
            rebuilt_nodes.extend(new_nodes)
            inserted = True
        rebuilt_nodes.append(node)
    if not inserted:
        rebuilt_nodes.extend(new_nodes)

    del model.graph.node[:]
    model.graph.node.extend(rebuilt_nodes)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    onnx.save(model, args.output)
    print(f"saved rewritten model: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
