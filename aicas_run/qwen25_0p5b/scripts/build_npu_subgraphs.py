#!/usr/bin/env python3
"""Build Qwen-shaped FP16 references and INT8 NPU proxy subgraphs.

The full Qwen export is FP16 while the current NPU partition accepts QDQ or
QLinearMatMul graphs.  This tool preserves the relevant Qwen tensor shapes in
small, self-contained FP16 reference graphs, then builds matching INT8 proxy
graphs that exercise the implemented NPU paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


FP16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True, help="Full Qwen ONNX model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, default=0, help="Decoder layer to inspect")
    return parser.parse_args()


def shape_from_value_info(value_info: onnx.ValueInfoProto) -> List[int]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise ValueError(f"missing shape for '{value_info.name}'")
    shape: List[int] = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise ValueError(f"non-static shape for '{value_info.name}'")
        shape.append(int(dim.dim_value))
    return shape


def build_shape_map(model: onnx.ModelProto) -> Dict[str, List[int]]:
    shape_map: Dict[str, List[int]] = {}
    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        try:
            shape_map[value_info.name] = shape_from_value_info(value_info)
        except ValueError:
            pass
    for initializer in model.graph.initializer:
        shape_map[initializer.name] = [int(dim) for dim in initializer.dims]
    return shape_map


def load_source_model(path: Path) -> onnx.ModelProto:
    # Shape inference only needs tensor metadata. Avoid loading the 1.2G weights.
    model = onnx.load(path, load_external_data=False)
    try:
        return onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except onnx.onnx_cpp2py_export.shape_inference.InferenceError:
        return model


def find_matmul(model: onnx.ModelProto, fragment: str) -> onnx.NodeProto:
    matches = [
        node
        for node in model.graph.node
        if node.op_type == "MatMul" and fragment in node.name
    ]
    if len(matches) != 1:
        names = [node.name for node in matches]
        raise ValueError(f"expected one MatMul containing '{fragment}', found {names}")
    return matches[0]


def find_attention_score_matmul(
    model: onnx.ModelProto, layer: int, initializer_names: Iterable[str]
) -> onnx.NodeProto:
    score_name = f"/layers.{layer}/self_attn/MatMul"
    init_names = set(initializer_names)
    matches = [
        node
        for node in model.graph.node
        if node.op_type == "MatMul"
        and node.name.endswith(score_name)
        and len(node.input) == 2
        and node.input[1] not in init_names
    ]
    if len(matches) != 1:
        names = [node.name for node in matches]
        raise ValueError(
            f"expected one attention score MatMul ending in '{score_name}', found {names}"
        )
    return matches[0]


def value_info(name: str, dtype: int, shape: Sequence[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, list(shape))


def zero_initializer(name: str, dtype: np.dtype, shape: Sequence[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.zeros(tuple(shape), dtype=dtype), name=name)


def scalar_initializer(name: str, value: object, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray([value], dtype=dtype), name=name)


def make_model(
    name: str,
    nodes: Sequence[onnx.NodeProto],
    inputs: Sequence[onnx.ValueInfoProto],
    outputs: Sequence[onnx.ValueInfoProto],
    initializers: Sequence[onnx.TensorProto],
) -> onnx.ModelProto:
    graph = helper.make_graph(
        list(nodes), name, list(inputs), list(outputs), initializer=list(initializers)
    )
    model = helper.make_model(
        graph,
        producer_name="qwen25_npu_subgraph_builder",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = onnx.IR_VERSION
    onnx.checker.check_model(model)
    return model


def save_model(model: onnx.ModelProto, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = output_path.with_suffix(output_path.suffix + ".data")
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
        convert_attribute=False,
    )


def qlinear_matmul_node(
    name: str, a: str, b: str, output: str, prefix: str
) -> onnx.NodeProto:
    return helper.make_node(
        "QLinearMatMul",
        [
            a,
            f"{prefix}_a_scale",
            f"{prefix}_a_zp",
            b,
            f"{prefix}_b_scale",
            f"{prefix}_b_zp",
            f"{prefix}_y_scale",
            f"{prefix}_y_zp",
        ],
        [output],
        name=name,
    )


def qlinear_params(prefix: str) -> List[onnx.TensorProto]:
    return [
        scalar_initializer(f"{prefix}_a_scale", 0.02, np.float32),
        scalar_initializer(f"{prefix}_a_zp", 128, np.uint8),
        scalar_initializer(f"{prefix}_b_scale", 0.02, np.float32),
        scalar_initializer(f"{prefix}_b_zp", 128, np.uint8),
        scalar_initializer(f"{prefix}_y_scale", 0.04, np.float32),
        scalar_initializer(f"{prefix}_y_zp", 128, np.uint8),
    ]


def build_fp16_mlp(shapes: Dict[str, List[int]], gate: onnx.NodeProto, up: onnx.NodeProto,
                   down: onnx.NodeProto) -> onnx.ModelProto:
    input_shape = shapes[gate.input[0]]
    gate_weight_shape = shapes[gate.input[1]]
    up_weight_shape = shapes[up.input[1]]
    down_weight_shape = shapes[down.input[1]]
    hidden_shape = input_shape[:-1] + [gate_weight_shape[-1]]
    output_shape = input_shape[:-1] + [down_weight_shape[-1]]
    nodes = [
        helper.make_node("MatMul", ["input", "gate_weight"], ["gate"], name=gate.name),
        helper.make_node("Sigmoid", ["gate"], ["gate_sigmoid"], name="qwen_swiglu_sigmoid"),
        helper.make_node("Mul", ["gate", "gate_sigmoid"], ["gate_silu"], name="qwen_swiglu_silu"),
        helper.make_node("MatMul", ["input", "up_weight"], ["up"], name=up.name),
        helper.make_node("Mul", ["gate_silu", "up"], ["swiglu"], name="qwen_swiglu_mul"),
        helper.make_node("MatMul", ["swiglu", "down_weight"], ["output"], name=down.name),
    ]
    return make_model(
        "qwen25_fp16_mlp_reference",
        nodes,
        [value_info("input", FP16, input_shape)],
        [value_info("output", FP16, output_shape)],
        [
            zero_initializer("gate_weight", np.float16, gate_weight_shape),
            zero_initializer("up_weight", np.float16, up_weight_shape),
            zero_initializer("down_weight", np.float16, down_weight_shape),
        ],
    )


def build_qlinear_mlp(shapes: Dict[str, List[int]], gate: onnx.NodeProto,
                      down: onnx.NodeProto) -> onnx.ModelProto:
    input_shape = shapes[gate.input[0]]
    gate_weight_shape = shapes[gate.input[1]]
    down_weight_shape = shapes[down.input[1]]
    hidden_shape = input_shape[:-1] + [gate_weight_shape[-1]]
    output_shape = input_shape[:-1] + [down_weight_shape[-1]]
    nodes = [
        qlinear_matmul_node("qwen_mlp_gate_projection", "input_q", "gate_weight_q", "hidden_q", "gate"),
        qlinear_matmul_node("qwen_mlp_down_projection", "hidden_q", "down_weight_q", "output_q", "down"),
    ]
    initializers = [
        zero_initializer("gate_weight_q", np.uint8, gate_weight_shape),
        zero_initializer("down_weight_q", np.uint8, down_weight_shape),
        *qlinear_params("gate"),
        *qlinear_params("down"),
    ]
    return make_model(
        "qwen25_qlinear_mlp_npu_proxy",
        nodes,
        [value_info("input_q", U8, input_shape)],
        [value_info("output_q", U8, output_shape)],
        initializers,
    )


def build_fp16_lm_head(shapes: Dict[str, List[int]], lm_head: onnx.NodeProto) -> onnx.ModelProto:
    input_shape = shapes[lm_head.input[0]]
    weight_shape = shapes[lm_head.input[1]]
    output_shape = input_shape[:-1] + [weight_shape[-1]]
    return make_model(
        "qwen25_fp16_lm_head_reference",
        [helper.make_node("MatMul", ["input", "weight"], ["output"], name=lm_head.name)],
        [value_info("input", FP16, input_shape)],
        [value_info("output", FP16, output_shape)],
        [zero_initializer("weight", np.float16, weight_shape)],
    )


def build_qlinear_lm_head(shapes: Dict[str, List[int]], lm_head: onnx.NodeProto) -> onnx.ModelProto:
    input_shape = shapes[lm_head.input[0]]
    weight_shape = shapes[lm_head.input[1]]
    output_shape = input_shape[:-1] + [weight_shape[-1]]
    return make_model(
        "qwen25_qlinear_lm_head_npu_proxy",
        [qlinear_matmul_node("qwen_lm_head", "input_q", "weight_q", "output_q", "lm_head")],
        [value_info("input_q", U8, input_shape)],
        [value_info("output_q", U8, output_shape)],
        [zero_initializer("weight_q", np.uint8, weight_shape), *qlinear_params("lm_head")],
    )


def attention_shapes(
    shapes: Dict[str, List[int]], score: onnx.NodeProto, fallback_hidden_shape: Sequence[int]
) -> Tuple[List[int], List[int], List[int]]:
    q_shape = shapes.get(score.input[0])
    k_transposed_shape = shapes.get(score.input[1])
    if q_shape and k_transposed_shape:
        if len(q_shape) != 4 or len(k_transposed_shape) != 4:
            raise ValueError(
                f"expected rank-4 attention score operands, got {q_shape} x {k_transposed_shape}"
            )
        k_shape = list(k_transposed_shape)
        k_shape[-1], k_shape[-2] = k_shape[-2], k_shape[-1]
        return q_shape, k_shape, list(q_shape)

    # The exported graph omits value_info for some attention layout tensors.
    # Qwen2.5-0.5B uses head_dim=64; derive [B,H,S,64] from its [B,S,896]
    # residual input and verify that the hidden size has an integral head count.
    if len(fallback_hidden_shape) != 3:
        raise ValueError(f"expected [B,S,hidden] fallback shape, got {fallback_hidden_shape}")
    batch, sequence, hidden = fallback_hidden_shape
    head_dim = 64
    if hidden % head_dim != 0:
        raise ValueError(f"Qwen hidden size {hidden} is not divisible by head_dim {head_dim}")
    heads = hidden // head_dim
    q_shape = [batch, heads, sequence, head_dim]
    k_shape = [batch, heads, sequence, head_dim]
    return q_shape, k_shape, list(q_shape)


def build_fp16_attention(
    shapes: Dict[str, List[int]], score: onnx.NodeProto, fallback_hidden_shape: Sequence[int]
) -> onnx.ModelProto:
    q_shape, k_shape, v_shape = attention_shapes(shapes, score, fallback_hidden_shape)
    score_shape = q_shape[:-1] + [k_shape[-2]]
    output_shape = q_shape
    nodes = [
        helper.make_node("Transpose", ["key"], ["key_t"], perm=[0, 1, 3, 2], name="qwen_attention_key_transpose"),
        helper.make_node("MatMul", ["query", "key_t"], ["scores"], name=score.name),
        helper.make_node("Softmax", ["scores"], ["probabilities"], axis=-1, name="qwen_attention_softmax"),
        helper.make_node("MatMul", ["probabilities", "value"], ["output"], name="qwen_attention_value_matmul"),
    ]
    return make_model(
        "qwen25_fp16_attention_reference",
        nodes,
        [
            value_info("query", FP16, q_shape),
            value_info("key", FP16, k_shape),
            value_info("value", FP16, v_shape),
        ],
        [value_info("output", FP16, output_shape)],
        [],
    )


def build_qlinear_attention(
    shapes: Dict[str, List[int]], score: onnx.NodeProto, fallback_hidden_shape: Sequence[int]
) -> onnx.ModelProto:
    q_shape, k_shape, v_shape = attention_shapes(shapes, score, fallback_hidden_shape)
    score_shape = q_shape[:-1] + [k_shape[-2]]
    output_shape = q_shape
    nodes = [
        helper.make_node("DequantizeLinear", ["key_q", "layout_scale", "layout_zp"], ["key_f"], name="qwen_attention_key_dq"),
        helper.make_node("Transpose", ["key_f"], ["key_t_f"], perm=[0, 1, 3, 2], name="qwen_attention_key_transpose"),
        helper.make_node("QuantizeLinear", ["key_t_f", "layout_scale", "layout_zp"], ["key_t_q"], name="qwen_attention_key_q"),
        qlinear_matmul_node("qwen_attention_score", "query_q", "key_t_q", "scores_q", "score"),
        helper.make_node("DequantizeLinear", ["scores_q", "softmax_scale", "softmax_zp"], ["scores_f"], name="qwen_attention_scores_dq"),
        helper.make_node("Softmax", ["scores_f"], ["probabilities_f"], axis=-1, name="qwen_attention_softmax"),
        helper.make_node("QuantizeLinear", ["probabilities_f", "softmax_scale", "softmax_zp"], ["probabilities_q"], name="qwen_attention_probabilities_q"),
        qlinear_matmul_node("qwen_attention_value", "probabilities_q", "value_q", "output_q", "value"),
    ]
    initializers = [
        scalar_initializer("layout_scale", 0.02, np.float32),
        scalar_initializer("layout_zp", 128, np.uint8),
        scalar_initializer("softmax_scale", 1.0 / 255.0, np.float32),
        scalar_initializer("softmax_zp", 0, np.uint8),
        *qlinear_params("score"),
        *qlinear_params("value"),
    ]
    return make_model(
        "qwen25_qlinear_attention_npu_proxy",
        nodes,
        [
            value_info("query_q", U8, q_shape),
            value_info("key_q", U8, k_shape),
            value_info("value_q", U8, v_shape),
        ],
        [value_info("output_q", U8, output_shape)],
        initializers,
    )


def build_fp16_attention_core(
    shapes: Dict[str, List[int]], score: onnx.NodeProto, fallback_hidden_shape: Sequence[int]
) -> onnx.ModelProto:
    q_shape, k_shape, v_shape = attention_shapes(shapes, score, fallback_hidden_shape)
    k_transposed_shape = list(k_shape)
    k_transposed_shape[-1], k_transposed_shape[-2] = (
        k_transposed_shape[-2],
        k_transposed_shape[-1],
    )
    nodes = [
        helper.make_node("MatMul", ["query", "key_t"], ["scores"], name=score.name),
        helper.make_node("Softmax", ["scores"], ["probabilities"], axis=-1, name="qwen_attention_softmax"),
        helper.make_node("MatMul", ["probabilities", "value"], ["output"], name="qwen_attention_value_matmul"),
    ]
    return make_model(
        "qwen25_fp16_attention_core_reference",
        nodes,
        [
            value_info("query", FP16, q_shape),
            value_info("key_t", FP16, k_transposed_shape),
            value_info("value", FP16, v_shape),
        ],
        [value_info("output", FP16, q_shape)],
        [],
    )


def build_qlinear_attention_core(
    shapes: Dict[str, List[int]], score: onnx.NodeProto, fallback_hidden_shape: Sequence[int]
) -> onnx.ModelProto:
    q_shape, k_shape, v_shape = attention_shapes(shapes, score, fallback_hidden_shape)
    k_transposed_shape = list(k_shape)
    k_transposed_shape[-1], k_transposed_shape[-2] = (
        k_transposed_shape[-2],
        k_transposed_shape[-1],
    )
    nodes = [
        qlinear_matmul_node("qwen_attention_score", "query_q", "key_t_q", "scores_q", "score"),
        helper.make_node("DequantizeLinear", ["scores_q", "softmax_scale", "softmax_zp"], ["scores_f"], name="qwen_attention_scores_dq"),
        helper.make_node("Softmax", ["scores_f"], ["probabilities_f"], axis=-1, name="qwen_attention_softmax"),
        helper.make_node("QuantizeLinear", ["probabilities_f", "softmax_scale", "softmax_zp"], ["probabilities_q"], name="qwen_attention_probabilities_q"),
        qlinear_matmul_node("qwen_attention_value", "probabilities_q", "value_q", "output_q", "value"),
    ]
    initializers = [
        scalar_initializer("softmax_scale", 1.0 / 255.0, np.float32),
        scalar_initializer("softmax_zp", 0, np.uint8),
        *qlinear_params("score"),
        *qlinear_params("value"),
    ]
    return make_model(
        "qwen25_qlinear_attention_core_npu_proxy",
        nodes,
        [
            value_info("query_q", U8, q_shape),
            value_info("key_t_q", U8, k_transposed_shape),
            value_info("value_q", U8, v_shape),
        ],
        [value_info("output_q", U8, q_shape)],
        initializers,
    )


def node_summary(node: onnx.NodeProto, shapes: Dict[str, List[int]]) -> Dict[str, object]:
    return {
        "name": node.name,
        "inputs": [{"name": name, "shape": shapes.get(name)} for name in node.input],
        "outputs": [{"name": name, "shape": shapes.get(name)} for name in node.output],
    }


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_model).resolve()
    output_dir = Path(args.output_dir).resolve()
    model = load_source_model(source_path)
    shapes = build_shape_map(model)
    initializer_names = {item.name for item in model.graph.initializer}

    layer_prefix = f"/layers.{args.layer}/mlp/"
    gate = find_matmul(model, layer_prefix + "gate_proj/MatMul")
    up = find_matmul(model, layer_prefix + "up_proj/MatMul")
    down = find_matmul(model, layer_prefix + "down_proj/MatMul")
    lm_head = find_matmul(model, "/lm_head/MatMul")
    score = find_attention_score_matmul(model, args.layer, initializer_names)

    models = {
        "mlp_fp16_reference": build_fp16_mlp(shapes, gate, up, down),
        "mlp_qlinear_npu_proxy": build_qlinear_mlp(shapes, gate, down),
        "lm_head_fp16_reference": build_fp16_lm_head(shapes, lm_head),
        "lm_head_qlinear_npu_proxy": build_qlinear_lm_head(shapes, lm_head),
        "attention_fp16_reference": build_fp16_attention(shapes, score, shapes[gate.input[0]]),
        "attention_qlinear_npu_proxy": build_qlinear_attention(shapes, score, shapes[gate.input[0]]),
        "attention_core_fp16_reference": build_fp16_attention_core(shapes, score, shapes[gate.input[0]]),
        "attention_core_qlinear_npu_proxy": build_qlinear_attention_core(shapes, score, shapes[gate.input[0]]),
    }
    for label, output_model in models.items():
        save_model(output_model, output_dir / f"qwen25_{label}.onnx")

    manifest = {
        "source_model": str(source_path),
        "decoder_layer": args.layer,
        "selected_nodes": {
            "mlp_gate": node_summary(gate, shapes),
            "mlp_up": node_summary(up, shapes),
            "mlp_down": node_summary(down, shapes),
            "lm_head": node_summary(lm_head, shapes),
            "attention_score": node_summary(score, shapes),
        },
        "models": {
            label: {
                "path": str((output_dir / f"qwen25_{label}.onnx").resolve()),
                "node_count": len(output_model.graph.node),
                "initializer_count": len(output_model.graph.initializer),
            }
            for label, output_model in models.items()
        },
        "limitations": [
            "FP16 references retain Qwen topology and static shapes but use zero weights.",
            "INT8 proxies exercise current QLinearMatMul/QDQ NPU paths and do not claim Qwen numerical equivalence.",
            "The MLP proxy covers the 896x4864 and 4864x896 projections; it omits FP16 RMSNorm and SwiGLU.",
            "The attention proxy covers key transpose, score/value MatMul, and Softmax; it omits RoPE and causal-mask construction.",
            "The attention-core proxy receives a pre-transposed key tensor to isolate score/value MatMul and Softmax from the layout limitation.",
        ],
    }
    (output_dir / "subgraph_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "models": list(models)}, indent=2))


if __name__ == "__main__":
    main()
