#!/usr/bin/env python3
"""Analyze operators and MatMul/Gemm shapes in the Qwen2.5 prefill ONNX model."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import onnx


KNOWN_COMPILE_RISK_OPS = {
    "ScatterND",
    "NonZero",
    "Range",
    "Where",
    "Equal",
    "Less",
    "Greater",
    "TopK",
    "Loop",
    "If",
}

SHAPE_LAYOUT_OPS = {
    "Cast",
    "Concat",
    "Constant",
    "ConstantOfShape",
    "Expand",
    "Gather",
    "Reshape",
    "Shape",
    "Slice",
    "Squeeze",
    "Tile",
    "Transpose",
    "Unsqueeze",
}

NPU_INTEREST_OPS = {
    "MatMul",
    "Gemm",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Softmax",
    "ReduceMean",
    "Sqrt",
    "Pow",
    "Erf",
    "Sigmoid",
    "Tanh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--matmul-json", required=True)
    parser.add_argument("--unsupported-md", required=True)
    parser.add_argument("--top-initializers", type=int, default=20)
    parser.add_argument("--infer-shapes", action="store_true")
    parser.add_argument("--keep-inferred-model", default="")
    return parser.parse_args()


def dim_to_value(dim: onnx.TensorShapeProto.Dimension) -> Any:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return dim.dim_param
    return None


def value_info_shape(value_info: onnx.ValueInfoProto) -> Optional[List[Any]]:
    if not value_info.type.HasField("tensor_type"):
        return None
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    return [dim_to_value(dim) for dim in tensor_type.shape.dim]


def value_info_dtype(value_info: onnx.ValueInfoProto) -> Optional[str]:
    if not value_info.type.HasField("tensor_type"):
        return None
    elem_type = value_info.type.tensor_type.elem_type
    if elem_type == 0:
        return None
    return onnx.TensorProto.DataType.Name(elem_type)


def build_shape_dtype_maps(model: onnx.ModelProto) -> Tuple[Dict[str, List[Any]], Dict[str, str]]:
    shapes: Dict[str, List[Any]] = {}
    dtypes: Dict[str, str] = {}
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        shape = value_info_shape(item)
        dtype = value_info_dtype(item)
        if shape is not None:
            shapes[item.name] = shape
        if dtype is not None:
            dtypes[item.name] = dtype
    for init in model.graph.initializer:
        shapes[init.name] = [int(dim) for dim in init.dims]
        dtypes[init.name] = onnx.TensorProto.DataType.Name(init.data_type)
    return shapes, dtypes


def product(values: Iterable[Any]) -> Optional[int]:
    result = 1
    for value in values:
        if not isinstance(value, int):
            return None
        result *= value
    return result


def matmul_mkn(a_shape: Optional[List[Any]], b_shape: Optional[List[Any]]) -> Tuple[Any, Any, Any, List[Any]]:
    if not a_shape or not b_shape or len(a_shape) < 2 or len(b_shape) < 2:
        return None, None, None, []
    m = a_shape[-2]
    k = a_shape[-1]
    n = b_shape[-1]
    batch = a_shape[:-2]
    return m, k, n, batch


def get_attr(node: onnx.NodeProto, name: str) -> Any:
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return None


def analyze_matmul_nodes(model: onnx.ModelProto, shapes: Dict[str, List[Any]], dtypes: Dict[str, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    initializer_names = {item.name for item in model.graph.initializer}
    for index, node in enumerate(model.graph.node):
        if node.op_type not in {"MatMul", "Gemm"}:
            continue
        a_name = node.input[0] if len(node.input) > 0 else ""
        b_name = node.input[1] if len(node.input) > 1 else ""
        a_shape = shapes.get(a_name)
        b_shape = shapes.get(b_name)
        output_name = node.output[0] if node.output else ""
        output_shape = shapes.get(output_name)
        m, k, n, batch = matmul_mkn(a_shape, b_shape)
        row = {
            "index": index,
            "name": node.name,
            "op_type": node.op_type,
            "domain": node.domain,
            "a": a_name,
            "b": b_name,
            "output": output_name,
            "a_shape": a_shape,
            "b_shape": b_shape,
            "output_shape": output_shape,
            "a_dtype": dtypes.get(a_name),
            "b_dtype": dtypes.get(b_name),
            "output_dtype": dtypes.get(output_name),
            "b_is_initializer": b_name in initializer_names,
            "m": m,
            "k": k,
            "n": n,
            "batch": batch,
            "batch_elements": product(batch),
            "macs_per_batch": product([m, k, n]) if all(isinstance(x, int) for x in [m, k, n]) else None,
        }
        if node.op_type == "Gemm":
            row["alpha"] = get_attr(node, "alpha")
            row["beta"] = get_attr(node, "beta")
            row["transA"] = get_attr(node, "transA")
            row["transB"] = get_attr(node, "transB")
        rows.append(row)
    return rows


def summarize_shapes(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, Any, Any, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row.get("m"), row.get("k"), row.get("n"), json.dumps(row.get("batch", [])))
        if key not in grouped:
            grouped[key] = {
                "m": row.get("m"),
                "k": row.get("k"),
                "n": row.get("n"),
                "batch": row.get("batch", []),
                "count": 0,
                "b_initializer_count": 0,
                "example_names": [],
            }
        item = grouped[key]
        item["count"] += 1
        if row.get("b_is_initializer"):
            item["b_initializer_count"] += 1
        if len(item["example_names"]) < 5:
            item["example_names"].append(row.get("name") or row.get("output"))
    return sorted(grouped.values(), key=lambda item: (-item["count"], str(item["m"]), str(item["k"]), str(item["n"])))


def initializer_summary(model: onnx.ModelProto, limit: int) -> List[Dict[str, Any]]:
    rows = []
    for init in model.graph.initializer:
        byte_size = 0
        if init.raw_data:
            byte_size = len(init.raw_data)
        rows.append(
            {
                "name": init.name,
                "dtype": onnx.TensorProto.DataType.Name(init.data_type),
                "shape": [int(dim) for dim in init.dims],
                "raw_bytes": byte_size,
            }
        )
    return sorted(rows, key=lambda item: item["raw_bytes"], reverse=True)[:limit]


def write_json(path: str, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_unsupported_report(path: str, payload: Dict[str, Any]) -> None:
    lines = [
        "# Qwen2.5-0.5B Prefill ONNX Compile Risk Report",
        "",
        f"Model: `{payload['model']}`",
        f"Node count: `{payload['node_count']}`",
        f"Initializer count: `{payload['initializer_count']}`",
        "",
        "## Custom Domains",
        "",
    ]
    custom_domains = payload["custom_domains"]
    if custom_domains:
        for domain, count in custom_domains.items():
            lines.append(f"- `{domain}`: {count}")
    else:
        lines.append("- None")

    lines.extend(["", "## Risk Ops", ""])
    risk_ops = payload["risk_ops"]
    if risk_ops:
        for op_type, count in risk_ops.items():
            lines.append(f"- `{op_type}`: {count}")
    else:
        lines.append("- None from the current heuristic list")

    lines.extend(["", "## Notes", ""])
    lines.append("- Shape/layout ops are not necessarily unsupported, but they may dominate lowering complexity.")
    lines.append("- MatMul/Gemm details are stored in the MatMul shape JSON.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_path = args.model
    inferred_path = ""
    if args.infer_shapes:
        if args.keep_inferred_model:
            inferred_path = args.keep_inferred_model
        else:
            tmp = tempfile.NamedTemporaryFile(prefix="qwen25_prefill_inferred_", suffix=".onnx", delete=False)
            inferred_path = tmp.name
            tmp.close()
        onnx.shape_inference.infer_shapes_path(args.model, inferred_path, data_prop=True)
        model_path = inferred_path

    model = onnx.load(model_path)
    shapes, dtypes = build_shape_dtype_maps(model)

    op_counter = Counter(node.op_type for node in model.graph.node)
    domain_counter = Counter(node.domain or "ai.onnx" for node in model.graph.node)
    custom_domains = Counter(node.domain for node in model.graph.node if node.domain)
    risk_ops = Counter(node.op_type for node in model.graph.node if node.op_type in KNOWN_COMPILE_RISK_OPS)
    shape_layout_ops = Counter(node.op_type for node in model.graph.node if node.op_type in SHAPE_LAYOUT_OPS)
    npu_interest_ops = Counter(node.op_type for node in model.graph.node if node.op_type in NPU_INTEREST_OPS)
    matmul_rows = analyze_matmul_nodes(model, shapes, dtypes)

    summary = {
        "model": args.model,
        "analyzed_model": model_path,
        "shape_inference_enabled": bool(args.infer_shapes),
        "inferred_model": inferred_path,
        "ir_version": model.ir_version,
        "opset_import": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "value_info_count": len(model.graph.value_info),
        "op_counts": dict(op_counter.most_common()),
        "domain_counts": dict(domain_counter.most_common()),
        "custom_domains": dict(custom_domains.most_common()),
        "risk_ops": dict(risk_ops.most_common()),
        "shape_layout_ops": dict(shape_layout_ops.most_common()),
        "npu_interest_ops": dict(npu_interest_ops.most_common()),
        "matmul_total": sum(1 for item in matmul_rows if item["op_type"] == "MatMul"),
        "gemm_total": sum(1 for item in matmul_rows if item["op_type"] == "Gemm"),
        "matmul_or_gemm_total": len(matmul_rows),
        "matmul_or_gemm_with_static_b": sum(1 for item in matmul_rows if item["b_is_initializer"]),
        "top_initializers": initializer_summary(model, args.top_initializers),
    }

    matmul_payload = {
        "model": args.model,
        "analyzed_model": model_path,
        "shape_inference_enabled": bool(args.infer_shapes),
        "rows": matmul_rows,
        "shape_groups": summarize_shapes(matmul_rows),
    }

    write_json(args.summary_json, summary)
    write_json(args.matmul_json, matmul_payload)
    write_unsupported_report(args.unsupported_md, summary)
    print(json.dumps({
        "node_count": summary["node_count"],
        "matmul_total": summary["matmul_total"],
        "gemm_total": summary["gemm_total"],
        "custom_domains": summary["custom_domains"],
        "risk_ops": summary["risk_ops"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
