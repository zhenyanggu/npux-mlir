#!/usr/bin/env python3
"""Inspect ONNX model inputs and outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def tensor_value_info_to_dict(value_info: onnx.ValueInfoProto) -> Dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    shape: List[Any] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    elem_type = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
    return {"name": value_info.name, "elem_type": elem_type, "shape": shape}


def main() -> None:
    args = parse_args()
    model = onnx.load(args.model)
    payload = {
        "model": args.model,
        "ir_version": model.ir_version,
        "opset_import": [
            {"domain": item.domain, "version": item.version} for item in model.opset_import
        ],
        "inputs": [tensor_value_info_to_dict(item) for item in model.graph.input],
        "outputs": [tensor_value_info_to_dict(item) for item in model.graph.output],
        "initializer_count": len(model.graph.initializer),
        "node_count": len(model.graph.node),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
