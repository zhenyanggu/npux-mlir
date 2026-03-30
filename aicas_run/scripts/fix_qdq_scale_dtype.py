#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Tuple

import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Q/DQ scale initializers from float16 to float32."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-load", action="store_true")
    parser.add_argument("--providers", default="CPUExecutionProvider")
    return parser.parse_args()


def convert_qdq_scales(model: onnx.ModelProto) -> Tuple[onnx.ModelProto, List[str]]:
    initializers: Dict[str, onnx.TensorProto] = {item.name: item for item in model.graph.initializer}
    names_to_fix = set()

    for node in model.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        if len(node.input) < 2:
            continue
        scale_name = node.input[1]
        tensor = initializers.get(scale_name)
        if tensor is None:
            continue
        if tensor.data_type != TensorProto.FLOAT16:
            continue
        names_to_fix.add(scale_name)

    if not names_to_fix:
        return model, []

    new_initializers = []
    fixed_names: List[str] = []
    for tensor in model.graph.initializer:
        if tensor.name not in names_to_fix:
            new_initializers.append(tensor)
            continue
        array = numpy_helper.to_array(tensor).astype("float32", copy=False)
        new_tensor = numpy_helper.from_array(array, name=tensor.name)
        new_initializers.append(new_tensor)
        fixed_names.append(tensor.name)

    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)
    return model, fixed_names


def main() -> int:
    args = parse_args()
    model = onnx.load(args.input)
    model, fixed_names = convert_qdq_scales(model)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    onnx.save(model, args.output)
    print(f"saved fixed model: {args.output}")
    print(f"fixed qdq scale tensors: {len(fixed_names)}")
    if fixed_names:
        print(f"sample fixed tensors: {fixed_names[:10]}")

    if args.check_load:
        providers = [item.strip() for item in args.providers.split(",") if item.strip()]
        ort.InferenceSession(args.output, providers=providers)
        print("ort load check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
