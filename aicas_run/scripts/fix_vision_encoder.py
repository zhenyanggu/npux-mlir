import argparse

import numpy as np
import onnx
from onnx import helper


TARGET_NODE_NAME = "/vision_model/embeddings/Flatten_1"
RESHAPE_SHAPE_NAME = "flatten_1_reshape_shape"
DEFAULT_INPUT = "aicas_run/models/vision_encoder.onnx"
DEFAULT_OUTPUT = "aicas_run/models/vision_encoder_fixed.onnx"


def upsert_initializer(graph, name, values):
    for idx, initializer in enumerate(graph.initializer):
        if initializer.name == name:
            del graph.initializer[idx]
            break
    graph.initializer.append(
        helper.make_tensor(
            name=name,
            data_type=onnx.TensorProto.INT64,
            dims=list(values.shape),
            vals=values,
        )
    )


def rewrite_flatten_to_reshape(model):
    graph = model.graph
    found = False
    for node in graph.node:
        if node.name != TARGET_NODE_NAME:
            continue
        found = True
        if len(node.input) != 1:
            raise RuntimeError(
                f"unexpected input count for {TARGET_NODE_NAME}: {len(node.input)}"
            )
        node.op_type = "Reshape"
        del node.attribute[:]
        node.input.append(RESHAPE_SHAPE_NAME)
        break
    if not found:
        raise RuntimeError(f"target node not found: {TARGET_NODE_NAME}")

    upsert_initializer(
        graph, RESHAPE_SHAPE_NAME, np.array([-1, 1], dtype=np.int64)
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Replace Flatten_1(axis=2) with Reshape([-1, 1])."
    )
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model = onnx.load(args.input)
    fixed_model = rewrite_flatten_to_reshape(model)
    onnx.save(fixed_model, args.output)
    print(f"模型修复完成，已保存为 {args.output}")


if __name__ == "__main__":
    main()
