import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_common import print_generated_summary, save_float_io


def build_unified_maxpool_model(input_shape, model_path):
    """Build a QDQ-style MaxPool graph on the shared 4D benchmark shape."""
    scale_value = 0.039
    wrap_bias = 1e-3

    initializers = [
        helper.make_tensor("scale", TensorProto.FLOAT, [], [scale_value]),
        helper.make_tensor("zero_point", TensorProto.INT8, [], [0]),
        helper.make_tensor("wrap_add", TensorProto.FLOAT, [], [wrap_bias]),
        helper.make_tensor("wrap_sub", TensorProto.FLOAT, [], [-wrap_bias]),
    ]

    nodes = [
        helper.make_node("Add", ["input", "wrap_add"], ["wrapped_input"], name="node_add"),
        helper.make_node(
            "QuantizeLinear",
            ["wrapped_input", "scale", "zero_point"],
            ["input_q"],
            name="quantize_before_maxpool",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["input_q", "scale", "zero_point"],
            ["input_dq"],
            name="dequant_before_maxpool",
        ),
        helper.make_node(
            "MaxPool",
            ["input_dq"],
            ["maxpool_fp32"],
            kernel_shape=[2, 2],
            pads=[0, 0, 0, 0],
            strides=[2, 2],
            name="node_maxpool",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["maxpool_fp32", "scale", "zero_point"],
            ["maxpool_q"],
            name="quantize_after_maxpool",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["maxpool_q", "scale", "zero_point"],
            ["maxpool_dq"],
            name="dequant_after_maxpool",
        ),
        helper.make_node("Add", ["maxpool_dq", "wrap_sub"], ["output"], name="node_sub"),
    ]

    output_shape = [input_shape[0], input_shape[1], input_shape[2] // 2, input_shape[3] // 2]
    graph = helper.make_graph(
        nodes,
        "unified_maxpool_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, model_path)


def main():
    """Export the unified MaxPool case and serialize deterministic IO artifacts."""
    model_name = "unified_maxpool"
    input_shape = (1, 64, 32, 48)
    workdir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workdir, "model.onnx")
    input_path = os.path.join(workdir, f"{model_name}_input.bin")
    golden_path = os.path.join(workdir, f"{model_name}_output_golden.bin")

    np.random.seed(2026)
    build_unified_maxpool_model(input_shape, model_path)
    save_float_io(model_path, input_path, golden_path, np.random.randn(*input_shape).astype(np.float32))
    print_generated_summary(model_name, input_shape, ["Target op: MaxPoolSingleOut", "Output shape: (1, 64, 16, 24)"])


if __name__ == "__main__":
    main()
