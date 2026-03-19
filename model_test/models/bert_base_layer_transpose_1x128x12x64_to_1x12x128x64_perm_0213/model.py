import os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

MODEL_NAME = "bert_base_layer_transpose_1x128x12x64_to_1x12x128x64_perm_0213"
INPUT_SHAPE = (1, 128, 12, 64)
PERM = (0, 2, 1, 3)


def validate_dq_op_q_pattern(model, target_op_type):
    producer = {}
    consumers = {}
    for node in model.graph.node:
        for output in node.output:
            producer[output] = node
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    for node in model.graph.node:
        if node.op_type != target_op_type:
            continue
        has_dq_input = any(
            input_name in producer and producer[input_name].op_type == "DequantizeLinear"
            for input_name in node.input
        )
        has_q_output = any(
            consumer.op_type == "QuantizeLinear"
            for consumer in consumers.get(node.output[0], [])
        )
        if has_dq_input and has_q_output:
            return True
    return False


def build_qdq_transpose_model(model_path):
    scale_value = 0.039
    zero_point = 0
    wrap_bias = 1e-3
    output_shape = [INPUT_SHAPE[i] for i in PERM]

    initializers = [
        helper.make_tensor("scale", TensorProto.FLOAT, [], [scale_value]),
        helper.make_tensor("zero_point", TensorProto.INT8, [], [zero_point]),
        helper.make_tensor("wrap_add", TensorProto.FLOAT, [], [wrap_bias]),
        helper.make_tensor("wrap_sub", TensorProto.FLOAT, [], [-wrap_bias]),
    ]

    nodes = [
        helper.make_node("Add", ["input", "wrap_add"], ["wrapped_input"], name="node_add"),
        helper.make_node(
            "QuantizeLinear", ["wrapped_input", "scale", "zero_point"], ["input_q"],
            name="quantize_before_transpose",
        ),
        helper.make_node(
            "DequantizeLinear", ["input_q", "scale", "zero_point"], ["input_dq"],
            name="dequant_before_transpose",
        ),
        helper.make_node(
            "Transpose", ["input_dq"], ["transpose_fp32"], perm=list(PERM), name="node_transpose",
        ),
        helper.make_node(
            "QuantizeLinear", ["transpose_fp32", "scale", "zero_point"], ["transpose_q"],
            name="quantize_after_transpose",
        ),
        helper.make_node(
            "DequantizeLinear", ["transpose_q", "scale", "zero_point"], ["dequantized_output"],
            name="dequant_after_transpose",
        ),
        helper.make_node("Add", ["dequantized_output", "wrap_sub"], ["output"], name="node_sub"),
    ]

    graph = helper.make_graph(
        nodes,
        "transpose_qdq_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(INPUT_SHAPE))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, model_path)

    reloaded = onnx.load(model_path)
    if not validate_dq_op_q_pattern(reloaded, "Transpose"):
        raise RuntimeError("Generated Transpose model does not satisfy DQ -> Transpose -> Q pattern")


def main():
    workdir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workdir, "model.onnx")
    input_path = os.path.join(workdir, f"{MODEL_NAME}_input.bin")
    golden_path = os.path.join(workdir, f"{MODEL_NAME}_output_golden.bin")

    np.random.seed(2026)
    build_qdq_transpose_model(model_path)

    test_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    test_input.tofile(input_path)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: test_input})[0]
    output.astype(np.float32).tofile(golden_path)

    print("Generated files:")
    print(f"  {os.path.basename(model_path)}")
    print(f"  {os.path.basename(input_path)}")
    print(f"  {os.path.basename(golden_path)}")
    print(f"Case: {MODEL_NAME}, input shape: {INPUT_SHAPE}, perm: {PERM}")

if __name__ == "__main__":
    main()
