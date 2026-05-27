import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def symmetric_scale_from_max_abs(max_abs):
    """Convert one floating-point range into a symmetric int8 scale."""
    return float(max(max_abs / 127.0, 1e-6))


def quantize_to_int8(array, scale):
    """Round a float32 tensor into the signed int8 range under one scale."""
    quantized = np.clip(np.round(array / scale), -128, 127).astype(np.int8)
    return quantized


def validate_dq_op_q_pattern(model, target_op_type):
    """Verify that the target op is surrounded by DQ on inputs and Q on outputs."""
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


def build_qdq_matmul_model(input_shape, model_path, seed=2026):
    """Build the unified MatMul testcase with the same DQ -> MatMul -> Q topology as the original."""
    rng = np.random.default_rng(seed)
    hidden = input_shape[-1]
    wrap_bias = 1e-3

    weight_fp32 = rng.standard_normal(size=(hidden, hidden), dtype=np.float32) / np.sqrt(hidden)
    calib_input = rng.standard_normal(size=input_shape, dtype=np.float32)
    shifted_input = calib_input + wrap_bias
    calib_output = np.matmul(shifted_input, weight_fp32)

    input_scale = symmetric_scale_from_max_abs(np.max(np.abs(shifted_input)))
    weight_scale = symmetric_scale_from_max_abs(np.max(np.abs(weight_fp32)))
    # Leave a small safety margin so boundary values do not flip sign after int8 saturation.
    output_scale = symmetric_scale_from_max_abs(np.max(np.abs(calib_output)) * 1.05)

    weight_q = quantize_to_int8(weight_fp32, weight_scale)

    initializers = [
        helper.make_tensor("input_scale", TensorProto.FLOAT, [], [input_scale]),
        helper.make_tensor("weight_scale", TensorProto.FLOAT, [], [weight_scale]),
        helper.make_tensor("output_scale", TensorProto.FLOAT, [], [output_scale]),
        helper.make_tensor("zero_point", TensorProto.INT8, [], [0]),
        helper.make_tensor("wrap_add", TensorProto.FLOAT, [], [wrap_bias]),
        helper.make_tensor("wrap_sub", TensorProto.FLOAT, [], [-wrap_bias]),
        helper.make_tensor(
            "weight_q",
            TensorProto.INT8,
            list(weight_q.shape),
            weight_q.reshape(-1).tolist(),
        ),
    ]

    nodes = [
        helper.make_node("Add", ["input", "wrap_add"], ["wrapped_input"], name="node_add"),
        helper.make_node(
            "QuantizeLinear",
            ["wrapped_input", "input_scale", "zero_point"],
            ["input_q"],
            name="quantize_before_matmul",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["input_q", "input_scale", "zero_point"],
            ["input_dq"],
            name="dequant_before_matmul_input",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["weight_q", "weight_scale", "zero_point"],
            ["weight_dq"],
            name="dequant_before_matmul_weight",
        ),
        helper.make_node(
            "MatMul",
            ["input_dq", "weight_dq"],
            ["matmul_fp32"],
            name="node_matmul",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["matmul_fp32", "output_scale", "zero_point"],
            ["matmul_q"],
            name="quantize_after_matmul",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["matmul_q", "output_scale", "zero_point"],
            ["dequantized_output"],
            name="dequant_after_matmul",
        ),
        helper.make_node(
            "Add",
            ["dequantized_output", "wrap_sub"],
            ["output"],
            name="node_sub",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "unified_matmul_qdq_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(input_shape))],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, model_path)

    reloaded = onnx.load(model_path)
    if not validate_dq_op_q_pattern(reloaded, "MatMul"):
        raise RuntimeError("Generated MatMul model does not satisfy DQ -> MatMul -> Q pattern")


def main():
    """Export the unified MatMul testcase and serialize deterministic IO artifacts."""
    model_name = "unified_matmul"
    workdir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workdir, "model.onnx")
    input_path = os.path.join(workdir, f"{model_name}_input.bin")
    golden_path = os.path.join(workdir, f"{model_name}_output_golden.bin")

    input_shape = (1, 128, 768)

    np.random.seed(2026)

    build_qdq_matmul_model(input_shape, model_path)

    test_input = np.random.randn(*input_shape).astype(np.float32)
    test_input.tofile(input_path)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: test_input})[0]
    output.astype(np.float32).tofile(golden_path)

    model = onnx.load(model_path)
    if not validate_dq_op_q_pattern(model, "MatMul"):
        raise RuntimeError("Pattern 检查失败: 未找到 DQ -> MatMul -> Q")

    print("Generated files:")
    print(f"  {os.path.basename(model_path)}")
    print(f"  {os.path.basename(input_path)}")
    print(f"  {os.path.basename(golden_path)}")
    print(f"Input shape: {input_shape}, dtype: float32")
    print("Model structure: input -> Add -> Q -> DQ -> MatMul -> Q -> DQ -> Sub -> output")
    print("Pattern check: PASS (DequantizeLinear -> MatMul -> QuantizeLinear)")


if __name__ == "__main__":
    main()
