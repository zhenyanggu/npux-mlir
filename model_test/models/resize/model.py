import os
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort


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


def build_qdq_resize_model(input_shape, model_path):
    """构建满足 DequantizeLinear -> Resize -> QuantizeLinear 的测试模型。"""
    scale_value = 0.039
    zero_point = 0
    wrap_bias = 1e-3

    initializers = [
        helper.make_tensor("scale", TensorProto.FLOAT, [], [scale_value]),
        helper.make_tensor("zero_point", TensorProto.INT8, [], [zero_point]),
        helper.make_tensor("wrap_add", TensorProto.FLOAT, [], [wrap_bias]),
        helper.make_tensor("wrap_sub", TensorProto.FLOAT, [], [-wrap_bias]),
        helper.make_tensor("roi", TensorProto.FLOAT, [0], []),
        helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]),
    ]

    nodes = [
        helper.make_node("Add", ["input", "wrap_add"], ["wrapped_input"], name="node_add"),
        helper.make_node(
            "QuantizeLinear",
            ["wrapped_input", "scale", "zero_point"],
            ["input_q"],
            name="quantize_before_resize",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["input_q", "scale", "zero_point"],
            ["input_dq"],
            name="dequant_before_resize",
        ),
        helper.make_node(
            "Resize",
            ["input_dq", "roi", "scales"],
            ["resize_fp32"],
            mode="nearest",
            name="node_resize",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["resize_fp32", "scale", "zero_point"],
            ["resize_q"],
            name="quantize_after_resize",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["resize_q", "scale", "zero_point"],
            ["dequantized_output"],
            name="dequant_after_resize",
        ),
        helper.make_node(
            "Add",
            ["dequantized_output", "wrap_sub"],
            ["output"],
            name="node_sub",
        ),
    ]

    output_shape = [input_shape[0], input_shape[1], input_shape[2] * 2, input_shape[3] * 2]
    graph = helper.make_graph(
        nodes,
        "resize_qdq_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, model_path)

    reloaded = onnx.load(model_path)
    if not validate_dq_op_q_pattern(reloaded, "Resize"):
        raise RuntimeError("Generated Resize model does not satisfy DQ -> Resize -> Q pattern")

    print(f"✓ 生成满足 DQ -> Resize -> Q 的模型: {model_path}")


def main():
    model_name = "resize"
    workdir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workdir, "model.onnx")
    input_path = os.path.join(workdir, f"{model_name}_input.bin")
    golden_path = os.path.join(workdir, f"{model_name}_output_golden.bin")

    input_shape = (1, 64, 56, 56)

    np.random.seed(2026)

    build_qdq_resize_model(input_shape, model_path)

    # 生成测试数据
    test_input = np.random.randn(*input_shape).astype(np.float32)
    test_input.tofile(input_path)

    # 使用 ORT 运行模型生成 golden 输出
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: test_input})[0]
    output.astype(np.float32).tofile(golden_path)

    print("\n✓ 生成文件:")
    print(f"  {os.path.basename(model_path)}")
    print(f"  {os.path.basename(input_path)}")
    print(f"  {os.path.basename(golden_path)}")
    print(f"\n✓ 输入 shape: {input_shape}, dtype: float32")
    print("✓ 模型结构: input -> Add -> Q -> DQ -> Resize -> Q -> DQ -> Sub -> output")

    model = onnx.load(model_path)
    if validate_dq_op_q_pattern(model, "Resize"):
        print("✓ Pattern 检查通过: DequantizeLinear -> Resize -> QuantizeLinear")
    else:
        raise RuntimeError("Pattern 检查失败: 未找到 DQ -> Resize -> Q")


if __name__ == "__main__":
    main()
