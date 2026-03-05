import os
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


def build_quantized_averagepool_model(input_shape, model_path):
    """
    手动构建带 int8 链路的 AveragePool 量化模型
    注意: ONNX 标准 AveragePool 不支持 int8 输入，因此采用 QDQ 包裹结构
    结构:
    input(fp32) -> Q(i8) -> Identity(i8) -> DQ(fp32) -> AveragePool(fp32)
    -> Q(i8) -> Identity(i8) -> DQ(fp32) -> output(fp32)
    """
    scale_value = 0.039
    zero_point = 0

    scale_init = helper.make_tensor("scale", TensorProto.FLOAT, [], [scale_value])
    zp_init = helper.make_tensor("zero_point", TensorProto.INT8, [], [zero_point])

    nodes = [
        helper.make_node(
            "QuantizeLinear",
            inputs=["input", "scale", "zero_point"],
            outputs=["quantized_input"],
            name="quantize_input",
        ),
        helper.make_node(
            "Identity",
            inputs=["quantized_input"],
            outputs=["identity_before_output"],
            name="identity_before_averagepool",
        ),
        helper.make_node(
            "DequantizeLinear",
            inputs=["identity_before_output", "scale", "zero_point"],
            outputs=["dequant_before_pool"],
            name="dequant_before_averagepool",
        ),
        helper.make_node(
            "AveragePool",
            inputs=["dequant_before_pool"],
            outputs=["pooled_fp32"],
            kernel_shape=[2, 2],
            pads=[0, 0, 0, 0],
            strides=[2, 2],
            name="averagepool_fp32",
        ),
        helper.make_node(
            "QuantizeLinear",
            inputs=["pooled_fp32", "scale", "zero_point"],
            outputs=["pooled_i8"],
            name="quantize_after_averagepool",
        ),
        helper.make_node(
            "Identity",
            inputs=["pooled_i8"],
            outputs=["identity_after_output"],
            name="identity_after_averagepool",
        ),
        helper.make_node(
            "DequantizeLinear",
            inputs=["identity_after_output", "scale", "zero_point"],
            outputs=["output"],
            name="dequantize_output",
        ),
    ]

    graph_inputs = [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
    ]

    output_shape = [
        input_shape[0],
        input_shape[1],
        input_shape[2] // 2,
        input_shape[3] // 2,
    ]
    graph_outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    ]

    graph = helper.make_graph(
        nodes,
        "averagepool_int8_graph",
        graph_inputs,
        graph_outputs,
        [scale_init, zp_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    print(f"✓ 手动构建的 AveragePool(QDQ) 模型已保存: {model_path}")


def main():
    model_name = "averagepool"
    workdir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workdir, "model.onnx")
    input_path = os.path.join(workdir, f"{model_name}_input.bin")
    golden_path = os.path.join(workdir, f"{model_name}_output_golden.bin")

    input_shape = (1, 64, 56, 56)

    np.random.seed(2026)

    build_quantized_averagepool_model(input_shape, model_path)

    test_input = np.random.randn(*input_shape).astype(np.float32)
    test_input.tofile(input_path)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: test_input})[0]
    output.astype(np.float32).tofile(golden_path)

    print("\n✓ 生成文件:")
    print(f"  {os.path.basename(model_path)}")
    print(f"  {os.path.basename(input_path)}")
    print(f"  {os.path.basename(golden_path)}")
    print(f"\n✓ 输入 shape: {input_shape}, dtype: float32")
    print("✓ 模型结构: input(fp32) -> Q(i8) -> Identity(i8) -> DQ -> AveragePool(fp32) -> Q(i8) -> Identity(i8) -> DQ -> output(fp32)")

    model = onnx.load(model_path)
    for node in model.graph.node:
        if node.op_type == "AveragePool":
            print(f"✓ 找到 AveragePool 节点: {node.name}")
            print(f"  输入: {node.input[0]}")
            print(f"  输出: {node.output[0]}")


if __name__ == "__main__":
    main()
