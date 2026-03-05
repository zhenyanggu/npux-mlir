import os
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


def build_quantized_maxpool_model(input_shape, model_path):
    """
    手动构建包含 int8 MaxPool 的量化模型
    结构: input(fp32) -> QuantizeLinear -> Identity(i8) -> MaxPool(i8) -> Identity(i8) -> DequantizeLinear -> output(fp32)
    """
    # 量化参数 (对称量化，zero_point=0)
    scale_value = 0.039  # 基于 [-5, 5] 范围估算: 5/127 ≈ 0.039
    zero_point = 0
    
    # 创建常量节点
    scale_init = helper.make_tensor("scale", TensorProto.FLOAT, [], [scale_value])
    zp_init = helper.make_tensor("zero_point", TensorProto.INT8, [], [zero_point])
    
    # 定义节点
    nodes = [
        # 1. QuantizeLinear: fp32 -> i8
        helper.make_node(
            "QuantizeLinear",
            inputs=["input", "scale", "zero_point"],
            outputs=["quantized_input"],
            name="quantize_input"
        ),
        
        # 2. Identity (i8): 前置包裹
        helper.make_node(
            "Identity",
            inputs=["quantized_input"],
            outputs=["identity_before_output"],
            name="identity_before_maxpool"
        ),
        
        # 3. MaxPool (i8): 核心算子
        helper.make_node(
            "MaxPool",
            inputs=["identity_before_output"],
            outputs=["maxpooled"],
            kernel_shape=[2, 2],
            pads=[0, 0, 0, 0],
            strides=[2, 2],
            name="maxpool_i8"
        ),
        
        # 4. Identity (i8): 后置包裹
        helper.make_node(
            "Identity",
            inputs=["maxpooled"],
            outputs=["identity_after_output"],
            name="identity_after_maxpool"
        ),
        
        # 5. DequantizeLinear: i8 -> fp32
        helper.make_node(
            "DequantizeLinear",
            inputs=["identity_after_output", "scale", "zero_point"],
            outputs=["output"],
            name="dequantize_output"
        ),
    ]
    
    # 定义输入输出
    graph_inputs = [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
    ]
    
    # MaxPool 输出 shape: (1, 64, 28, 28)
    output_shape = [input_shape[0], input_shape[1], input_shape[2] // 2, input_shape[3] // 2]
    graph_outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    ]
    
    # 创建图
    graph = helper.make_graph(
        nodes,
        "maxpool_int8_graph",
        graph_inputs,
        graph_outputs,
        [scale_init, zp_init]
    )
    
    # 创建模型
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    
    # 检查并保存
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    print(f"✓ 手动构建的 int8 MaxPool 模型已保存: {model_path}")


def main():
    model_name = "maxpool"
    workdir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workdir, "model.onnx")
    input_path = os.path.join(workdir, f"{model_name}_input.bin")
    golden_path = os.path.join(workdir, f"{model_name}_output_golden.bin")

    input_shape = (1, 64, 56, 56)

    np.random.seed(2026)

    # 手动构建包含 int8 MaxPool 的模型
    build_quantized_maxpool_model(input_shape, model_path)

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
    print(f"✓ 模型结构: input(fp32) -> Q -> Identity(i8) -> MaxPool(i8) -> Identity(i8) -> DQ -> output(fp32)")
    
    # 验证模型中确实有 int8 的 MaxPool
    model = onnx.load(model_path)
    for node in model.graph.node:
        if node.op_type == "MaxPool":
            print(f"✓ 找到 MaxPool 节点: {node.name}")
            print(f"  输入: {node.input[0]}")
            print(f"  输出: {node.output[0]}")


if __name__ == "__main__":
    main()
