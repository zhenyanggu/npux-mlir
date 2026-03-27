import onnx_graphsurgeon as gs
import onnx
import numpy as np

# 1. 加载模型
graph = gs.import_onnx(onnx.load("vision_encoder.onnx"))

# 2. 遍历节点并替换目标 Flatten 节点
for node in graph.nodes:
    if node.name == "/vision_model/embeddings/Flatten_1":
        # 将操作类型改为 Reshape
        node.op = "Reshape"
        
        # 创建一个常量 shape 张量，值为 [-1, 1]
        shape_const = gs.Constant(
            name="flatten_1_reshape_shape", 
            values=np.array([-1, 1], dtype=np.int64)
        )
        
        # 修改节点输入：保留原数据输入，增加 shape 输入
        node.inputs = [node.inputs[0], shape_const]
        
        # 删除 Flatten 特有的 axis 属性
        if "axis" in node.attrs:
            del node.attrs["axis"]

# 3. 清理并保存图结构
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "vision_encoder_fixed.onnx")
print("模型修复完成，已保存为 vision_encoder_fixed.onnx")