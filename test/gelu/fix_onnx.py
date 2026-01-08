import onnx
from onnx import numpy_helper
import numpy as np

def sanitize_onnx(input_path, output_path):
    print(f"Loading {input_path}...")
    model = onnx.load(input_path)
    graph = model.graph

    initializers = {t.name: t for t in graph.initializer}
    
    cleaned_count = 0

    for node in graph.node:
        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            # 1. 处理 Scale (Input[1])
            if len(node.input) > 1:
                scale_name = node.input[1]
                if scale_name in initializers:
                    scale_tensor = initializers[scale_name]
                    scale_arr = numpy_helper.to_array(scale_tensor)
                    
                    # 只要是 float64，或者 Rank != 0，都强制转为 float32 标量
                    # 注意：这里加了 dtype=np.float32
                    if scale_arr.dtype == np.float64 or scale_arr.ndim > 0:
                        print(f"[{node.name}] Fixing Scale: {scale_arr.dtype} shape {scale_arr.shape} -> float32 Scalar")
                        
                        # 获取数值 (如果是数组取第一个，如果是标量直接取)
                        scalar_val = scale_arr.flatten()[0] if scale_arr.ndim > 0 else scale_arr.item()
                        
                        # [关键修正] 显式指定 dtype=np.float32
                        new_scale_tensor = numpy_helper.from_array(
                            np.array(scalar_val, dtype=np.float32), 
                            name=scale_name
                        )
                        
                        graph.initializer.remove(scale_tensor)
                        graph.initializer.append(new_scale_tensor)
                        initializers[scale_name] = new_scale_tensor
                        cleaned_count += 1
            
            # 2. 处理 Zero Point (Input[2])
            if len(node.input) > 2:
                zp_name = node.input[2]
                if zp_name in initializers:
                    zp_tensor = initializers[zp_name]
                    zp_arr = numpy_helper.to_array(zp_tensor)
                    
                    # ZP 也顺手检查一下，保持标量
                    if zp_arr.ndim > 0:
                        print(f"[{node.name}] Fixing ZP Rank: {zp_arr.shape} -> Scalar")
                        scalar_val = zp_arr.flatten()[0]
                        # ZP 类型通常是 int8/uint8，保持原类型即可
                        new_zp_tensor = numpy_helper.from_array(
                            np.array(scalar_val, dtype=zp_arr.dtype), 
                            name=zp_name
                        )
                        graph.initializer.remove(zp_tensor)
                        graph.initializer.append(new_zp_tensor)
                        initializers[zp_name] = new_zp_tensor

            # 3. 移除 axis
            for attr in list(node.attribute):
                if attr.name == "axis":
                    print(f"[{node.name}] Removing 'axis' attribute")
                    node.attribute.remove(attr)

    print(f"Fixed {cleaned_count} tensors.")
    print(f"Saving to {output_path}...")
    onnx.save(model, output_path)
    print("Done.")

if __name__ == "__main__":
    # 使用你之前导出的原始 ONNX (注意输入文件名)
    sanitize_onnx("model.onnx", "fixed_model.onnx")