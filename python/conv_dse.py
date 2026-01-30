import math
import json
import argparse
import os
import onnx
from onnx import numpy_helper
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# ==========================================
# 1. 硬件定义 (保持不变)
# ==========================================
@dataclass
class HardwareConfig:
    spm_size_bytes: int
    acc_size_bytes: int
    
    dtype_input: int = 1     # int8
    dtype_acc: int = 4       # int32
    
    # 物理阵列参数 (不可变)
    u_oh: int = 4
    u_ow: int = 8
    u_oc: int = 32
    u_ic: int = 32

    measured_latency_us: float = 15.579
    measured_bandwidth_mbps: float = 1133.40 
    measured_conv_us = 27

# ==========================================
# 2. 层参数 (保持不变)
# ==========================================
@dataclass
class LayerParams:
    name: str
    H: int; W: int; IC: int; OC: int
    K_h: int; K_w: int; S: int; P: int

    @property
    def OH(self): 
        return (self.H + 2 * self.P - self.K_h) // self.S + 1
    @property
    def OW(self): 
        return (self.W + 2 * self.P - self.K_w) // self.S + 1

# ==========================================
# 3. ONNX 解析器 (保持不变)
# ==========================================
class OnnxModelParser:
    @staticmethod
    def parse(onnx_path: str, input_shape_override: List[int] = None) -> List[LayerParams]:
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        print(f"[Parser] Loading model from {onnx_path}...")
        model = onnx.load(onnx_path)
        graph = model.graph
        
        # --- 关键修改开始 ---
        # 1. 获取模型原始输入维度
        input_tensor = graph.input[0]
        # 获取原始维度列表 (如果是动态的，dim_value 可能是 0 或 -1)
        orig_dims = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
        
        # 2. 判断是否需要覆盖
        if input_shape_override:
            print(f"[Parser] Overriding input shape to: {input_shape_override}")
            # 强制覆盖
            if input_tensor.type.tensor_type.shape.dim:
                # 确保长度匹配，否则可能出错 (例如 NCHW)
                if len(input_shape_override) != len(orig_dims):
                    print(f"[Warning] Override shape length {len(input_shape_override)} "
                          f"!= model input rank {len(orig_dims)}")
                
                # 清除旧维度并设置新维度
                for i, dim_val in enumerate(input_shape_override):
                    if i < len(input_tensor.type.tensor_type.shape.dim):
                        input_tensor.type.tensor_type.shape.dim[i].dim_value = dim_val
        else:
            # 如果没有提供 override，检查原始维度是否合法 (全为正整数)
            is_dynamic = any(d <= 0 for d in orig_dims)
            if is_dynamic:
                raise ValueError(
                    f"Model has dynamic input shape {orig_dims}. "
                    "You MUST provide a static shape using --shape (e.g., --shape 1,3,224,224)"
                )
            else:
                print(f"[Parser] Using model static shape: {orig_dims}")
        # --- 关键修改结束 ---

        # 3. Shape Inference (推断中间层形状)
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"[Warning] Shape inference failed: {e}. Dimensions might be inaccurate.")

        # 重新获取 graph (因为 infer_shapes 可能返回新对象)
        graph = model.graph
        value_info = {vi.name: vi for vi in graph.value_info}
        value_info.update({vi.name: vi for vi in graph.input})
        value_info.update({vi.name: vi for vi in graph.output})
        initializers = {init.name: init for init in graph.initializer}

        layers = []
        
        for node in graph.node:
            if node.op_type == 'Conv':
                attr = {a.name: a for a in node.attribute}
                kernel_shape = attr['kernel_shape'].ints
                k_h, k_w = kernel_shape[0], kernel_shape[1]
                strides = attr['strides'].ints if 'strides' in attr else [1, 1]
                s = strides[0]
                pads = attr['pads'].ints if 'pads' in attr else [0, 0, 0, 0]
                # 处理 asymmetric padding，这里简化处理，取 max 或者第一个
                p = pads[0] 
                
                input_name = node.input[0]
                if input_name in value_info:
                    input_shape = value_info[input_name].type.tensor_type.shape.dim
                    # NCHW
                    if len(input_shape) >= 4:
                        ic = input_shape[1].dim_value
                        h = input_shape[2].dim_value
                        w = input_shape[3].dim_value
                    else:
                        print(f"[Warning] Layer {node.name} input rank < 4. Skipping.")
                        continue
                else:
                    print(f"[Warning] Could not find shape for input {input_name}, skipping layer {node.name}")
                    continue

                weight_name = node.input[1]
                if weight_name in initializers:
                    weight_tensor = initializers[weight_name]
                    oc = weight_tensor.dims[0]
                elif weight_name in value_info:
                     weight_shape = value_info[weight_name].type.tensor_type.shape.dim
                     oc = weight_shape[0].dim_value
                else:
                    print(f"[Warning] Could not determine weights for {node.name}")
                    continue

                # 再次检查 H/W 是否有效
                if h <= 0 or w <= 0:
                    print(f"[Warning] Shape inference returned invalid dims for {node.name} (H={h}, W={w}). "
                          "Check if input shape is correct.")
                    continue

                layer = LayerParams(name=node.name, H=h, W=w, IC=ic, OC=oc, K_h=k_h, K_w=k_w, S=s, P=p)
                layers.append(layer)

        return layers

# ==========================================
# 4. Cost Model (保持不变)
# ==========================================
class AdvancedCostModel:
    def __init__(self, hw: HardwareConfig):
        self.hw = hw

    def get_raw_input_tile_dim(self, t_oh, t_ow, layer):
        t_ih = (t_oh - 1) * layer.S + layer.K_h
        t_iw = (t_ow - 1) * layer.S + layer.K_w
        return t_ih, t_iw

    def evaluate(self, layer: LayerParams, t_oh, t_ow, t_oc, t_ic):
        real_tile_ic = max(32,min(t_ic, layer.IC))
        real_tile_oc = max(32,min(t_oc, layer.OC))

        # 2. ACC 空间检查
        acc_needed = t_oh * t_ow * real_tile_oc * self.hw.dtype_acc
        
        # 3. SPM 空间检查
        raw_t_ih, raw_t_iw = self.get_raw_input_tile_dim(t_oh, t_ow, layer)
        input_needed = raw_t_ih * raw_t_iw * real_tile_ic * self.hw.dtype_input
        weight_needed = layer.K_h * layer.K_w * real_tile_ic * real_tile_oc * self.hw.dtype_input
        output_needed = t_oh * t_ow * real_tile_oc * self.hw.dtype_input
        
        spm_needed = input_needed + weight_needed
        if output_needed > spm_needed:
            spm_needed = output_needed

        # 4. 判定是否溢出
        if acc_needed > self.hw.acc_size_bytes:
            return None # ACC OOM
        if spm_needed > self.hw.spm_size_bytes:
            return None # SPM OOM

        # 5. 性能计算 (Latency)
        n_h = math.ceil(layer.OH / t_oh)
        n_w = math.ceil(layer.OW / t_ow)
        n_spatial = n_h * n_w
        n_oc = math.ceil(layer.OC / t_oc)
        n_ic = math.ceil(layer.IC / t_ic)
        
        # Traffic Calculation
        traffic_in = input_needed * n_spatial * n_oc * n_ic
        traffic_wgt = weight_needed * n_spatial * n_oc * n_ic
        traffic_out = layer.OH * layer.OW * layer.OC * self.hw.dtype_input
        traffic_bias = layer.OC * self.hw.dtype_acc
        
        total_traffic_mb = (traffic_in + traffic_wgt + traffic_out + traffic_bias) / 1024**2
        time_transfer_ms = (total_traffic_mb / self.hw.measured_bandwidth_mbps) * 1000
        
        # Compute / Instruction Time
        total_tiles = n_spatial * n_oc * n_ic
        time_inst_ms = (total_tiles * self.hw.measured_latency_us) / 1000
        
        est_latency_ms = time_transfer_ms + time_inst_ms

        return {
            'est_latency_ms': est_latency_ms,
            'spm_util': spm_needed / self.hw.spm_size_bytes,
            'acc_util': acc_needed / self.hw.acc_size_bytes,
            't_oh': t_oh, 't_ow': t_ow, 't_ic': t_ic, 't_oc': t_oc,
            'config_str': f"{t_oh}x{t_ow}_{t_ic}x{t_oc}"
        }

# ==========================================
# 5. DSE 探索逻辑
# ==========================================
class DesignSpaceExplorer:
    def __init__(self, layers: List[LayerParams], total_mem_bytes: int):
        self.layers = layers
        self.total_mem_bytes = total_mem_bytes
        self.base_hw = HardwareConfig(0, 0)

    def search_layer(self, layer: LayerParams, model: AdvancedCostModel):
        min_lat = float('inf')
        best_res = None
        
        def safe_range(base_dim, max_dim, step):
            limit = min(max_dim + step, 128) 
            return range(step, limit, step)

        r_oh = safe_range(self.base_hw.u_oh, layer.OH, self.base_hw.u_oh)
        r_ow = safe_range(self.base_hw.u_ow, layer.OW, self.base_hw.u_ow)
        r_ic = safe_range(self.base_hw.u_ic, layer.IC, self.base_hw.u_ic)
        r_oc = safe_range(self.base_hw.u_oc, layer.OC, self.base_hw.u_oc)

        for t_ic in r_ic:
            for t_oc in r_oc:
                for t_oh in r_oh:
                    for t_ow in r_ow:
                        res = model.evaluate(layer, t_oh, t_ow, t_oc, t_ic)
                        if res:
                            if res['est_latency_ms'] < min_lat:
                                min_lat = res['est_latency_ms']
                                best_res = res
        return best_res

    def run(self):
        print(f"Starting DSE (Total Mem: {self.total_mem_bytes/1024:.0f} KB)...")
        best_global = None
        min_global_lat = float('inf')

        for ratio in [i/10.0 for i in range(1, 10)]:
            spm = int(self.total_mem_bytes * ratio)
            acc = self.total_mem_bytes - spm
            
            hw = HardwareConfig(spm, acc)
            model = AdvancedCostModel(hw)
            
            total_lat = 0
            layer_data = []
            is_valid_config = True
            
            for layer in self.layers:
                res = self.search_layer(layer, model)
                if not res:
                    is_valid_config = False
                    break
                total_lat += res['est_latency_ms']
                layer_data.append({
                    'layer_name': layer.name,
                    'input_shape': f"{layer.IC}x{layer.H}x{layer.W}",
                    'output_shape': f"{layer.OC}x{layer.OH}x{layer.OW}",
                    'tile_params': {
                        't_oh': res['t_oh'], 't_ow': res['t_ow'], 
                        't_ic': res['t_ic'], 't_oc': res['t_oc']
                    },
                    'latency_ms': res['est_latency_ms'],
                    'spm_util': res['spm_util'],
                    'acc_util': res['acc_util']
                })
            
            if is_valid_config:
                print(f"  Ratio {ratio:.1f} (SPM:{spm//1024}k/ACC:{acc//1024}k) -> {total_lat:.2f} ms")
                if total_lat < min_global_lat:
                    min_global_lat = total_lat
                    best_global = {
                        'best_ratio': ratio,
                        'hw_config': {'spm_bytes': spm, 'acc_bytes': acc},
                        'total_latency_ms': total_lat,
                        'layers': layer_data
                    }
            else:
                print(f"  Ratio {ratio:.1f} -> Failed")

        return best_global

# ==========================================
# 6. JSON 导出工具
# ==========================================
def export_to_json(data, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\n✅ Results exported to {filename}")
    except Exception as e:
        print(f"❌ Failed to export JSON: {e}")

# ==========================================
# 7. 主执行逻辑 (修改后)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSE Tool for ONNX Models")
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help='Path to the input ONNX model file')
    parser.add_argument('-o', '--output', type=str, required=False, 
                        help='Path to the output JSON file')
    parser.add_argument('--mem', type=int, default=512, 
                        help='Total memory size in KB (default: 512)')
    # 新增 shape 参数
    parser.add_argument('--shape', type=str, default=None,
                        help='Override input shape, format: N,C,H,W (e.g. 1,3,224,224). '
                             'Required if model has dynamic input.')
    
    args = parser.parse_args()

    input_path = args.input
    total_mem_bytes = args.mem * 1024

    # 处理 shape 参数字符串转列表
    shape_override = None
    if args.shape:
        try:
            shape_override = [int(x) for x in args.shape.split(',')]
        except ValueError:
            print("❌ Error: Invalid format for --shape. Use comma separated numbers like 1,3,224,224")
            exit(1)

    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}.json"

    # 1. 解析 ONNX
    try:
        # 这里不再把 override 写死，而是传 args 解析出来的结果 (None 或 List)
        layers = OnnxModelParser.parse(input_path, input_shape_override=shape_override)
    except ValueError as e:
        print(f"❌ Input Shape Error: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Failed to parse ONNX model: {e}")
        exit(1)
    
    if not layers:
        print("❌ No valid convolutional layers found.")
        exit(1)

    print(f"\nModel: {input_path} ({len(layers)} Conv Layers detected)")
    print(f"Output: {output_path}")
    
    # 2. 运行 DSE
    dse = DesignSpaceExplorer(layers, total_mem_bytes)
    best = dse.run()

    # 3. 输出
    if best:
        print("\n" + "="*60)
        print(f"✅ Best Configuration found")
        print(f"   Total Latency: {best['total_latency_ms']:.2f} ms")
        print("="*60)
        export_to_json(best, output_path)
    else:
        print("\n❌ All configurations failed.")