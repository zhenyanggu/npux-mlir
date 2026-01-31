import math
import json
import argparse
import os
import onnx
from onnx import numpy_helper
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# ==========================================
# 1. 硬件定义 (更新为 Manual VGG 版的参数)
# ==========================================
@dataclass
class HardwareConfig:
    spm_size_bytes: int
    acc_size_bytes: int
    
    dtype_input: int = 1     # int8
    dtype_acc: int = 4       # int32
    
    # 物理阵列参数
    sys_array_size: int = 32 

    measured_bandwidth_mbps: float = 1300.0 
    
    # 原子操作参数 (Manual VGG Algo)
    # 原子操作定义：计算 32(OC) x 32(IC) x 32(Spatial Output) 的卷积
    measured_conv_us: float = 5.0 
    # 指令开销
    instr_overhead_us: float = 1.54

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
        
        # 1. 获取模型原始输入维度
        input_tensor = graph.input[0]
        orig_dims = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
        
        # 2. 判断是否需要覆盖
        if input_shape_override:
            print(f"[Parser] Overriding input shape to: {input_shape_override}")
            if input_tensor.type.tensor_type.shape.dim:
                if len(input_shape_override) != len(orig_dims):
                    print(f"[Warning] Override shape length {len(input_shape_override)} "
                          f"!= model input rank {len(orig_dims)}")
                for i, dim_val in enumerate(input_shape_override):
                    if i < len(input_tensor.type.tensor_type.shape.dim):
                        input_tensor.type.tensor_type.shape.dim[i].dim_value = dim_val
        else:
            is_dynamic = any(d <= 0 for d in orig_dims)
            if is_dynamic:
                raise ValueError(f"Model has dynamic input shape {orig_dims}. Please use --shape.")
            else:
                print(f"[Parser] Using model static shape: {orig_dims}")

        # 3. Shape Inference
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"[Warning] Shape inference failed: {e}. Dimensions might be inaccurate.")

        graph = model.graph
        value_info = {vi.name: vi for vi in graph.value_info}
        value_info.update({vi.name: vi for vi in graph.input})
        value_info.update({vi.name: vi for vi in graph.output})
        initializers = {init.name: init for init in graph.initializer}

        layers = []
        
        for node in graph.node:
            if node.op_type == 'Conv':
                attr = {a.name: a for a in node.attribute}
                # kernel_shape 可能缺失，优先从权重推断
                k_h = k_w = None
                if 'kernel_shape' in attr and len(attr['kernel_shape'].ints) >= 2:
                    kernel_shape = attr['kernel_shape'].ints
                    k_h, k_w = kernel_shape[0], kernel_shape[1]

                strides = attr['strides'].ints if 'strides' in attr else [1, 1]
                s = strides[0] if len(strides) > 0 else 1
                pads = attr['pads'].ints if 'pads' in attr else [0, 0, 0, 0]
                p = pads[0] if len(pads) > 0 else 0 
                
                input_name = node.input[0]
                if input_name in value_info:
                    input_shape = value_info[input_name].type.tensor_type.shape.dim
                    if len(input_shape) >= 4:
                        ic = input_shape[1].dim_value
                        h = input_shape[2].dim_value
                        w = input_shape[3].dim_value
                    else:
                        continue
                else:
                    continue

                weight_name = node.input[1]
                if weight_name in initializers:
                    weight_tensor = initializers[weight_name]
                    oc = weight_tensor.dims[0]
                    if k_h is None or k_w is None:
                        # weight dims: [OC, IC, K_h, K_w]
                        if len(weight_tensor.dims) >= 4:
                            k_h, k_w = weight_tensor.dims[2], weight_tensor.dims[3]
                elif weight_name in value_info:
                     weight_shape = value_info[weight_name].type.tensor_type.shape.dim
                     oc = weight_shape[0].dim_value
                     if k_h is None or k_w is None:
                         if len(weight_shape) >= 4:
                             k_h, k_w = weight_shape[2].dim_value, weight_shape[3].dim_value
                else:
                    continue

                if k_h is None or k_w is None:
                    continue

                if h <= 0 or w <= 0:
                    continue

                layer = LayerParams(name=node.name, H=h, W=w, IC=ic, OC=oc, K_h=k_h, K_w=k_w, S=s, P=p)
                layers.append(layer)

        return layers

# ==========================================
# 4. Cost Model 
# ==========================================
class AdvancedCostModel:
    def __init__(self, hw: HardwareConfig):
        self.hw = hw

    def get_raw_input_tile_dim(self, t_oh, t_ow, layer):
        # 计算生成 t_oh * t_ow 输出所需的输入尺寸
        t_ih = (t_oh - 1) * layer.S + layer.K_h
        t_iw = (t_ow - 1) * layer.S + layer.K_w
        return t_ih, t_iw

    def evaluate(self, layer: LayerParams, t_oh, t_ow, t_oc, t_ic):
        # 1. 空间需求计算 (Space Constraints - Manual Logic)
        raw_t_ih, raw_t_iw = self.get_raw_input_tile_dim(t_oh, t_ow, layer)
        
        size_ifm_tile = raw_t_ih * raw_t_iw * t_ic * self.hw.dtype_input
        size_wgt_tile = layer.K_h * layer.K_w * t_ic * t_oc * self.hw.dtype_input
        size_ofm_tile = t_oh * t_ow * t_oc * self.hw.dtype_input 
        
        # 严格共存策略: SPM 必须同时存下 IFM + Weight + OFM
        total_spm_needed = size_ifm_tile + size_wgt_tile + size_ofm_tile
        
        # ACC 存 Partial Sum (Int32)
        size_acc_needed = t_oh * t_ow * t_oc * self.hw.dtype_acc

        # 检查溢出
        if total_spm_needed > self.hw.spm_size_bytes:
            return None # SPM OOM
        if size_acc_needed > self.hw.acc_size_bytes:
            return None # ACC OOM

        # 2. 性能计算 (Latency Calculation - Manual Logic)
        
        # Global Loops Counts
        n_cout = math.ceil(layer.OC / t_oc)
        n_h    = math.ceil(layer.OH / t_oh)
        n_w    = math.ceil(layer.OW / t_ow)
        n_cin  = math.ceil(layer.IC / t_ic)

        # 计算 conv_tile 被调用的总次数
        total_global_tiles = n_cout * n_h * n_w * n_cin
        
        # --- A. DMA Traffic Calculation ---
        
        # 假设 Worst Case: 每次 conv_tile 都搬运数据 (无复用)
        total_dma_ifm_bytes = total_global_tiles * size_ifm_tile
        total_dma_wgt_bytes = total_global_tiles * size_wgt_tile
        
        # Bias
        bytes_bias_per_tile = t_oc * self.hw.dtype_acc
        total_dma_bias_bytes = total_global_tiles * bytes_bias_per_tile

        # OFM Store (Write Back)
        # 搬出次数 = n_cout * n_h * n_w (不乘 n_cin)
        total_dma_ofm_bytes = (n_cout * n_h * n_w) * size_ofm_tile

        # --- B. Compute Latency Calculation ---
        
        # 这里的 32 是 sys_array_size
        num_spatial_micro_ops = math.ceil((t_oh * t_ow) / 32.0)
        num_cout_micro_ops = math.ceil(t_oc / 32.0)
        num_cin_micro_ops = math.ceil(t_ic / 32.0)
        
        ops_per_conv_tile = num_cout_micro_ops * num_spatial_micro_ops * num_cin_micro_ops
        total_atomic_ops = total_global_tiles * ops_per_conv_tile
        
        # --- C. Total Latency Summation ---
        
        bw_byte_per_us = self.hw.measured_bandwidth_mbps

        total_traffic_bytes = total_dma_ifm_bytes + total_dma_wgt_bytes + total_dma_bias_bytes + total_dma_ofm_bytes

        # 指令数计算
        num_mvin_ifm = total_global_tiles
        num_mvin_wgt = total_global_tiles
        num_mvin_bias = total_global_tiles
        num_mvout_ofm = n_cout * n_h * n_w
        num_dma_instructions = num_mvin_ifm + num_mvin_wgt + num_mvin_bias + num_mvout_ofm

        lat_dma_us = (num_dma_instructions * self.hw.instr_overhead_us) + (total_traffic_bytes / bw_byte_per_us)
        lat_compute_us = total_atomic_ops * self.hw.measured_conv_us

        total_latency_ms = (lat_dma_us + lat_compute_us) / 1000.0

        return {
            'est_latency_ms': total_latency_ms,
            'spm_util': total_spm_needed / self.hw.spm_size_bytes,
            'acc_util': size_acc_needed / self.hw.acc_size_bytes,
            't_oh': t_oh, 't_ow': t_ow, 't_ic': t_ic, 't_oc': t_oc,
            'config_str': f"{t_oh}x{t_ow}_{t_ic}x{t_oc}" # 保留给 debug 用
        }

# ==========================================
# 5. DSE 探索逻辑 (适配新 Cost Model)
# ==========================================
class DesignSpaceExplorer:
    def __init__(self, layers: List[LayerParams], total_mem_bytes: int):
        self.layers = layers
        self.total_mem_bytes = total_mem_bytes
        self.base_hw = HardwareConfig(0, 0) # 用于获取默认步长等

    def search_layer(self, layer: LayerParams, model: AdvancedCostModel):
        min_lat = float('inf')
        best_res = None
        
        # 采用 Manual VGG 的搜索步长策略 (更细致)
        def safe_range(limit, step, max_val=224):
            end = min(limit, max_val)
            if end < step: return [end]
            return range(step, end + 1, step)

        # 搜索空间定义
        # H/W: 步长为 4 (Manual VGG 逻辑)
        r_oh = safe_range(layer.OH, step=4, max_val=64) 
        r_ow = safe_range(layer.OW, step=4, max_val=64)
        
        # IC/OC: 步长为 32
        start_ic = 32 if layer.IC >= 32 else layer.IC
        start_oc = 32 if layer.OC >= 32 else layer.OC
        
        r_ic = range(start_ic, min(layer.IC, 256) + 1, 32)
        r_oc = range(start_oc, min(layer.OC, 256) + 1, 32)

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
        print(f"Starting DSE [Algorithm: Manual VGG Logic] (Total Mem: {self.total_mem_bytes/1024:.0f} KB)...")
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
                
                # [关键] 保持原 ONNX 版 JSON 的字段结构
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
                print(f"  Ratio {ratio:.1f} -> Failed (OOM)")

        return best_global

# ==========================================
# 6. JSON 导出工具 (保持不变)
# ==========================================
def export_to_json(data, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\n✅ Results exported to {filename}")
    except Exception as e:
        print(f"❌ Failed to export JSON: {e}")

# ==========================================
# 7. 主执行逻辑 (保持不变)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSE Tool for ONNX Models (Manual VGG Algo)")
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help='Path to the input ONNX model file')
    parser.add_argument('-o', '--output', type=str, required=False, 
                        help='Path to the output JSON file')
    parser.add_argument('--mem', type=int, default=512, 
                        help='Total memory size in KB (default: 512)')
    parser.add_argument('--shape', type=str, default=None,
                        help='Override input shape, format: N,C,H,W (e.g. 1,3,224,224)')
    
    args = parser.parse_args()

    input_path = args.input
    total_mem_bytes = args.mem * 1024

    shape_override = None
    if args.shape:
        try:
            shape_override = [int(x) for x in args.shape.split(',')]
        except ValueError:
            print("❌ Error: Invalid format for --shape.")
            exit(1)

    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}.json"

    # 1. 解析 ONNX
    try:
        layers = OnnxModelParser.parse(input_path, input_shape_override=shape_override)
    except Exception as e:
        print(f"❌ Failed to parse ONNX model: {e}")
        exit(1)
    
    if not layers:
        print("❌ No valid convolutional layers found.")
        exit(1)

    print(f"\nModel: {input_path} ({len(layers)} Conv Layers detected)")
    print(f"Algorithm: Manual VGG Style (Strict Memory + Atomic Ops)")
    print(f"Output: {output_path}")
    
    # 2. 运行 DSE
    dse = DesignSpaceExplorer(layers, total_mem_bytes)
    best = dse.run()

    # 3. 输出
    if best:
        print("\n" + "="*60)
        print(f"✅ Best Configuration found")
        print(f"  Total Latency: {best['total_latency_ms']:.2f} ms")
        print("="*60)
        export_to_json(best, output_path)
    else:
        print("\n❌ All configurations failed.")