import math
import json
import argparse
import os
import onnx
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any

# ==========================================
# 1. 硬件与分块配置定义
# ==========================================
@dataclass
class HardwareConfig:
    """硬件配置参数"""
    spm_size_bytes: int
    acc_size_bytes: int
    
    dtype_input: int = 1     # int8
    dtype_acc: int = 4       # int32
    
    # 物理阵列参数
    sys_array_size: int = 32 
    measured_bandwidth_mbps: float = 1300.0 
    
    # 原子操作参数 (计算 32x32x32 卷积)
    measured_conv_us: float = 5.0 
    instr_overhead_us: float = 1.54

@dataclass
class TileResult:
    """单层分块评估结果"""
    t_oh: int
    t_ow: int
    t_ic: int
    t_oc: int
    latency_ms: float
    spm_util: float
    acc_util: float


@dataclass
class LayerParams:
    """卷积层参数"""
    name: str
    H: int
    W: int
    IC: int
    OC: int
    K_h: int
    K_w: int
    S: int
    P: int

    @property
    def OH(self) -> int: 
        return (self.H + 2 * self.P - self.K_h) // self.S + 1
        
    @property
    def OW(self) -> int: 
        return (self.W + 2 * self.P - self.K_w) // self.S + 1

# ==========================================
# 2. ONNX 解析器
# ==========================================
class OnnxModelParser:
    """用于解析 ONNX 模型并提取卷积层信息的工具类"""
    
    @staticmethod
    def parse(onnx_path: str, input_shape_override: Optional[List[int]] = None) -> List[LayerParams]:
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"未找到 ONNX 文件: {onnx_path}")

        print(f"[Parser] 正在加载模型: {onnx_path}...")
        model = onnx.load(onnx_path)
        graph = model.graph
        
        # 处理输入维度
        input_tensor = graph.input[0]
        orig_dims = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
        
        if input_shape_override:
            print(f"[Parser] 覆盖输入维度为: {input_shape_override}")
            if input_tensor.type.tensor_type.shape.dim:
                if len(input_shape_override) != len(orig_dims):
                    print(f"[Warning] 覆盖维度长度 {len(input_shape_override)} 与模型输入秩 {len(orig_dims)} 不一致")
                for i, dim_val in enumerate(input_shape_override):
                    if i < len(input_tensor.type.tensor_type.shape.dim):
                        input_tensor.type.tensor_type.shape.dim[i].dim_value = dim_val
        else:
            if any(d <= 0 for d in orig_dims):
                raise ValueError(f"模型存在动态输入维度 {orig_dims}，请使用 --shape 参数指定静态维度。")
            print(f"[Parser] 使用模型静态维度: {orig_dims}")

        # 形状推断
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"[Warning] 形状推断失败: {e}。后续维度可能不准确。")

        return OnnxModelParser._extract_conv_layers(model.graph)

    @staticmethod
    def _extract_conv_layers(graph) -> List[LayerParams]:
        """从计算图中提取 Conv 节点信息"""
        value_info = {vi.name: vi for vi in graph.value_info}
        value_info.update({vi.name: vi for vi in graph.input})
        value_info.update({vi.name: vi for vi in graph.output})
        initializers = {init.name: init for init in graph.initializer}

        layers = []
        for node in graph.node:
            if node.op_type != 'Conv':
                continue

            attr = {a.name: a for a in node.attribute}
            k_h, k_w = None, None
            if 'kernel_shape' in attr and len(attr['kernel_shape'].ints) >= 2:
                kernel_shape = attr['kernel_shape'].ints
                k_h, k_w = kernel_shape[0], kernel_shape[1]

            strides = attr.get('strides').ints if 'strides' in attr else [1, 1]
            s = strides[0] if strides else 1
            pads = attr.get('pads').ints if 'pads' in attr else [0, 0, 0, 0]
            p = pads[0] if pads else 0 
            
            # 解析输入特征图 (IC, H, W)
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

            # 解析权重 (OC, K_h, K_w)
            weight_name = node.input[1]
            if weight_name in initializers:
                weight_tensor = initializers[weight_name]
                oc = weight_tensor.dims[0]
                if k_h is None or k_w is None and len(weight_tensor.dims) >= 4:
                    k_h, k_w = weight_tensor.dims[2], weight_tensor.dims[3]
            elif weight_name in value_info:
                 weight_shape = value_info[weight_name].type.tensor_type.shape.dim
                 oc = weight_shape[0].dim_value
                 if k_h is None or k_w is None and len(weight_shape) >= 4:
                     k_h, k_w = weight_shape[2].dim_value, weight_shape[3].dim_value
            else:
                continue

            if k_h is None or k_w is None or h <= 0 or w <= 0:
                continue

            layers.append(LayerParams(name=node.name, H=h, W=w, IC=ic, OC=oc, K_h=k_h, K_w=k_w, S=s, P=p))

        return layers

# ==========================================
# 3. 成本模型 (Cost Model)
# ==========================================
class CostModel:
    """评估特定分块策略在给定硬件下的延迟与内存利用率"""
    
    def __init__(self, hw: HardwareConfig):
        self.hw = hw

    def _get_raw_input_tile_dim(self, t_oh: int, t_ow: int, layer: LayerParams) -> tuple:
        """计算生成 t_oh * t_ow 输出所需的输入尺寸"""
        t_ih = (t_oh - 1) * layer.S + layer.K_h
        t_iw = (t_ow - 1) * layer.S + layer.K_w
        return t_ih, t_iw

    def evaluate(self, layer: LayerParams, t_oh: int, t_ow: int, t_oc: int, t_ic: int) -> Optional[TileResult]:
        """评估分块配置，若超出内存限制则返回 None"""
        # 1. 空间需求计算
        raw_t_ih, raw_t_iw = self._get_raw_input_tile_dim(t_oh, t_ow, layer)
        
        size_ifm_tile = raw_t_ih * raw_t_iw * t_ic * self.hw.dtype_input
        size_wgt_tile = layer.K_h * layer.K_w * t_ic * t_oc * self.hw.dtype_input
        size_ofm_tile = t_oh * t_ow * t_oc * self.hw.dtype_input 
        
        total_spm_needed = size_ifm_tile + size_wgt_tile + size_ofm_tile
        size_acc_needed = t_oh * t_ow * t_oc * self.hw.dtype_acc

        # 内存溢出检查
        if total_spm_needed > self.hw.spm_size_bytes or size_acc_needed > self.hw.acc_size_bytes:
            return None

        # 2. 性能计算
        n_cout = math.ceil(layer.OC / t_oc)
        n_h    = math.ceil(layer.OH / t_oh)
        n_w    = math.ceil(layer.OW / t_ow)
        n_cin  = math.ceil(layer.IC / t_ic)

        total_global_tiles = n_cout * n_h * n_w * n_cin
        
        # DMA 数据量计算
        total_dma_ifm_bytes = total_global_tiles * size_ifm_tile
        total_dma_wgt_bytes = total_global_tiles * size_wgt_tile
        total_dma_bias_bytes = total_global_tiles * (t_oc * self.hw.dtype_acc)
        total_dma_ofm_bytes = (n_cout * n_h * n_w) * size_ofm_tile

        # 计算延迟
        num_spatial_micro_ops = math.ceil((t_oh * t_ow) / self.hw.sys_array_size)
        num_cout_micro_ops = math.ceil(t_oc / self.hw.sys_array_size)
        num_cin_micro_ops = math.ceil(t_ic / self.hw.sys_array_size)
        
        ops_per_conv_tile = num_cout_micro_ops * num_spatial_micro_ops * num_cin_micro_ops
        total_atomic_ops = total_global_tiles * ops_per_conv_tile
        
        total_traffic_bytes = total_dma_ifm_bytes + total_dma_wgt_bytes + total_dma_bias_bytes + total_dma_ofm_bytes
        num_dma_instructions = (total_global_tiles * 3) + (n_cout * n_h * n_w)

        lat_dma_us = (num_dma_instructions * self.hw.instr_overhead_us) + (total_traffic_bytes / self.hw.measured_bandwidth_mbps)
        lat_compute_us = total_atomic_ops * self.hw.measured_conv_us
        total_latency_ms = (lat_dma_us + lat_compute_us) / 1000.0

        return TileResult(
            t_oh=t_oh, t_ow=t_ow, t_ic=t_ic, t_oc=t_oc,
            latency_ms=total_latency_ms,
            spm_util=total_spm_needed / self.hw.spm_size_bytes,
            acc_util=size_acc_needed / self.hw.acc_size_bytes
        )

# ==========================================
# 4. 优化器 (Fixed Hardware Tiling Optimizer)
# ==========================================
class TileOptimizer:
    """在给定的硬件配置下，搜索最优的卷积分块策略"""
    
    def __init__(self, hw: HardwareConfig, layers: List[LayerParams]):
        self.hw = hw
        self.layers = layers
        self.cost_model = CostModel(hw)

    def _get_safe_range(self, limit: int, step: int, max_val: int = 64) -> List[int]:
        """生成安全的搜索空间范围"""
        end = min(limit, max_val)
        if end < step: 
            return [end]
        return list(range(step, end + 1, step))

    def search_best_tile(self, layer: LayerParams) -> Optional[TileResult]:
        """为单层搜索最佳分块"""
        min_lat = float('inf')
        best_res = None
        
        r_oh = self._get_safe_range(layer.OH, step=4, max_val=64) 
        r_ow = self._get_safe_range(layer.OW, step=4, max_val=64)
        
        start_ic = 32 if layer.IC >= 32 else layer.IC
        start_oc = 32 if layer.OC >= 32 else layer.OC
        r_ic = range(start_ic, min(layer.IC, 256) + 1, 32)
        r_oc = range(start_oc, min(layer.OC, 256) + 1, 32)

        for t_ic in r_ic:
            for t_oc in r_oc:
                for t_oh in r_oh:
                    for t_ow in r_ow:
                        res = self.cost_model.evaluate(layer, t_oh, t_ow, t_oc, t_ic)
                        if res and res.latency_ms < min_lat:
                            min_lat = res.latency_ms
                            best_res = res
                            
        return best_res

    def run_optimization(self) -> Optional[Dict[str, Any]]:
        """运行全网络层优化并汇总结果"""
        print(f"正在进行分块优化 (SPM: {self.hw.spm_size_bytes/1024:.0f} KB, ACC: {self.hw.acc_size_bytes/1024:.0f} KB)...")
        
        total_lat = 0.0
        layer_data = []
        
        for layer in self.layers:
            res = self.search_best_tile(layer)
            if not res:
                print(f"❌ 错误: 层 {layer.name} 在当前硬件限制下无法找到合适的分块(OOM)。")
                return None
            
            total_lat += res.latency_ms
            layer_data.append({
                'layer_name': layer.name,
                'input_shape': f"{layer.IC}x{layer.H}x{layer.W}",
                'output_shape': f"{layer.OC}x{layer.OH}x{layer.OW}",
                'tile_params': {
                    't_oh': res.t_oh, 't_ow': res.t_ow, 
                    't_ic': res.t_ic, 't_oc': res.t_oc
                },
                'latency_ms': res.latency_ms,
                'spm_util': res.spm_util,
                'acc_util': res.acc_util
            })
        
        return {
            'hw_config': {'spm_bytes': self.hw.spm_size_bytes, 'acc_bytes': self.hw.acc_size_bytes},
            'total_latency_ms': total_lat,
            'layers': layer_data
        }

# ==========================================
# 5. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiling Optimizer for ONNX Models (Fixed Hardware)")
    parser.add_argument('-i', '--input', type=str, required=True, help='输入 ONNX 模型文件路径')
    parser.add_argument('-o', '--output', type=str, required=False, help='输出 JSON 文件路径')
    parser.add_argument('--spm', type=int, required=True, help='SPM 容量大小 (单位: KB)')
    parser.add_argument('--acc', type=int, required=True, help='ACC 容量大小 (单位: KB)')
    parser.add_argument('--shape', type=str, default=None, help='覆盖输入形状, 格式: N,C,H,W (例如 1,3,224,224)')
    
    args = parser.parse_args()

    # 处理输入参数
    spm_bytes = args.spm * 1024
    acc_bytes = args.acc * 1024
    output_path = args.output or f"{os.path.splitext(os.path.basename(args.input))[0]}_optimized.json"

    shape_override = None
    if args.shape:
        try:
            shape_override = [int(x) for x in args.shape.split(',')]
        except ValueError:
            print("❌ 错误: --shape 参数格式无效。")
            exit(1)

    # 1. 解析 ONNX
    try:
        layers = OnnxModelParser.parse(args.input, input_shape_override=shape_override)
    except Exception as e:
        print(f"❌ 解析 ONNX 模型失败: {e}")
        exit(1)

    def _write_json(data: Dict[str, Any]) -> bool:
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"✅ 结果已成功导出至 {output_path}")
            return True
        except Exception as e:
            print(f"❌ 导出 JSON 失败: {e}")
            return False
    
    # 检查是否存在卷积操作
    if not layers:
        empty_result = {
            'hw_config': {'spm_bytes': spm_bytes, 'acc_bytes': acc_bytes},
            'total_latency_ms': 0.0,
            'layers': [],
            'meta': {
                'model_path': args.input,
                'message': '未检测到有效的卷积(Conv)操作，输出空分块配置。'
            }
        }
        print("ℹ️ 未在模型中检测到有效的卷积 (Conv) 操作，正在输出空分块配置 JSON。")
        _write_json(empty_result)
        exit(0)

    print(f"\n模型: {args.input} (共检测到 {len(layers)} 个卷积层)")
    print(f"输出路径: {output_path}")
    
    # 2. 运行优化器
    hw_config = HardwareConfig(spm_size_bytes=spm_bytes, acc_size_bytes=acc_bytes)
    optimizer = TileOptimizer(hw_config, layers)
    result = optimizer.run_optimization()

    # 3. 输出结果
    if result:
        print("\n" + "="*60)
        print(f"✅ 成功找到最佳分块配置")
        print(f"  总延迟评估: {result['total_latency_ms']:.2f} ms")
        print("="*60)

        _write_json(result)
    else:
        print("\n❌ 优化失败：硬件资源限制过严，部分层发生 OOM。")