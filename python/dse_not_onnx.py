import math
import json
import argparse
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# ==========================================
# 1. 硬件定义
# ==========================================
@dataclass
class HardwareConfig:
    spm_size_bytes: int
    acc_size_bytes: int
    
    dtype_input: int = 1     # int8
    dtype_acc: int = 4       # int32
    
    # 物理阵列参数 (System Array / PEs)
    # 依据描述：一次原子计算处理 32 OC, 32 IC, 32 Spatial Pixels
    sys_array_size: int = 32 

    measured_bandwidth_mbps: float = 1300.0 
    
    # 用户指定：每做一次最小的卷积运算(原子操作)的延时
    # 原子操作定义：计算 32(OC) x 32(IC) x 32(Spatial Output) 的卷积
    measured_conv_us: float = 5.0 
    # 每条指令的开销（包含 mvin/mvout/conv 等）
    instr_overhead_us: float = 1.54

    # 新增：ACC 量化写回 SPM 的指令延时（暂时建模为 0）
    acc_to_spm_quantize_us: float = 0.0

# ==========================================
# 2. 层参数定义
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
# 3. 模型构建器 (替代 ONNX Parser)
# ==========================================
class ModelBuilder:
    @staticmethod
    def build_vgg16_layers(input_h=224, input_w=224) -> List[LayerParams]:
        """
        手动构建 VGG16 的卷积层结构
        结构参考：Standard VGG16 (Conv3-64 -> ... -> MaxPool -> ...)
        这里简化 MaxPool 带来的降采样，手动调整 H/W
        """
        layers = []
        
        # 定义一个简单的辅助函数来添加层并更新尺寸
        current_h, current_w = input_h, input_w
        
        def add_conv(name, ic, oc, k=3, s=1, p=1):
            nonlocal current_h, current_w
            # VGG 使用 padding=1 保持尺寸 (对于 k=3)
            layer = LayerParams(name, current_h, current_w, ic, oc, k, k, s, p)
            layers.append(layer)
            # 更新输出尺寸
            current_h = layer.OH
            current_w = layer.OW

        def add_pool():
            nonlocal current_h, current_w
            current_h //= 2
            current_w //= 2

        # Block 1
        add_conv("conv1_1", 3, 64)
        add_conv("conv1_2", 64, 64)
        add_pool() # 112
        
        # Block 2
        add_conv("conv2_1", 64, 128)
        add_conv("conv2_2", 128, 128)
        add_pool() # 56
        
        # Block 3
        add_conv("conv3_1", 128, 256)
        add_conv("conv3_2", 256, 256)
        add_conv("conv3_3", 256, 256)
        add_pool() # 28
        
        # Block 4
        add_conv("conv4_1", 256, 512)
        add_conv("conv4_2", 512, 512)
        add_conv("conv4_3", 512, 512)
        add_pool() # 14
        
        # Block 5
        add_conv("conv5_1", 512, 512)
        add_conv("conv5_2", 512, 512)
        add_conv("conv5_3", 512, 512)
        add_pool() # 7

        return layers

# ==========================================
# 4. Cost Model (核心修改)
# ==========================================
class AdvancedCostModel:
    def __init__(self, hw: HardwareConfig, enable_acc_to_spm: bool = False):
        self.hw = hw
        self.enable_acc_to_spm = enable_acc_to_spm

    def get_raw_input_tile_dim(self, t_oh, t_ow, layer):
        # 计算生成 t_oh * t_ow 输出所需的输入尺寸
        t_ih = (t_oh - 1) * layer.S + layer.K_h
        t_iw = (t_ow - 1) * layer.S + layer.K_w
        return t_ih, t_iw

    def evaluate(self, layer: LayerParams, t_oh, t_ow, t_oc, t_ic):
        """
        根据用户描述的 specific dataflow 进行评估
        """
        # 0. 规整化维度
        # 虽然分块参数是 t_ic，但硬件底层一次处理 32 channel
        # 实际分配空间和传输时，通常按 32 对齐考虑，或者按实际大小考虑但计算按 32 步进
        # 这里假设空间按实际大小分配，计算延时按 ceil(x/32) 计算
        
        # 1. 空间需求计算 (Space Constraints)
        raw_t_ih, raw_t_iw = self.get_raw_input_tile_dim(t_oh, t_ow, layer)
        
        # SPM 必须同时存下：IFM Tile + Weight Tile (+ OFM Tile if 需要回写 SPM)
        # 若启用 ACC->SPM 量化写回指令，则无需在 SPM 预留 OFM 空间
        size_ifm_tile = raw_t_ih * raw_t_iw * t_ic * self.hw.dtype_input
        size_wgt_tile = layer.K_h * layer.K_w * t_ic * t_oc * self.hw.dtype_input
        size_ofm_tile = t_oh * t_ow * t_oc * self.hw.dtype_input # Int8 result
        
        if self.enable_acc_to_spm:
            total_spm_needed = size_ifm_tile + size_wgt_tile
        else:
            total_spm_needed = size_ifm_tile + size_wgt_tile + size_ofm_tile
        
        # ACC 存 Partial Sum (Int32)
        # ACC 需要存下当前 block 的所有输出通道的 psum
        size_acc_needed = t_oh * t_ow * t_oc * self.hw.dtype_acc

        # 检查溢出
        if total_spm_needed > self.hw.spm_size_bytes:
            return None # SPM OOM
        if size_acc_needed > self.hw.acc_size_bytes:
            return None # ACC OOM

        # 2. 性能计算 (Latency Calculation)
        # 按照用户描述的伪代码层级结构
        
        # Global Loops Counts
        n_cout = math.ceil(layer.OC / t_oc)
        n_h    = math.ceil(layer.OH / t_oh)
        n_w    = math.ceil(layer.OW / t_ow)
        n_cin  = math.ceil(layer.IC / t_ic)

        # 统计总量
        total_conv_tile_calls = 0
        total_dma_ifm_bytes = 0
        total_dma_wgt_bytes = 0
        total_dma_bias_bytes = 0
        
        # 模拟最外层循环: Global Tiling Loop
        # Order: Cout -> H -> W -> Cin
        # void conv() {
        #   for (i_Cout...) {
        #     for (i_H...) {
        #       for (i_W...) {
        #         for (i_Cin...) {
        #           conv_tile(); 
        #         }
        #         // move out result (Quantized)
        #       }
        #     }
        #   }
        # }

        # 计算 conv_tile 被调用的总次数
        total_global_tiles = n_cout * n_h * n_w * n_cin
        
        # --- A. DMA Traffic Calculation ---
        
        # 1. IFM Loading
        # 在 conv_tile 内部调用 mvin_tile_ifm_to_sram()
        # 每次 conv_tile 都会搬运一份 (t_h, t_w, t_ic) 的 IFM
        # 注意：虽然在 H/W 循环中 IFM 可能有重叠，或者 Cin 循环中 IFM 是新的
        # 根据用户描述："conv_tile 中 ... mvin_tile_ifm_to_sram()" 且无显式复用缓存说明
        # 我们假设每次都发生 DMA (这是 Worst Case，也是目前代码逻辑的体现)
        # 除非 H 很大减少了 mvin 次数 (即 n_h 变小)，这里已经通过 n_h 体现了
        total_dma_ifm_bytes = total_global_tiles * size_ifm_tile

        # 2. Weight Loading
        # 在 conv_tile 内部调用 mvin_tile_weight_to_sram()
        # 每次搬运 (t_oc, t_ic, k, k)
        total_dma_wgt_bytes = total_global_tiles * size_wgt_tile
        
        # 3. Bias Loading
        # 用户伪代码：for(j_Cout=0...t_Cout) { mvin_bias() ... }
        # 这发生在 conv_tile 内部。
        # 每个 conv_tile 内部会加载 t_oc 个 bias (int32)
        # 虽然通常 Bias 很小，但为了严谨加上
        bytes_bias_per_tile = t_oc * self.hw.dtype_acc
        total_dma_bias_bytes = total_global_tiles * bytes_bias_per_tile

        # 4. OFM Store (Write Back)
        # 只有在 Cin 循环结束 (i_Cin 遍历完) 后，ACC 结果才是完整的
        # 若启用 ACC->SPM 量化写回指令：直接从 ACC 量化写回 SPM（可建模为额外指令延时）
        # 否则：先写回 SPM 再搬出（保持原有逻辑）
        # 搬出次数 = n_cout * n_h * n_w (不乘 n_cin)
        total_dma_ofm_bytes = (n_cout * n_h * n_w) * size_ofm_tile

        # --- B. Compute Latency Calculation ---
        
        # 分析 conv_tile 内部的微循环
        # for j_Cout in 0..t_Cout step 32:
        #   for j_H, j_W in spatial step t_Hm, t_Wm: (Total spatial pixels = t_oh * t_ow)
        #     for j_Cin in 0..t_Cin step 32:
        #       compute_tile()
        
        # 一个 compute_tile 对应的原子操作数：
        # 它处理 32 个 Cout, 32 个 Cin, 和 32 个 Spatial Pixels (SA限制)
        
        # 1. 计算每个 conv_tile 需要多少次原子 compute_tile
        # Spatial chunks: 向上取整 (Total Pixels / 32)
        num_spatial_micro_ops = math.ceil((t_oh * t_ow) / 32.0)
        
        # Channel chunks
        num_cout_micro_ops = math.ceil(t_oc / 32.0)
        num_cin_micro_ops = math.ceil(t_ic / 32.0)
        
        ops_per_conv_tile = num_cout_micro_ops * num_spatial_micro_ops * num_cin_micro_ops
        
        # 2. 总原子操作数
        total_atomic_ops = total_global_tiles * ops_per_conv_tile
        
        # --- C. Total Latency Summation ---
        
        # 假设：目前无法并行 (Serial: DMA -> Compute -> DMA)
        # 带宽单位 MBps -> B/us (1 MBps ~= 1 B/us)
        bw_byte_per_us = self.hw.measured_bandwidth_mbps

        total_traffic_bytes = total_dma_ifm_bytes + total_dma_wgt_bytes + total_dma_bias_bytes + total_dma_ofm_bytes

        # DMA 相关延迟 = 搬运指令数 * 指令开销 + 数据量/带宽
        num_mvin_ifm = total_global_tiles
        num_mvin_wgt = total_global_tiles
        num_mvin_bias = total_global_tiles
        num_mvout_ofm = n_cout * n_h * n_w
        num_dma_instructions = num_mvin_ifm + num_mvin_wgt + num_mvin_bias + num_mvout_ofm
        if self.enable_acc_to_spm:
            # 量化写回发生在每个 (Cout,H,W) 完成 Cin 累加后一次
            num_dma_instructions += (n_cout * n_h * n_w)

        lat_dma_us = (num_dma_instructions * self.hw.instr_overhead_us) + (total_traffic_bytes / bw_byte_per_us)

        # Conv 测试为端到端（包含指令开销+计算开销）
        lat_compute_us = total_atomic_ops * self.hw.measured_conv_us

        # 额外的量化写回指令时延（若有）
        lat_instr_us = 0.0
        if self.enable_acc_to_spm:
            lat_instr_us += (n_cout * n_h * n_w) * self.hw.acc_to_spm_quantize_us
        
        total_latency_ms = (lat_dma_us + lat_instr_us + lat_compute_us) / 1000.0

        return {
            'est_latency_ms': total_latency_ms,
            'spm_util': total_spm_needed / self.hw.spm_size_bytes,
            'acc_util': size_acc_needed / self.hw.acc_size_bytes,
            't_oh': t_oh, 't_ow': t_ow, 't_ic': t_ic, 't_oc': t_oc,
            'breakdown': {
                'compute_ms': lat_compute_us / 1000.0,
                'transfer_ms': lat_dma_us / 1000.0,
                'instruction_ms': lat_instr_us / 1000.0
            }
        }

# ==========================================
# 5. DSE 探索逻辑
# ==========================================
class DesignSpaceExplorer:
    def __init__(self, layers: List[LayerParams], total_mem_bytes: int):
        self.layers = layers
        self.total_mem_bytes = total_mem_bytes
        self.base_hw = HardwareConfig(0, 0) # Temp

    def search_layer(self, layer: LayerParams, model: AdvancedCostModel):
        min_lat = float('inf')
        best_res = None
        
        # 搜索步长策略
        # Spatial: 尝试 4 的倍数，最大到 112 (VGG最大可能)
        # Channel: 尝试 32 的倍数 (硬件亲和性)
        
        def safe_range(limit, step, max_val=224):
            end = min(limit, max_val)
            # 确保至少执行一次 loop，即使用户给的 limit 很小
            if end < step: return [end]
            return range(step, end + 1, step)

        # 搜索空间定义
        # H/W: 假设输出分块至少 4x4, 最大不超过 layer 尺寸
        r_oh = safe_range(layer.OH, step=4, max_val=64) 
        r_ow = safe_range(layer.OW, step=4, max_val=64)
        
        # IC/OC: 必须是 32 的倍数 (或者实际小于32则取实际值)
        # 如果 layer.IC < 32, 则 t_ic = layer.IC
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

    def run(self, enable_acc_to_spm: bool = False):
        mode_name = "ACC->SPM Quantize" if enable_acc_to_spm else "Baseline"
        print(f"Starting DSE [{mode_name}] (Total Mem: {self.total_mem_bytes/1024:.0f} KB)...")
        best_global = None
        min_global_lat = float('inf')

        # 扫描 SPM/ACC 的划分比例
        for ratio in [i/10.0 for i in range(1, 10)]: # 0.1 to 0.9
            spm = int(self.total_mem_bytes * ratio)
            acc = self.total_mem_bytes - spm
            
            hw = HardwareConfig(spm_size_bytes=spm, acc_size_bytes=acc)
            model = AdvancedCostModel(hw, enable_acc_to_spm=enable_acc_to_spm)
            
            total_lat = 0
            layer_data = []
            is_valid_config = True
            
            print(f"Checking Ratio {ratio} (SPM:{spm//1024}KB, ACC:{acc//1024}KB)...", end="\r")
            
            for layer in self.layers:
                res = self.search_layer(layer, model)
                if not res:
                    is_valid_config = False
                    break # 该比例无法满足某一层的最小分块需求
                
                total_lat += res['est_latency_ms']
                layer_data.append({
                    'layer': layer.name,
                    'shape_in': f"{layer.IC}x{layer.H}x{layer.W}",
                    'shape_out': f"{layer.OC}x{layer.OH}x{layer.OW}",
                    'tile': f"{res['t_oc']}x{res['t_oh']}x{res['t_ow']}x{res['t_ic']}",
                    'lat_ms': round(res['est_latency_ms'], 2),
                    'compute_ms': round(res['breakdown']['compute_ms'], 2),
                    'transfer_ms': round(res['breakdown']['transfer_ms'], 2),
                    'instruction_ms': round(res['breakdown']['instruction_ms'], 2),
                    'spm_util': round(res['spm_util'], 2),
                    'acc_util': round(res['acc_util'], 2)
                })
            
            if is_valid_config:
                # 逐个硬件配置打印最佳分块和延迟
                print(f"\n[Ratio {ratio:.1f}] SPM {spm//1024}KB / ACC {acc//1024}KB -> Total {total_lat:.2f} ms")
                print(f"{'Layer':<12} | {'Input':<14} | {'Output':<14} | {'Tile (OcxHxWxIc)':<20} | {'Lat':<7} | {'Instr':<7} | {'Xfer':<7} | {'Comp':<7} | {'SPM%':<5} | {'ACC%':<5}")
                print("-" * 130)
                for l in layer_data:
                    print(f"{l['layer']:<12} | {l['shape_in']:<14} | {l['shape_out']:<14} | {l['tile']:<20} | {l['lat_ms']:<7} | {l['instruction_ms']:<7} | {l['transfer_ms']:<7} | {l['compute_ms']:<7} | {l['spm_util']:<5} | {l['acc_util']:<5}")

                if total_lat < min_global_lat:
                    min_global_lat = total_lat
                    best_global = {
                        'mode': mode_name,
                        'best_ratio': ratio,
                        'hw_config': {'spm_bytes': spm, 'acc_bytes': acc},
                        'total_latency_ms': total_lat,
                        'layers': layer_data
                    }
        print("\n" + "-"*50)
        return best_global

# ==========================================
# 6. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSE Tool for VGG Model (Manual Build)")
    parser.add_argument('-o', '--output', type=str, default="vgg16_dse_result.json",
                        help='Path to the output JSON file')
    parser.add_argument('--mem', type=int, default=512, 
                        help='Total memory size in KB (default: 512)')
    
    args = parser.parse_args()
    total_mem_bytes = args.mem * 1024

    # 1. 构建模型
    print(f"[Model] Building VGG16-like model...")
    layers = ModelBuilder.build_vgg16_layers()
    print(f"[Model] Generated {len(layers)} layers.")
    
    # 2. 运行 DSE
    dse = DesignSpaceExplorer(layers, total_mem_bytes)
    best = dse.run()

    # 2b. 运行 DSE（启用 ACC->SPM 量化写回）
    best_acc_to_spm = dse.run(enable_acc_to_spm=True)

    # 3. 输出结果
    if best:
        print("\n" + "="*60)
        print(f"✅ Best Configuration Found [Baseline]")
        print(f"   Memory Split : SPM {best['hw_config']['spm_bytes']//1024} KB / ACC {best['hw_config']['acc_bytes']//1024} KB")
        print(f"   Total Latency: {best['total_latency_ms']:.2f} ms")
        print("="*60)
        
        # 打印简要层级报告
        print(f"{'Layer':<12} | {'Input':<14} | {'Output':<14} | {'Tile (OcxHxWxIc)':<20} | {'Lat':<7} | {'Instr':<7} | {'Xfer':<7} | {'Comp':<7} | {'SPM%':<5} | {'ACC%':<5}")
        print("-" * 130)
        for l in best['layers']:
            print(f"{l['layer']:<12} | {l['shape_in']:<14} | {l['shape_out']:<14} | {l['tile']:<20} | {l['lat_ms']:<7} | {l['instruction_ms']:<7} | {l['transfer_ms']:<7} | {l['compute_ms']:<7} | {l['spm_util']:<5} | {l['acc_util']:<5}")

        results = {'baseline': best}
    else:
        results = {'baseline': None}

    if best_acc_to_spm:
        print("\n" + "="*60)
        print(f"✅ Best Configuration Found [ACC->SPM Quantize]")
        print(f"   Memory Split : SPM {best_acc_to_spm['hw_config']['spm_bytes']//1024} KB / ACC {best_acc_to_spm['hw_config']['acc_bytes']//1024} KB")
        print(f"   Total Latency: {best_acc_to_spm['total_latency_ms']:.2f} ms")
        print("="*60)

        print(f"{'Layer':<12} | {'Input':<14} | {'Output':<14} | {'Tile (OcxHxWxIc)':<20} | {'Lat':<7} | {'Instr':<7} | {'Xfer':<7} | {'Comp':<7} | {'SPM%':<5} | {'ACC%':<5}")
        print("-" * 130)
        for l in best_acc_to_spm['layers']:
            print(f"{l['layer']:<12} | {l['shape_in']:<14} | {l['shape_out']:<14} | {l['tile']:<20} | {l['lat_ms']:<7} | {l['instruction_ms']:<7} | {l['transfer_ms']:<7} | {l['compute_ms']:<7} | {l['spm_util']:<5} | {l['acc_util']:<5}")

        results['acc_to_spm'] = best_acc_to_spm
    else:
        results['acc_to_spm'] = None

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {args.output}")
    if not best and not best_acc_to_spm:
        print("\n❌ All configurations failed (OOM). Increase --mem.")