import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ==========================================
# 1. 硬件定义
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

    measured_latency_us: float = 6.737
    measured_bandwidth_mbps: float = 320.41

# ==========================================
# 2. 层参数 (新增 Padding)
# ==========================================
@dataclass
class LayerParams:
    name: str
    H: int; W: int; IC: int; OC: int
    K_h: int; K_w: int; S: int; P: int  # Added Padding

    @property
    def OH(self): 
        # 标准输出尺寸计算
        return (self.H + 2 * self.P - self.K_h) // self.S + 1
    @property
    def OW(self): 
        return (self.W + 2 * self.P - self.K_w) // self.S + 1

# ==========================================
# 3. 模型构建工具 (修正 ResNet18 尺寸)
# ==========================================
class ModelBuilder:
    @staticmethod
    def get_resnet18_layers() -> List[LayerParams]:
        layers = []
        # Conv1: 224->112, 7x7, s=2, padding=3 (Standard)
        # H_out = (224 + 6 - 7)/2 + 1 = 112
        layers.append(LayerParams("Conv1", 224, 224, 3, 64, 7, 7, 2, 3))
        
        # MaxPool 模拟: 112 -> 56
        current_h, current_w = 56, 56
        
        def add_basic_block(stage_idx, count, ic, oc, stride_first=1):
            nonlocal current_h, current_w
            for b in range(count):
                s = stride_first if b == 0 else 1
                # 3x3 Conv always uses Padding=1 in ResNet to maintain size (if s=1)
                name_c1 = f"S{stage_idx}_B{b}_C1"
                layers.append(LayerParams(name_c1, current_h, current_w, ic, oc, 3, 3, s, 1))
                
                # Update dims
                h_out = (current_h + 2*1 - 3) // s + 1
                w_out = (current_w + 2*1 - 3) // s + 1
                
                name_c2 = f"S{stage_idx}_B{b}_C2"
                layers.append(LayerParams(name_c2, h_out, w_out, oc, oc, 3, 3, 1, 1))
                
                current_h, current_w = h_out, w_out
                ic = oc

        # ResNet18 Structure
        add_basic_block(1, 2, 64, 64, stride_first=1)   # 56x56
        add_basic_block(2, 2, 64, 128, stride_first=2)  # 28x28
        add_basic_block(3, 2, 128, 256, stride_first=2) # 14x14
        add_basic_block(4, 2, 256, 512, stride_first=2) # 7x7
        
        return layers

# ==========================================
# 4. Cost Model
# ==========================================
class AdvancedCostModel:
    def __init__(self, hw: HardwareConfig):
        self.hw = hw

    def get_raw_input_tile_dim(self, t_oh, t_ow, layer):
        # 考虑 stride 带来的输入放大
        # 注意：这里计算的是 "Needed Input Receptive Field"
        t_ih = (t_oh - 1) * layer.S + layer.K_h
        t_iw = (t_ow - 1) * layer.S + layer.K_w
        return t_ih, t_iw

    def evaluate(self, layer: LayerParams, t_oh, t_ow, t_oc, t_ic):
        # 1. 实际数据维度 (Handling boundary cases)
        real_tile_ic = min(t_ic, layer.IC)
        real_tile_oc = min(t_oc, layer.OC)

        # 2. ACC 空间检查 (Output Stationary / Partial Sums)
        # 必须存储完整的输出 Tile (int32)
        acc_needed = t_oh * t_ow * real_tile_oc * self.hw.dtype_acc
        
        # 3. SPM 空间检查 (Double Buffering 通常需要 x2，这里先按单buffer算，如果OOM可进一步放宽)
        # Input Tile (Feature Map)
        raw_t_ih, raw_t_iw = self.get_raw_input_tile_dim(t_oh, t_ow, layer)
        input_needed = raw_t_ih * raw_t_iw * real_tile_ic * self.hw.dtype_input
        
        # Weight Tile
        weight_needed = layer.K_h * layer.K_w * real_tile_ic * real_tile_oc * self.hw.dtype_input
        
        spm_needed = input_needed + weight_needed

        # 4. 判定是否溢出
        if acc_needed > self.hw.acc_size_bytes:
            return None # ACC OOM
        if spm_needed > self.hw.spm_size_bytes:
            return None # SPM OOM

        # 5. 性能计算 (Latency)
        # 循环次数 (Ceil)
        n_h = math.ceil(layer.OH / t_oh)
        n_w = math.ceil(layer.OW / t_ow)
        n_spatial = n_h * n_w
        n_oc = math.ceil(layer.OC / t_oc)
        n_ic = math.ceil(layer.IC / t_ic)
        
        # Traffic Calculation
        # Input Traffic (简单模型：忽略重叠复用带来的节省，假设每次 Tile 都重新加载)
        traffic_in = input_needed * n_spatial * n_oc * n_ic
        # Weight Traffic
        traffic_wgt = weight_needed * n_spatial * n_oc * n_ic
        # Output Traffic
        traffic_out = layer.OH * layer.OW * layer.OC * self.hw.dtype_input
        
        total_traffic_mb = (traffic_in + traffic_wgt + traffic_out) / 1024**2
        
        # Time
        time_transfer_ms = (total_traffic_mb / self.hw.measured_bandwidth_mbps) * 1000
        
        # Compute / Instruction Time
        # 硬件按固定阵列大小计算，即使真实数据较少也要跑满周期
        cycles_per_tile = (math.ceil(t_ic/self.hw.u_ic) * math.ceil(t_oc/self.hw.u_oc) * layer.K_h * layer.K_w)
        # 这里的 cycles 是指内层乘加次数，简化为指令数评估
        # 实际指令数 = Load + Load + Compute + Store
        # 简化模型：使用 measured_latency_us 作为每 "Block 指令" 的平均开销
        total_tiles = n_spatial * n_oc * n_ic
        time_inst_ms = (total_tiles * self.hw.measured_latency_us) / 1000
        
        est_latency_ms = time_transfer_ms + time_inst_ms

        return {
            'est_latency_ms': est_latency_ms,
            'spm_util': spm_needed / self.hw.spm_size_bytes,
            'acc_util': acc_needed / self.hw.acc_size_bytes,
            'config': f"{t_oh}x{t_ow}_{t_ic}x{t_oc}"
        }

# ==========================================
# 5. DSE 探索逻辑 (修复 Search Loop)
# ==========================================
class DesignSpaceExplorer:
    def __init__(self, layers: List[LayerParams], total_mem_bytes: int):
        self.layers = layers
        self.total_mem_bytes = total_mem_bytes
        self.base_hw = HardwareConfig(0, 0)

    def search_layer(self, layer: LayerParams, model: AdvancedCostModel):
        min_lat = float('inf')
        best_res = None
        
        # 修复 Range 逻辑：
        # 1. 确保至少尝试一次最小切分 (base_hw)，即使 layer.OH 很小
        # 2. 限制最大切分不超过 64/128 (搜索剪枝)
        # 3. 如果 layer 维度小于硬件维度 (e.g. 7 vs 8)，range函数需要处理
        
        def safe_range(base_dim, max_dim, step):
            # 确保 stop 至少比 start 大，保证循环至少执行一次
            limit = min(max_dim + step, 128) 
            return range(step, limit, step)

        # 搜索空间定义
        r_oh = safe_range(self.base_hw.u_oh, layer.OH, self.base_hw.u_oh)
        r_ow = safe_range(self.base_hw.u_ow, layer.OW, self.base_hw.u_ow)
        
        # Channel 搜索通常可以步长更大，或者搜索全部
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

        # 遍历 SPM/ACC 配比: 10% - 90%
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
                    # Debug: 打印这一层为什么挂了
                    # print(f"  [DEBUG] Layer {layer.name} Failed at ratio {ratio} (OH={layer.OH}, OW={layer.OW})")
                    break
                total_lat += res['est_latency_ms']
                layer_data.append((layer.name, res))
            
            if is_valid_config:
                print(f"  Ratio {ratio:.1f} (SPM:{spm//1024}k/ACC:{acc//1024}k) -> {total_lat:.2f} ms")
                if total_lat < min_global_lat:
                    min_global_lat = total_lat
                    best_global = {'ratio': ratio, 'lat': total_lat, 'layers': layer_data}
            else:
                print(f"  Ratio {ratio:.1f} -> Failed (OOM or Invalid)")

        return best_global

# ==========================================
# 6. 执行
# ==========================================
if __name__ == "__main__":
    layers = ModelBuilder.get_resnet18_layers()
    print(f"Model: ResNet18 ({len(layers)} Conv Layers)")
    
    # 验证最后一层尺寸，确保没有 Collapse
    last = layers[-1]
    print(f"Check Last Layer: {last.name}, Output: {last.OH}x{last.OW}, IC={last.IC}, OC={last.OC}")

    dse = DesignSpaceExplorer(layers, 512 * 1024)
    best = dse.run()

    if best:
        print("\n" + "="*60)
        print(f"✅ Best Configuration: Ratio {best['ratio']}")
        print(f"   SPM: {int(512*best['ratio'])} KB | ACC: {int(512*(1-best['ratio']))} KB")
        print(f"   Total Latency: {best['lat']:.2f} ms")
        print("="*60)
        print(f"{'Layer':<15} | {'Tile (OHxOW_ICxOC)':<20} | {'Lat (ms)':<10} | {'SPM%':<6} | {'ACC%':<6}")
        print("-" * 70)
        for name, data in best['layers']:
            print(f"{name:<15} | {data['config']:<20} | {data['est_latency_ms']:<10.2f} | "
                  f"{data['spm_util']*100:<6.1f} | {data['acc_util']*100:<6.1f}")
    else:
        print("\n❌ All configurations failed. Please check hardware constraints.")