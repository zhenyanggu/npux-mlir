# 三个拆分模型的 VPU 算子映射分析

## 1. 分析对象

本分析按你说的 emb / encoder / decoder 三个拆分后模型进行：

1. `aicas_run/embed_tokens_fp16.onnx`
2. `aicas_run/vision_encoder_fixed.onnx`
3. `aicas_run/tmp/decoder_model_merged_fp16_rewritten.onnx`

同时对比了原始 decoder：`aicas_run/decoder_model_merged_fp16.onnx`（仍含 `com.microsoft` 自定义算子）。

## 2. 实际算子统计（拆分后三模型）

### 2.1 节点规模

- embed: 2 nodes, 2 种算子
- encoder: 873 nodes, 34 种算子
- decoder(rewritten): 3315 nodes, 25 种算子

### 2.2 三模型算子并集（37种）里最主要的计算算子

- `Mul`: 593
- `Reshape`: 634（形状类，不是算术）
- `MatMul`: 386
- `Add`: 402
- `Transpose`: 344（数据重排）
- `Gather`: 267（索引类）
- `Concat`: 267（拼接类）
- `Softmax`: 44
- `Sigmoid`: 32
- `ReduceMean`: 115
- `Sqrt`: 90
- `Div`: 92
- `Sub`: 90

## 3. 哪些算子可以放到 VPU 执行

按 VPU ISA（`aicas_run/VPU.md`）能力，建议分为三类。

### 3.1 直接适配（优先放 VPU）

这个文档是ai写的，下面的matmul和conv等原来脉动阵列做的事情还是脉动阵列做，忘记说了

1. `MatMul` / `Conv`（Conv可im2col后转MatMul）
   - 对应 VPU: `vfwmac`/`vfwmul` + ACC + `vle/vse`
   - 这是三模型最核心热点（特别是 encoder/decoder）。
2. `Add` / `Sub` / `Mul`
   - 对应 VPU: `vfadd`/`vfsub`/`vfmul`
3. `ReduceSum` / `ReduceMean`
   - 对应 VPU: `vfredsum`（Mean 再乘系数）
4. `Clip`
   - 对应 VPU: `vfmin`/`vfmax`
5. `Pow(x,2)`（encoder里都是平方）
   - 可重写为 `Mul(x,x)`，走 `vfmul`

### 3.2 可由 VPU 组合实现（建议作为第二优先级）

1. `Div`
   - 用 `vrecip` + `vfmul` 近似实现（需精度评估）
2. `Sqrt`
   - 用 `vrsqrt` + 乘法/倒数组合
3. `Softmax`
   - `max-reduce` + `sub` + `exp` + `sum-reduce` + `div`
   - VPU对应：`vfredmax`/`vfsub`/`vexp2`/`vfredsum`/`vrecip`
4. `Sigmoid` / `Tanh`
   - 可由 `exp` + 基本算子拼出来（近似实现）

### 3.3 不建议放 VPU（留给前端/CPU/DMA/调度层）

1. 形状/布局类：
   - `Reshape`, `Transpose`, `Unsqueeze`, `Squeeze`, `Concat`, `Split`, `Flatten`, `Expand`, `Tile`, `Slice`, `Shape`, `Range`, `ConstantOfShape`
2. 索引/控制流类：
   - `Gather`, `GatherND`, `ScatterND`, `Where`, `Equal`, `Less`, `Greater`, `NonZero`, `Not`
3. embed 模型中的 `Gather` 本质是 embedding 索引读取，不是 VPU 向量算术热点。

## 4. 这三个模型里“不需要”或“可暂不实现”的 VPU 模块/算子

> 这里的“不需要”指：从这三张图的语义看，不是推理主路径必需，可后置。

### 4.1 可暂不投入的指令子集

1. 纯整数逻辑/移位链路
   - `vand`, `vor`, `vxor`, `vsll`, `vsra`, `vsrl`
2. 整数饱和/无符号扩展族
   - `vsadd`, `vssub`, `vminu`, `vmaxu`, `vsaddu`, `vssubu`
3. 量化/反量化搬运族（针对 INT8/INT16 部署）
   - `vfncvt.i8`, `vncvt.s8`, `vncvt.s16`, `vwmv.s16`, `vfncvt.s8.f` 等
4. 掩码相关（文档中也标注未实现）
   - `vmerge`, `Mask Ops`
5. `vlog2`
   - 当前三模型没有直接需求（softmax/激活路径更多依赖 exp/recip/rsqrt）

### 4.2 模块级可裁剪结论

1. **必须保留**：LSU、MAC(含ACC)、BF16 FPU、SFU(`exp/recip/rsqrt`)、浮点规约(`vfredsum/vfredmax`)。
2. **可弱化/后置**：整数位运算与移位、量化转换链路、掩码子系统。

## 5. 额外提醒（精度格式）

- 这三模型数据类型主要是 `FLOAT/FLOAT16`，而 VPU 规范以 `BF16` + 整数为主。
- 若要大规模下沉到 VPU，建议先确定：
  1. 前端是否统一转 BF16；
  2. `Div/Sqrt/Softmax/Sigmoid/Tanh` 的近似误差阈值；
  3. MatMul/Softmax/LayerNorm 作为第一批下沉优先级。

## 6. 对原始 decoder 的补充

- `aicas_run/decoder_model_merged_fp16.onnx` 仍有：
  - `RotaryEmbedding` 64
  - `SkipSimplifiedLayerNormalization` 64
  - `MultiHeadAttention` 32
- 你当前的 rewritten decoder 已把这些展开为标准 ONNX 基础算子；VPU 映射应以 rewritten 版本为准。

