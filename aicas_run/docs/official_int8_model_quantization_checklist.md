# 官方 int8 ONNX 模型量化清单

本文档只基于以下 ONNX 模型本体结构分析，不基于仓库内脚本推断：

- [vision_encoder_int8.onnx](/home/gugugu2404/research/npu-mlir/npux-mlir/aicas_run/models/vision_encoder_int8.onnx)
- [decoder_model_merged_int8.onnx](/home/gugugu2404/research/npu-mlir/npux-mlir/aicas_run/models/decoder_model_merged_int8.onnx)
- [embed_tokens_int8.onnx](/home/gugugu2404/research/npu-mlir/npux-mlir/aicas_run/models/embed_tokens_int8.onnx)

## 量化方式说明

- `动态非对称激活量化`
  - 图中表现为 `DynamicQuantizeLinear -> MatMulInteger/ConvInteger`
  - 激活在运行时被量化，带 `zero_point`，属于动态仿射量化，通常是非对称量化
- `INT8 对称权重量化`
  - 权重张量是 `INT8`
  - 直接检查到 `weight_zero_point = 0`
- `UINT8 非对称权重量化`
  - 权重张量是 `UINT8`
  - 且 `zero_point != 0`
  - 这类形式出现在 embedding table / position embedding
- `保持浮点`
  - 没有进入整数核，仍由浮点算子执行

## 总体结论

- `vision_encoder_int8.onnx`
  - 线性层和卷积层使用：`激活动态非对称 + 权重INT8对称`
  - `position embedding` 使用：`UINT8非对称 -> DequantizeLinear`
  - attention 内部两步 `MatMul` 仍是浮点，没有量化成整数核
- `decoder_model_merged_int8.onnx`
  - 所有主要线性层使用：`激活动态非对称 + 权重INT8对称`
  - `MultiHeadAttention`、`RotaryEmbedding`、LayerNorm 类、`repeat_kv/Gather` 没有量化成整数核
- `embed_tokens_int8.onnx`
  - embedding table 使用：`UINT8非对称`
  - 形式为 `Gather(UINT8 weight) -> DequantizeLinear -> FLOAT output`

## Vision Encoder 清单

### Embedding 与输入侧

- `patch_embedding/Conv`
  - 状态：已量化
  - 方式：`DynamicQuantizeLinear -> ConvInteger`
  - 量化类型：`激活动态非对称 + 权重INT8对称`

- `embeddings/position_embedding`
  - 状态：已量化存储
  - 方式：`Gather(UINT8 weight_quantized) -> DequantizeLinear`
  - 量化类型：`UINT8非对称`
  - 备注：直接检查到 `zero_point = 116`

- `connector/modality_projection/proj`
  - 状态：已量化
  - 方式：`DynamicQuantizeLinear -> MatMulInteger`
  - 量化类型：`激活动态非对称 + 权重INT8对称`

### Encoder Blocks

`vision_model/encoder/layers.0 ~ layers.11` 每一层结构一致：

- 已量化
  - `self_attn/q_proj`
  - `self_attn/k_proj`
  - `self_attn/v_proj`
  - `self_attn/out_proj`
  - `mlp/fc1`
  - `mlp/fc2`
  - 方式：`DynamicQuantizeLinear -> MatMulInteger`
  - 量化类型：`激活动态非对称 + 权重INT8对称`

- 未量化
  - `self_attn/MatMul`
  - `self_attn/MatMul_1`
  - 说明：这两个是 attention 内部的两步浮点矩阵乘
  - 类型：`保持浮点`

- 其余保持浮点
  - `Softmax`
  - 大量 `Add / Mul / Reshape / Transpose / Slice`

### Vision 统计

- `ConvInteger = 1`
- `MatMulInteger = 73`
- 残留浮点 `MatMul = 24`
  - 即 `12 层 x 2 个`

### Vision 分层清单

| 层号 | 已量化 | 未量化 | 量化方式说明 |
| --- | --- | --- | --- |
| 0 | `self_attn/q_proj`, `self_attn/k_proj`, `self_attn/v_proj`, `self_attn/out_proj`, `mlp/fc1`, `mlp/fc2` | `self_attn/MatMul`, `self_attn/MatMul_1` | 已量化部分均为 `DynamicQuantizeLinear -> MatMulInteger`，即 `激活动态非对称 + 权重INT8对称` |
| 1 | `self_attn/q_proj`, `self_attn/k_proj`, `self_attn/v_proj`, `self_attn/out_proj`, `mlp/fc1`, `mlp/fc2` | `self_attn/MatMul`, `self_attn/MatMul_1` | 同上 |
| 2 | `self_attn/q_proj`, `self_attn/k_proj`, `self_attn/v_proj`, `self_attn/out_proj`, `mlp/fc1`, `mlp/fc2` | `self_attn/MatMul`, `self_attn/MatMul_1` | 同上 |
| 3 | `self_attn/q_proj`, `self_attn/k_proj`, `self_attn/v_proj`, `self_attn/out_proj`, `mlp/fc1`, `mlp/fc2` | `self_attn/MatMul`, `self_attn/MatMul_1` | 同上 |
| 4 | `self_attn/q_proj`, `self_attn/k_proj`, `self_attn/v_proj`, `self_attn/out_proj`, `mlp/fc1`, `mlp/fc2` | `self_attn/MatMul`, `self_attn/MatMul_1` | 同上 |
| 5 | `self_attn/q_proj`, `self_attn/k_proj`, `self_attn/v_proj`, `self_attn/out_proj`, `mlp/fc1`, `mlp/fc2` | `self_attn/MatMul`, `self_attn/MatMul_1` | 同上 |
| 6 | `self_attn/q_proj`, `self_attn/k_proj`, `self_attn/v_proj`, `self_attn/out_proj`, `mlp/fc1`, `mlp/fc2` | `self_attn/MatMul`, `self_attn/MatMul_1` | 同上 |
| 7 | `self_attn/q_proj`, `self_attn/k_proj`, `self_attn/v_proj`, `self_attn/out_proj`, `mlp/fc1`, `mlp/fc2` | `self_attn/MatMul`, `self_attn/MatMul_1` | 同上 |
| 8 | `self_attn/q_proj`, `self_attn/k_proj`, `self_attn/v_proj`, `self_attn/out_proj`, `mlp/fc1`, `mlp/fc2` | `self_attn/MatMul`, `self_attn/MatMul_1` | 同上 |
| 9 | `self_attn/q_proj`, `self_attn/k_proj`, `self_attn/v_proj`, `self_attn/out_proj`, `mlp/fc1`, `mlp/fc2` | `self_attn/MatMul`, `self_attn/MatMul_1` | 同上 |
| 10 | `self_attn/q_proj`, `self_attn/k_proj`, `self_attn/v_proj`, `self_attn/out_proj`, `mlp/fc1`, `mlp/fc2` | `self_attn/MatMul`, `self_attn/MatMul_1` | 同上 |
| 11 | `self_attn/q_proj`, `self_attn/k_proj`, `self_attn/v_proj`, `self_attn/out_proj`, `mlp/fc1`, `mlp/fc2` | `self_attn/MatMul`, `self_attn/MatMul_1` | 同上 |

## Decoder 清单

### 每层统一模式

`model/layers.0 ~ layers.31` 每一层结构一致：

- 已量化
  - `attn/q_proj`
  - `attn/k_proj`
  - `attn/v_proj`
  - `attn/o_proj`
  - `mlp/gate_proj`
  - `mlp/up_proj`
  - `mlp/down_proj`
  - 方式：`DynamicQuantizeLinear -> MatMulInteger`
  - 量化类型：`激活动态非对称 + 权重INT8对称`

- 未量化
  - `attn/MultiHeadAttention`
  - `attn/q_rotary/RotaryEmbedding`
  - `attn/k_rotary/RotaryEmbedding`
  - `attn/repeat_kv/Gather`
  - `input_layernorm`
  - `post_attention_layernorm`
  - 类型：`保持浮点 / 融合浮点算子`

### 层外节点

- `lm_head`
  - 状态：已量化
  - 方式：`DynamicQuantizeLinear -> MatMulInteger`
  - 量化类型：`激活动态非对称 + 权重INT8对称`

- `final_norm_layernorm`
  - 状态：未量化成整数核
  - 类型：`保持浮点`
  - 备注：其输出后面接了 `DynamicQuantizeLinear`，再喂给 `lm_head`

### Decoder 统计

- `MatMulInteger = 225`
  - `32 层 x 7 + lm_head 1`
- 普通浮点 `MatMul = 0`
- 未量化的重要节点统计
  - `MultiHeadAttention = 32`
  - `RotaryEmbedding = 64`
  - `SkipSimplifiedLayerNormalization = 64`
  - `SimplifiedLayerNormalization = 1`
  - `repeat_kv/Gather = 128`

### Decoder 分层清单

| 层号 | 已量化 | 未量化 | 量化方式说明 |
| --- | --- | --- | --- |
| 0 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | `attn/MultiHeadAttention`, `attn/q_rotary/RotaryEmbedding`, `attn/k_rotary/RotaryEmbedding`, `attn/repeat_kv/Gather`, `input_layernorm`, `post_attention_layernorm` | 已量化部分均为 `DynamicQuantizeLinear -> MatMulInteger`，即 `激活动态非对称 + 权重INT8对称` |
| 1 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 2 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 3 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 4 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 5 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 6 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 7 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 8 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 9 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 10 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 11 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 12 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 13 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 14 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 15 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 16 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 17 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 18 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 19 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 20 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 21 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 22 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 23 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 24 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 25 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 26 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 27 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 28 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 29 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 30 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |
| 31 | `attn/q_proj`, `attn/k_proj`, `attn/v_proj`, `attn/o_proj`, `mlp/gate_proj`, `mlp/up_proj`, `mlp/down_proj` | 同上 | 同上 |

## Embed Tokens 清单

- `embedding table`
  - 状态：已量化存储
  - 方式：`Gather(weight_quantized) -> DequantizeLinear`
  - 量化类型：`UINT8非对称`
  - 备注：
    - `weight_quantized` 是 `UINT8`
    - 直接检查到 `weight_zero_point = 168`

- `Gather`
  - 状态：不是 `MatMulInteger / ConvInteger` 那种整数核
  - 实际含义：对量化后的 embedding 表做查表，再立刻反量化成浮点输出

## 一句话总结

这批官方 int8 ONNX 模型是混合量化方案：

- 线性层/卷积层：`激活动态非对称 + 权重INT8对称`
- embedding / position embedding：`UINT8非对称权重量化`
- attention 核心软算子、Rotary、LayerNorm、Softmax、repeat_kv 等：大多仍保持浮点
