# Decoder Custom Op 拆分与编译指南

本文档用于复用以下流程：
- 将带 `com.microsoft` 自定义算子的 decoder ONNX 模型离线改写为标准 ONNX 子图
- 用 `onnx-mlir` 尝试编译改写后的模型

适用于当前 `aicas_run` 的 SmolVLM2 decoder（FP16/FP32），也可作为其他类似模型的模板。

## 1. 目录规范

建议统一按下面的布局管理：

- 模型输入/输出：`aicas_run/models/`
- 脚本：`aicas_run/scripts/`
- 编译中间产物和 `.so`：`aicas_run/tmp/`
- 过程文档：`aicas_run/docs/`

本流程使用脚本：
- `aicas_run/scripts/rewrite_decoder_custom_ops.py`
- `aicas_run/scripts/validate_decoder_rewrite.py`

## 2. 环境准备

在仓库根目录执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate npux-mlir
source scripts/activate_env.sh
```

## 3. 改写前检查（可选但推荐）

统计模型中的自定义算子数量：

```bash
python3 - <<'PY'
import onnx
from collections import Counter

model = onnx.load("aicas_run/models/decoder_model_merged.onnx")
counter = Counter((n.domain, n.op_type) for n in model.graph.node)
print({k: v for k, v in counter.items() if k[0] == "com.microsoft" or (k[0] == "" and k[1] == "SimplifiedLayerNormalization")})
PY
```

## 4. 执行改写（核心步骤）

```bash
python3 aicas_run/scripts/rewrite_decoder_custom_ops.py \
  aicas_run/models/decoder_model_merged.onnx \
  -o aicas_run/models/decoder_model_merged_fp32_rewritten.onnx
```

说明：
- 该脚本会改写 `SimplifiedLayerNormalization / SkipSimplifiedLayerNormalization / RotaryEmbedding / MultiHeadAttention`
- 脚本已支持按输入张量 dtype 自动生成 MHA 的 `scale` 常量，兼容 FP16/FP32

## 5. 改写后检查（推荐）

```bash
python3 - <<'PY'
import onnx
from collections import Counter

model = onnx.load("aicas_run/models/decoder_model_merged_fp32_rewritten.onnx")
counter = Counter((n.domain, n.op_type) for n in model.graph.node)
custom = {k: v for k, v in counter.items() if k[0] == "com.microsoft" or k[1] == "Custom"}
print("custom:", custom)
PY
```

期望输出：`custom: {}`

## 6. 用 onnx-mlir 编译

```bash
mkdir -p aicas_run/tmp
build/Release/bin/onnx-mlir \
  aicas_run/models/decoder_model_merged_fp32_rewritten.onnx \
  -o aicas_run/tmp/decoder_model_merged_fp32_rewritten
```

期望产物：
- `aicas_run/tmp/decoder_model_merged_fp32_rewritten.so`
- （大模型常见）`aicas_run/tmp/decoder_model_merged_fp32_rewritten.constants.bin`

## 7. 数值一致性验证（可选）

```bash
python3 aicas_run/scripts/validate_decoder_rewrite.py \
  --original-decoder aicas_run/models/decoder_model_merged.onnx \
  --rewritten-decoder aicas_run/models/decoder_model_merged_fp32_rewritten.onnx \
  --skip-real --random-seq-len 3
```

脚本会根据 ORT 输入签名自动适配 dtype（包括 FP16/FP32 的 past KV）。

## 8. 跑其他模型时的最小改动点

优先只改命令参数，不改脚本逻辑：

1. 输入模型路径（`rewrite_decoder_custom_ops.py` 第一个参数）
2. 改写后输出路径（`-o`）
3. 编译输出前缀（`onnx-mlir -o`）

如果报错，优先检查：
- 改写前后是否仍有 `com.microsoft` 域算子
- MHA 输入是否含当前脚本未覆盖的可选输入（如 `past_key/past_value/cache_indirection`）
- 模型中张量 dtype 是否与常量 dtype 一致

## 9. 一键回放（当前 FP32 成功路径）

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate npux-mlir
source scripts/activate_env.sh

python3 aicas_run/scripts/rewrite_decoder_custom_ops.py \
  aicas_run/models/decoder_model_merged.onnx \
  -o aicas_run/models/decoder_model_merged_fp32_rewritten.onnx

build/Release/bin/onnx-mlir \
  aicas_run/models/decoder_model_merged_fp32_rewritten.onnx \
  -o aicas_run/tmp/decoder_model_merged_fp32_rewritten
```
