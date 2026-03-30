# GEMM/MatMul 尺寸分析与分块建议（aicas_run）

更新时间：2026-03-27  
分析环境：`conda env = npux-mlir`，`onnx==1.17.0`，使用 `onnx.shape_inference` 解析

## 1. 说明

- `aicas_run/throughput_eval.py` 是吞吐测试脚本（调用本地 `llama-server`），不直接分析 ONNX 图中的 GEMM/MatMul。
- 本文结论来自对 `aicas_run` 下 ONNX 模型的实际图结构扫描和 shape inference。

## 2. 分析对象

- `aicas_run/decoder_model_merged_fp16.onnx`
- `aicas_run/vision_encoder_fp16.onnx`
- `aicas_run/embed_tokens_fp16.onnx`
- `aicas_run/results/quantized/decoder_prefill_only_test.onnx`
- `aicas_run/results/quantized/decoder_prefill_two_test.onnx`
- `aicas_run/results/quantized/vision_test.onnx`

## 3. 结论总览

- 本批模型中未检测到 `Gemm`，仅有 `MatMul`。
- `embed_tokens_fp16.onnx` 中无 `MatMul/Gemm`。
- 量化前后（fp16 vs quantized）`MatMul` 尺寸分布一致。
- 动态维度主要出现在 Decoder：`M=sequence_length`。
- Vision 模型 `MatMul` 维度为静态常数（本批模型中）。

## 4. MatMul 常见尺寸

说明：以下按 `M,K,N` 记法给出（即 `A[M,K] x B[K,N] -> C[M,N]`）。

### 4.1 Decoder（fp16/quantized 一致）

模型：
- `decoder_model_merged_fp16.onnx`
- `decoder_prefill_only_test.onnx`
- `decoder_prefill_two_test.onnx`

`MatMul` 总数：225

高频尺寸：
- `(sequence_length, 960, 960)` ×64
- `(sequence_length, 960, 320)` ×64
- `(sequence_length, 960, 2560)` ×64
- `(sequence_length, 2560, 960)` ×32
- `(sequence_length, 960, 49280)` ×1（LM Head）

维度特征：
- `M` 为动态符号维 `sequence_length`。
- `K` 主要集中在 `960`，FFN 方向涉及 `2560`，输出词表投影为大 `N=49280`。

### 4.2 Vision（fp16/quantized 一致）

模型：
- `vision_encoder_fp16.onnx`
- `vision_test.onnx`

`MatMul` 总数：97

高频尺寸：
- `(1024, 768, 768)` ×48
- `(1024, 64, 1024)` ×12
- `(1024, 1024, 64)` ×12
- `(1024, 768, 3072)` ×12
- `(1024, 3072, 768)` ×12
- `(64, 12288, 960)` ×1

维度特征：
- 主体是固定 `M=1024` 的静态矩阵乘，典型 ViT/MLP 结构（`768 <-> 3072`）。

## 5. 动态维度风险判断

是否存在动态维度：**有，但主要集中在 Decoder**。

- Decoder：`M=sequence_length` 动态，意味着 prefill 与 decode 的最优分块可能不同。
- Vision：当前模型中关键 MatMul 维度是静态值，可采用固定优化配置。

## 6. 对分块策略与 NPU 存储分配的建议

说明：本节基于**旧版建模**（固定 auto-tile、ACC 常驻占用、含旧的 `K<=2048` 约束）得到，作为历史参考保留。  
若面向当前 aicas 比赛的新版评估标准，请以本文 **第 10 节** 的结果为准。

结合 `aicas_run/results/*gemm_search_256*.json` 的既有搜索结果：

- Decoder 最优倾向：`SPM/ACC = 32KB / 224KB`（偏 ACC）。
- Vision 最优倾向：`SPM/ACC = 112KB / 144KB`（更均衡）。

建议：

1. 不要用一套统一配置覆盖 Decoder+Vision，建议至少双 Profile。
2. Decoder 再细分 prefill/decode：
   - decode（`M=1`）偏向小 `M` 场景优化。
   - prefill（`M=sequence_length`，随输入长度变化）建议分档调参（如 1/16/64/256）。
3. 对 `N=49280` 的 LM Head（单层但很大）单独策略处理，避免拖慢全局均值。
4. Vision 可按静态主形状做离线定制 tile，并固定存储分配策略。

## 7. 推荐分析工具/流程

推荐用于持续分析的最小工具链：

- `onnx.shape_inference`：提取图内 `MatMul/Gemm` 实际 `M/K/N`。
- Netron：人工快速核对关键大算子（如 LM Head）。
- `model_test/scripts/conv_tile.py`：做 SRAM split + tile 搜索（当前结果即来自该类流程）。
- （可选）`onnxruntime profiling`：做运行时热点交叉验证，避免纯静态统计偏差。

建议形成固定流程：
- Step 1：先静态提取尺寸分布（按频次 + 按估算 MAC 加权）。
- Step 2：对 top-k 尺寸做 tile/sram 搜索。
- Step 3：按 prefill/decode 与 vision 分别固化配置。

## 8. 与吞吐脚本的关系

- `aicas_run/throughput_eval.py` 输出的是服务端推理吞吐指标（prefill/decode tokens/s）。
- 本文用于解释“吞吐背后的算子形状与存储分配原因”，二者应联合使用：
  - 吞吐脚本用于端到端验证；
  - MatMul 尺寸分析用于定位瓶颈与指导 NPU 分块策略。

## 9. `seqlen` 典型值实验（参考 `throughput_eval.py` 场景）

实验日期：2026-03-27

### 9.1 实验目的

- 明确 Decoder 动态维 `sequence_length`（即 prefill 的 `M`）在实际输入下的典型值与分布；
- 为 `SPM/ACC` 容量取值提供可执行的分档依据。

### 9.2 结论先行

- 若按 `throughput_eval.py` 的固定长提示词（`LONG_PROMPT`）场景：
  - **典型值建议：`seqlen = 694`**（对应当前 `aicas_run/image.png`）。
  - 更稳健分档建议：`246 / 438 / 694 / 950 / 1206`。
- `ram600_best_config.json` 中的 `sequence_length=1245` 更接近偏保守上界档，不是中位典型值。

### 9.3 计算关系

该模型输入长度可写为：

- `seqlen = text_control_tokens + image_seq_len * num_images`

其中：

- `image_seq_len = 64`（见 `aicas_run/processor_config.json`）
- `num_images` 为图像切分后视觉块数（含全局图），上限为 `17 (=4x4+1)`。

### 9.4 实验设置

1. 文本 token 统计：
   - 使用本地 tokenizer 对 `throughput_eval.py` 风格模板  
     `"<|im_start|>User:<image>{prompt}<end_of_utterance>\\nAssistant:"` 计数。
2. 图像块数统计：
   - 按 `SmolVLM` 图像切分规则（`do_image_splitting=true`，`max_longest=2048`，块边长 `512`）估计 `num_images`。
3. 统计数据：
   - `aicas_run/image.png`（吞吐脚本默认图）
   - `aicas_run/data/ocrbench/eval_100.json`
   - `aicas_run/data/ocrbench/calib_64.json`
   - `aicas_run/data/ocrbench/images` 全量 1000 张（固定 `LONG_PROMPT`）

说明：
- 当前环境缺少 `openai` 包，无法直接跑 `throughput_eval.py` 从服务端回读 `prompt_tokens`；
- 本节采用离线等价输入构造统计，结论用于 `seqlen` 档位与容量规划。

### 9.5 结果明细

#### A) `throughput_eval.py` 默认输入（`aicas_run/image.png` + `LONG_PROMPT`）

- 图像尺寸：`1962 x 843`
- `num_images = 9`
- `text_control_tokens = 118`
- `seqlen = 118 + 64 * 9 = 694`

#### B) `eval_100.json`（100 张）

固定 `LONG_PROMPT`：
- `seqlen`：min/mean/p50/p90/p95/max = `246 / 280.56 / 246 / 380 / 438 / 758`
- `num_images` 分布：`2(76), 3(11), 4(3), 5(8), 7(1), 10(1)`

使用原始短问题（question）：
- `seqlen`：min/mean/p50/p90/p95/max = `141 / 178.05 / 143 / 277 / 335 / 659`

#### C) `calib_64.json`（64 张）

固定 `LONG_PROMPT`：
- `seqlen`：min/mean/p50/p90/p95/max = `246 / 307.0 / 246 / 438 / 438 / 950`
- `num_images` 分布：`2(45), 3(7), 5(9), 7(1), 13(2)`

使用原始短问题（question）：
- `seqlen`：min/mean/p50/p90/p95/max = `141 / 204.3 / 143 / 335 / 339 / 850`

#### D) `data/ocrbench/images` 全 1000 张（固定 `LONG_PROMPT`）

- `seqlen`：min/mean/p50/p90/p95/p99/max =  
  `246 / 386.42 / 246 / 694 / 950 / 1206 / 1206`
- `num_images` 分布：  
  `2(533), 3(73), 4(6), 5(243), 7(35), 9(39), 10(3), 13(28), 17(40)`

### 9.6 对 SPM/ACC 分档的直接建议

若以 `throughput_eval.py` 长提示词场景为目标，建议至少覆盖以下 `seqlen` 档位：

1. `246`（主峰，小图/低切分）
2. `438`（次峰，中等切分）
3. `694`（高负载常见档，且是当前默认图实测值）
4. `950`（高分位档，约 p95）
5. `1206`（上界档，接近 `num_images=17` 场景）

若只选单一“典型值”做快速评估，优先用：**`seqlen = 694`**。

## 10. AICAS 比赛新版 GEMM 建模与搜索结果

实验日期：2026-03-27

### 10.1 新版评估标准

面向 aicas 比赛，对 `model_test/scripts/conv_tile.py` 的 GEMM 建模口径做了如下修订：

1. **先不计纯计算量时间**。因为总 FLOPs 不随分块改变，当前仅评估：
   - 指令固定开销：`1.54us / instruction`
   - DMA 数据搬运时间
2. 参与建模的指令包括：
   - DMA 搬入 / 搬出指令
   - GEMM 计算指令
   - bias 搬运指令
3. 一条 GEMM 计算指令表示一次：
   - `(32 x k) * (k x 32)`
   - 其中 `k` **不再设置 2048 上限**
4. bias 建模规则：
   - 若该层有 bias，则每切换一次输出 `32 col`，需要 `mvin` 一次 bias
   - 一次 bias 大小固定为 `32 x int32 = 128B`
5. ACC 占用规则修订为：
   - 若 `K` 维 **不分块**（即一次算完整个 `K`），则 **ACC 不占空间**
   - 若 `K` 维 **发生分块**，则需要用 ACC 保存中间部分和，大小按 `t_n * t_m * int32` 估算
6. 搜索策略修订为：
   - 对每个给定的 `SPM/ACC` 划分，**显式搜索**该配置下最优的 `(t_n, t_m, t_k)`
   - 不再直接套旧版 `auto tile`
7. 循环顺序采用：
   - `K-tile` 在外层
   - 在内层同时评估 `A` 复用 / `B` 复用两种 DMA 复用方式，取较优者

### 10.2 AICAS 共用 workload 定义

本次不是分别为 Vision / Decoder 独立找配置，而是面向比赛形态，构造**一套共用 objective**：

- `Objective = Vision + Prefill + DecodeTokensWeight * DecodeStep`

其中：

- Vision 典型形状来自本文第 4.2 节
- Decoder Prefill / Decode 典型形状来自本文第 4.1 节
- Prefill 使用典型 `sequence_length = 694`
- Decode 使用 `M = 1`

Decoder 典型层按重复次数计入：

- `(694, 960, 960)` ×64
- `(694, 960, 320)` ×64
- `(694, 960, 2560)` ×64
- `(694, 2560, 960)` ×32
- `(694, 960, 49280)` ×1

Vision 典型层按重复次数计入：

- `(1024, 768, 768)` ×48
- `(1024, 64, 1024)` ×12
- `(1024, 1024, 64)` ×12
- `(1024, 768, 3072)` ×12
- `(1024, 3072, 768)` ×12
- `(64, 12288, 960)` ×1

### 10.3 Decode token 权重

为了避免继续使用旧的 `decode_tokens_assumed = 4096`，本次额外基于 `aicas_run/SmolVLM2.json` 中 100 条 OCRBench 样本结果，统计预测文本长度：

- 样本数：`100`
- 平均生成 token 数：`15.25`
- `p50 = 12`
- `p90 = 23`
- `p95 = 42`
- `max = 70`

因此新版默认采用：

- `DecodeTokensWeight = 15.25`

这比旧版 `4096` 更接近实际比赛单样本负载。

当前脚本已支持自动统计：

- 默认直接读取 `aicas_run/SmolVLM2.json`
- 使用仓库中的 tokenizer 文件自动估算平均 decode token 数
- 若需要，也可以继续通过 `--aicas-decode-tokens` 手动覆盖

### 10.4 搜索结果：总 SRAM = 256KB

搜索设置：

- `SPM + ACC = 256KB`
- 划分步长：`16KB`
- `jobs = 4`

最优结果：

- **`SPM / ACC = 208KB / 48KB`**
- Objective：`7051.78 ms`

Profile 分解：

- Vision：`707.91 ms`
- Prefill：`1750.55 ms`
- Decode：`301.20 ms / step`
- 加权后的 Decode：`4593.32 ms`

Top 候选：

1. `208KB / 48KB`：`7051.78 ms`
2. `192KB / 64KB`：`7091.58 ms`
3. `224KB / 32KB`：`7166.79 ms`

### 10.5 搜索结果：总 SRAM = 600KB

搜索设置：

- `SPM + ACC = 600KB`
- 划分步长：`16KB`
- `jobs = 4`

最优结果：

- **`SPM / ACC = 544KB / 56KB`**
- Objective：`6265.11 ms`

Profile 分解：

- Vision：`485.61 ms`
- Prefill：`1247.29 ms`
- Decode：`297.19 ms / step`
- 加权后的 Decode：`4532.21 ms`

Top 候选：

1. `544KB / 56KB`：`6265.11 ms`
2. `528KB / 72KB`：`6272.92 ms`
3. `512KB / 88KB`：`6272.94 ms`
4. `496KB / 104KB`：`6273.32 ms`
5. `560KB / 40KB`：`6285.14 ms`

### 10.6 结果解读

新版结果与旧版 `ram600_best_config.json` / `*gemm_search_256*.json` 的结论明显不同，核心原因有三点：

1. **ACC 不再被错误地常驻占用**
   - 只有 `K` 维实际分块时才需要 ACC
   - 这会显著降低“偏大 ACC”配置的虚假优势
2. **`K<=2048` 限制移除**
   - 大 `K` 层可以一次吃更大的 `t_k`
   - 这进一步降低了 K 分块次数与 ACC 压力
3. **显式 tile 搜索替代旧版 auto-tile**
   - 每个 `SPM/ACC` 配置下都会重新找最优 `(t_n, t_m, t_k)`
   - 因而能够更充分地把 SPM 用于提升 A/B tile 复用

综合来看，当前新版模型下有一个稳定趋势：

- **最优配置明显从“偏 ACC”转向“偏 SPM”**
- ACC 只需保留一个**较小但不为 0**的容量，用来覆盖确实需要 K 分块的层即可
- 一旦 SPM 足够大，Vision 与 Prefill 都能明显受益，而 Decode 改善相对有限

对于本批 workload：

- `256KB` 总量下，建议优先从 `208KB / 48KB` 附近开始试
- `600KB` 总量下，建议优先从 `544KB / 56KB`、`528KB / 72KB`、`512KB / 88KB` 这组附近开始试

### 10.7 脚本使用方式

先激活项目环境：

```bash
conda activate npux-mlir
source scripts/activate_env.sh
```

#### A) 面向 AICAS 比赛 workload 直接搜索共用配置

256KB：

```bash
python model_test/scripts/conv_tile.py \
  --op gemm \
  --gemm-source aicas \
  --sram-total 256 \
  --split-step-kb 16 \
  --aicas-prefill-seqlen 694 \
  --jobs 4 \
  -o aicas_256_search.json
```

600KB：

```bash
python model_test/scripts/conv_tile.py \
  --op gemm \
  --gemm-source aicas \
  --sram-total 600 \
  --split-step-kb 16 \
  --aicas-prefill-seqlen 694 \
  --jobs 4 \
  -o aicas_600_search.json
```

关键参数说明：

- `--gemm-source aicas`
  - 使用内置的 SmolVLM2 / OCRBench 典型 workload
- `--sram-total`
  - 总 SRAM 容量（KB）
- `--split-step-kb`
  - SPM/ACC 划分步长
- `--aicas-prefill-seqlen`
  - Prefill 的典型 `sequence_length`
- `--aicas-decode-tokens`
  - Decode 的加权 token 数；不传时自动从 `SmolVLM2.json` 统计
- `--aicas-model-dir`
  - tokenizer / 模型目录，默认 `aicas_run`
- `--aicas-decode-json`
  - 用于自动统计 decode token 数的结果 JSON，默认 `aicas_run/SmolVLM2.json`
- `--jobs`
  - 硬件候选并行搜索 worker 数

#### B) 对单个 ONNX 模型按新版 GEMM 模型建模

```bash
python model_test/scripts/conv_tile.py \
  --op gemm \
  --gemm-source onnx \
  --input aicas_run/decoder_model_merged_fp16.onnx \
  --spm 208 \
  --acc 48 \
  --dynamic-dim-map batch_size=1,sequence_length=694,past_sequence_length=0,total_sequence_length=694 \
  -o decoder_gemm_eval.json
```

说明：

- `--gemm-source onnx` 时，脚本会从 ONNX 图中提取 GEMM/MatMul 层
- `--gemm-source aicas` 时，不依赖 ONNX 解析，直接使用本文定义的典型 workload

### 10.8 当前结论

若当前目标是面向 aicas 比赛、使用一套共用配置覆盖 Vision + Prefill + Decode，则在**新版建模口径**下：

- `256KB`：优先考虑 **`208KB / 48KB`**
- `600KB`：优先考虑 **`544KB / 56KB`**

后续若要进一步提高稳健性，建议：

1. 将 `Prefill` 从单点 `694` 扩展为 `246 / 438 / 694 / 950 / 1206` 多档联合目标
2. 用真实硬件实测验证 `544/56`、`528/72`、`512/88` 这几个相邻配置
3. 若 Decode 实际平均生成长度与 `15.25` 偏差较大，再重新调整 `DecodeTokensWeight`
