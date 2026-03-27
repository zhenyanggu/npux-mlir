# GEMM/MatMul 尺寸分析与分块建议（aicas_run）

更新时间：2026-03-26  
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
