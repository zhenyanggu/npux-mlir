# NPU Board JSON 实验技术分析

## 1. 实验背景与目标

本文档基于 `model_test/board_json` 中的板级性能结果，对三组实验配置进行系统比较：
- `baseline`: 启用 cost model 分块与 DMA/layer fusion 优化的完整方案。
- `no_costmodel`: 禁用 cost model，对 Conv/GEMM 不做 cost-model 驱动的分块优化。
- `no_dma_elim`: 禁用 DMA elimination / 层融合（按实验命名与用户说明）。

分析目标有两部分：
- 量化 `cost model` 对 Conv/GEMM 分块策略的影响，重点观察 NPU 总时延、DMA 次数、DMA 时间和等待时间。
- 量化 `DMA elimination / 层融合` 对执行行为的影响，重点观察 `mvin_calls + mvout_calls` 的变化，并结合逐层数据解释 DMA 模式的变化。

本次纳入对比的模型包括 `resnet50`、`bert_base`、`vgg16` 和 `mnist`。目录中还存在 `no_relu_fusion`，但不属于本次用户指定的主比较对象，因此本文不展开。

## 2. 数据来源与指标定义

数据直接来自以下 JSON 文件：
- `profile_report.json`: 提供 `summary` 和逐层执行时间统计，包括 `total_ns`、`dma_in_ns`、`dma_out_ns`、`compute_ns`、`wait_irq_ns`、`mvin_calls`、`mvout_calls`、`compute_calls`。
- `profile_manifest.json`: 提供逐层 `macs`、`input_bytes`、`output_bytes`、`weight_bytes`、`fused_ops` 等静态信息。

本文使用的关键指标定义如下：
- `总时延`: `summary.total_ns`。
- `NPU 时延`: `summary.npu_total_ns`。
- `CPU 时延`: `summary.cpu_total_ns`。
- `DMA 次数`: 所有 NPU 层的 `mvin_calls + mvout_calls` 之和。
- `DMA 时间`: 所有 NPU 层的 `dma_in_ns + dma_out_ns` 之和。
- `等待时间`: 所有 NPU 层的 `wait_irq_ns` 之和。
- `多算子融合层数`: `fused_ops` 长度大于 1 的层数，用于观察图层面融合的显式痕迹。

说明：文中的“DMA elimination / 层融合”命名采用实验目录约定；但最终分析以 JSON 中的客观结果为准，不强行把结果解释成和命名完全一致。若结果与直觉不一致，本文会单独标注为“现象”与“推测原因”。

## 3. 总体结果总表

| 模型 | 配置 | 总时延(ms) | NPU时延(ms) | CPU时延(ms) | NPU占比 | DMA次数 | mvin | mvout | DMA时间(ms) | Compute时间(ms) | Wait IRQ(ms) | NPU层数 | 多算子融合层数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet50 | baseline | 3612.129 | 677.383 | 2934.746 | 0.1875 | 42,699 | 37,688 | 5,011 | 182.847 | 270.452 | 312.332 | 70 | 0 |
| resnet50 | no_costmodel | 4600.004 | 1661.251 | 2938.753 | 0.3611 | 298,651 | 117,330 | 181,321 | 902.722 | 272.047 | 720.077 | 70 | 0 |
| resnet50 | no_dma_elim | 3686.451 | 753.163 | 2933.289 | 0.2043 | 58,250 | 37,911 | 20,339 | 248.769 | 267.113 | 354.327 | 70 | 0 |
| bert_base | baseline | 7170.017 | 2092.913 | 5077.105 | 0.2919 | 57,228 | 56,560 | 668 | 264.819 | 1595.718 | 1754.013 | 196 | 0 |
| bert_base | no_costmodel | 8860.222 | 3781.043 | 5079.179 | 0.4267 | 342,690 | 324,828 | 17,862 | 1650.241 | 1561.164 | 2750.102 | 196 | 0 |
| bert_base | no_dma_elim | 6976.319 | 1902.493 | 5073.826 | 0.2727 | 2,436 | 1,732 | 704 | 130.833 | 1595.606 | 1689.949 | 196 | 0 |
| vgg16 | baseline | 4205.184 | 4198.954 | 6.230 | 0.9985 | 22,339 | 6,502 | 15,837 | 350.729 | 431.349 | 704.152 | 21 | 15 |
| vgg16 | no_costmodel | 5232.025 | 5225.845 | 6.180 | 0.9988 | 285,966 | 66,227 | 219,739 | 1235.526 | 358.456 | 1239.971 | 21 | 15 |
| vgg16 | no_dma_elim | 4167.888 | 4161.714 | 6.174 | 0.9985 | 11,926 | 6,629 | 5,297 | 342.489 | 431.216 | 707.984 | 21 | 15 |
| mnist | baseline | 2.122 | 0.546 | 1.577 | 0.2571 | 9 | 6 | 3 | 0.060 | 0.368 | 0.395 | 3 | 0 |
| mnist | no_costmodel | 2.993 | 1.408 | 1.585 | 0.4704 | 239 | 44 | 195 | 0.657 | 0.350 | 0.694 | 3 | 0 |
| mnist | no_dma_elim | 4.436 | 0.621 | 3.815 | 0.1399 | 17 | 10 | 7 | 0.097 | 0.387 | 0.422 | 3 | 0 |

从总表可以先得到三个整体结论：
- `no_costmodel` 在四个模型上都显著增加了 DMA 次数和 NPU 时延，说明 cost model 的核心收益首先来自“减少分块带来的数据搬运碎片化”。
- `resnet50` 在 `no_dma_elim` 下 DMA 次数明显上升，符合“关闭 DMA/层融合会导致更多数据搬运”的预期。
- `bert_base` 和 `vgg16` 在 `no_dma_elim` 下反而出现 DMA 次数下降，这与目录命名的直觉预期相反，说明当前实现中 DMA elimination/fusion pass 对不同模型的收益并不单调。这个现象非常值得在论文中如实记录。

## 4. baseline 对 no_costmodel 的比较：cost model 对分块的影响

| 模型 | 总时延变化 | NPU时延变化 | DMA次数变化 | DMA时间变化 | Compute时间变化 | Wait IRQ变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet50 | 987.875 (+27.35%) | 983.867 (+145.25%) | 255,952 (+599.43%) | 719.875 (+393.70%) | 1.596 (+0.59%) | 407.746 (+130.55%) |
| bert_base | 1690.205 (+23.57%) | 1688.130 (+80.66%) | 285,462 (+498.82%) | 1385.422 (+523.16%) | -34.554 (-2.17%) | 996.088 (+56.79%) |
| vgg16 | 1026.841 (+24.42%) | 1026.891 (+24.46%) | 263,627 (+1180.12%) | 884.796 (+252.27%) | -72.894 (-16.90%) | 535.818 (+76.09%) |
| mnist | 0.871 (+41.03%) | 0.862 (+158.06%) | 230 (+2555.56%) | 0.596 (+986.99%) | -0.018 (-4.93%) | 0.299 (+75.70%) |

### 4.1 关键观察

- `resnet50`: DMA 次数从 42,699 增加到 298,651，增幅 +599.43%；NPU 时延从 677.383 ms 增加到 1661.251 ms，增幅 +145.25%。
- `bert_base`: DMA 次数从 57,228 增加到 342,690，增幅 +498.82%；NPU 时延从 2092.913 ms 增加到 3781.043 ms，增幅 +80.66%。
- `vgg16`: DMA 次数从 22,339 增加到 285,966，增幅 +1180.12%；NPU 时延从 4198.954 ms 增加到 5225.845 ms，增幅 +24.46%。
- `mnist`: DMA 次数从 9 增加到 239，增幅 +2555.56%；NPU 时延从 0.546 ms 增加到 1.408 ms，增幅 +158.06%。

这组数据说明：
- `compute_ms` 在大多数模型上变化远小于 `DMA时间` 与 `Wait IRQ`，说明禁用 cost model 后，主要恶化点不是纯计算量变化，而是 tile 组织方式导致的访存和同步开销上升。
- `bert_base` 和 `vgg16` 的 DMA 次数增幅尤其大，分别从 `57,228 -> 342,690`、`22,339 -> 285,966`。这说明对于 GEMM 密集或深层 Conv 堆叠网络，错误的分块策略会把中间结果切得非常碎，从而带来海量搬运。
- `mnist` 绝对规模最小，但 DMA 次数也从 `9 -> 239`，说明即便是小模型，缺少合理分块策略仍会显著放大数据搬运次数；只是因为总计算量小，最终绝对时延增量有限。

### 4.2 对论文论点的支持

- 若论文需要证明“cost model 的主要价值在于分块质量而不是改变量化计算本身”，这组结果是非常直接的证据。
- 论证链条可以写成：`禁用 cost model -> 分块更碎片化 -> mvin/mvout 激增 -> DMA 时间与等待时间显著上升 -> NPU 总时延恶化`。
- 尤其在 `bert_base` 上，`compute_ms` 从 `1595.718 ms` 仅下降到 `1561.164 ms`，但 `DMA时间` 却从 `264.819 ms` 激增到 `1650.241 ms`，说明性能退化主因并不是算术计算，而是数据搬运路径失控。

## 5. baseline 对 no_dma_elim 的比较：重点分析 DMA 次数

| 模型 | 总时延变化 | NPU时延变化 | DMA次数变化 | mvin变化 | mvout变化 | DMA时间变化 | Wait IRQ变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet50 | 74.322 (+2.06%) | 75.779 (+11.19%) | 15,551 (+36.42%) | 223 | 15,328 | 65.923 (+36.05%) | 41.996 (+13.45%) |
| bert_base | -193.698 (-2.70%) | -190.420 (-9.10%) | -54,792 (-95.74%) | -54,828 | 36 | -133.986 (-50.60%) | -64.064 (-3.65%) |
| vgg16 | -37.295 (-0.89%) | -37.240 (-0.89%) | -10,413 (-46.61%) | 127 | -10,540 | -8.240 (-2.35%) | 3.831 (+0.54%) |
| mnist | 2.313 (+109.00%) | 0.075 (+13.78%) | 8 (+88.89%) | 4 | 4 | 0.037 (+61.00%) | 0.027 (+6.82%) |

### 5.1 总体观察

- `resnet50`: 符合预期。关闭 DMA elimination/层融合后，DMA 次数从 `42,699` 增加到 `58,250`，其中主要是 `mvout` 从 `5,011` 增长到 `20,339`，说明更多中间结果被写回。
- `bert_base`: 与预期相反。`no_dma_elim` 的 DMA 次数从 `57,228` 下降到 `2,436`，NPU 时延也从 `2092.913 ms` 降到 `1902.493 ms`。
- `vgg16`: 同样与预期相反。DMA 次数从 `22,339` 降到 `11,926`，总时延略降。
- `mnist`: 变化量很小，绝对值只有十几次 DMA，统计上更接近小模型噪声样本。

因此，这组实验不能简单写成“DMA elimination 一定减少 DMA 次数并提升性能”，更准确的表述应当是：
- 当前实现下，`DMA elimination / layer fusion` 对不同网络结构的收益具有明显模型相关性。
- 对残差加法较多、需要频繁写回中间张量的网络（如 `resnet50`），baseline 的收益非常明确。
- 对注意力/GEMM 主导或某些 Conv+Relu 组合，当前 baseline 方案可能引入了额外的数据搬运切分，导致结果并不总是优于 `no_dma_elim`。

### 5.2 resnet50 的逐层 DMA 变化

DMA 次数增加最多的层：

| 层名 | 算子类型 | baseline DMA | no_dma_elim DMA | 增量 | DMA时间增量(ms) | 总时延增量(ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `resnetv17_stage4__plus0` | Add | 4,153 | 6,144 | 1,991 | 5.078 | 7.158 |
| `resnetv17_stage4__plus2` | Add | 4,153 | 6,144 | 1,991 | 4.975 | 6.919 |
| `resnetv17_stage4__plus1` | Add | 4,153 | 6,144 | 1,991 | 4.945 | 6.854 |
| `resnetv17_stage3__plus4` | Add | 2,105 | 3,072 | 967 | 2.697 | 3.600 |
| `resnetv17_stage3__plus3` | Add | 2,105 | 3,072 | 967 | 2.706 | 3.617 |
| `resnetv17_stage3__plus0` | Add | 2,105 | 3,072 | 967 | 2.739 | 3.635 |
| `resnetv17_stage3__plus1` | Add | 2,105 | 3,072 | 967 | 2.702 | 3.599 |
| `resnetv17_stage3__plus5` | Add | 2,105 | 3,072 | 967 | 2.750 | 3.644 |

DMA 次数减少最多的层：

| 层名 | 算子类型 | baseline DMA | no_dma_elim DMA | 变化量 | DMA时间变化(ms) | 总时延变化(ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `resnetv17_stage2_conv3_fwd` | Conv | 2,384 | 928 | -1,456 | -2.704 | -4.278 |
| `resnetv17_conv0_fwd` | Conv | 926 | 190 | -736 | -0.960 | -1.742 |
| `resnetv17_stage1_conv5_fwd` | Conv | 270 | 24 | -246 | -0.389 | -0.547 |
| `resnetv17_stage1_conv1_fwd` | Conv | 270 | 24 | -246 | -0.379 | -0.573 |
| `resnetv17_stage1_conv8_fwd` | Conv | 270 | 24 | -246 | -0.390 | -0.590 |

解释：`resnet50` 中增量最大的层集中在残差分支相加节点（如 `stage3__plus*`、`stage4__plus*`）。这些 `Add` 层在 `no_dma_elim` 下出现大幅 `mvout` 增长，说明中间结果更多地被显式写回片外或至少写回可见缓冲区，符合“关闭融合后中间张量落地次数增加”的机制预期。

### 5.3 bert_base 的逐层 DMA 变化

DMA 次数增加最多的层：

| 层名 | 算子类型 | baseline DMA | no_dma_elim DMA | 增量 | DMA时间增量(ms) | 总时延增量(ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 无 | - | - | - | - | - | - |

DMA 次数减少最多的层：

| 层名 | 算子类型 | baseline DMA | no_dma_elim DMA | 变化量 | DMA时间变化(ms) | 总时延变化(ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `/bert/encoder/layer.4/attention/self/MatMul_output_0_QuantizeLinear` | QLinearMatMul | 3,084 | 36 | -3,048 | -7.702 | -10.948 |
| `/bert/encoder/layer.0/attention/self/MatMul_output_0_QuantizeLinear` | QLinearMatMul | 3,084 | 36 | -3,048 | -7.556 | -10.669 |
| `/bert/encoder/layer.1/attention/self/MatMul_output_0_QuantizeLinear` | QLinearMatMul | 3,084 | 36 | -3,048 | -7.541 | -10.683 |
| `/bert/encoder/layer.2/attention/self/MatMul_output_0_QuantizeLinear` | QLinearMatMul | 3,084 | 36 | -3,048 | -7.654 | -10.867 |
| `/bert/encoder/layer.9/attention/self/MatMul_output_0_QuantizeLinear` | QLinearMatMul | 3,084 | 36 | -3,048 | -7.735 | -11.031 |
| `/bert/encoder/layer.6/attention/self/MatMul_output_0_QuantizeLinear` | QLinearMatMul | 3,084 | 36 | -3,048 | -7.760 | -11.054 |
| `/bert/encoder/layer.3/attention/self/MatMul_output_0_QuantizeLinear` | QLinearMatMul | 3,084 | 36 | -3,048 | -7.700 | -10.914 |
| `/bert/encoder/layer.11/attention/self/MatMul_output_0_QuantizeLinear` | QLinearMatMul | 3,084 | 36 | -3,048 | -7.816 | -11.162 |

解释：`bert_base` 中变化最大的层主要是注意力内部的 `QLinearMatMul` 与后续量化节点。典型层如 `attention/self/MatMul_output_0_QuantizeLinear`，DMA 次数从 `3084` 降到 `36`。这说明当前 baseline 方案在这类层上触发了更细粒度的数据切分或中间结果搬运，反而比 `no_dma_elim` 更“碎”。

### 5.4 vgg16 的逐层 DMA 变化

DMA 次数增加最多的层：

| 层名 | 算子类型 | baseline DMA | no_dma_elim DMA | 增量 | DMA时间增量(ms) | 总时延增量(ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vgg0_conv7_fwd` | Conv | 41 | 586 | 545 | 3.305 | 3.034 |
| `vgg0_conv9_fwd` | Conv | 712 | 1,248 | 536 | 1.971 | 2.221 |
| `vgg0_conv8_fwd` | Conv | 712 | 1,248 | 536 | 2.221 | 1.096 |
| `vgg0_pool2_fwd` | MaxPoolSingleOut | 258 | 512 | 254 | 0.716 | 1.683 |
| `vgg0_pool1_fwd` | MaxPoolSingleOut | 132 | 256 | 124 | 0.368 | -0.097 |
| `vgg0_pool0_fwd` | MaxPoolSingleOut | 72 | 128 | 56 | 0.208 | -1.987 |
| `vgg0_conv11_fwd` | Conv | 60 | 76 | 16 | 0.329 | 0.156 |
| `vgg0_conv10_fwd` | Conv | 60 | 76 | 16 | 0.332 | 0.045 |

DMA 次数减少最多的层：

| 层名 | 算子类型 | baseline DMA | no_dma_elim DMA | 变化量 | DMA时间变化(ms) | 总时延变化(ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vgg0_conv0_fwd` | Conv | 3,334 | 758 | -2,576 | -2.912 | -10.666 |
| `vgg0_conv1_fwd` | Conv | 3,843 | 1,267 | -2,576 | -2.941 | -5.632 |
| `vgg0_conv5_fwd` | Conv | 2,336 | 864 | -1,472 | -2.746 | -5.718 |
| `vgg0_conv3_fwd` | Conv | 2,308 | 836 | -1,472 | -1.988 | -3.646 |
| `vgg0_conv2_fwd` | Conv | 2,120 | 648 | -1,472 | -1.984 | -7.104 |
| `vgg0_conv4_fwd` | Conv | 2,106 | 634 | -1,472 | -2.727 | -6.584 |
| `vgg0_conv6_fwd` | Conv | 2,336 | 864 | -1,472 | -2.729 | -4.348 |

解释：`vgg16` 的前几层 `Conv+Relu` 在 baseline 下 DMA 次数反而高于 `no_dma_elim`。同时两组配置的 `fused_ops` 仍然都显示为 `['Conv', 'Relu']`，说明差异不一定体现在图层面的显式融合标签，而可能体现在更底层的 tile 调度、DMA 插入与回写策略上。

### 5.5 mnist 的逐层 DMA 变化

DMA 次数增加最多的层：

| 层名 | 算子类型 | baseline DMA | no_dma_elim DMA | 增量 | DMA时间增量(ms) | 总时延增量(ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Convolution28` | Conv | 3 | 7 | 4 | 0.019 | 0.028 |
| `Convolution110` | Conv | 3 | 7 | 4 | 0.018 | 0.048 |

DMA 次数减少最多的层：

| 层名 | 算子类型 | baseline DMA | no_dma_elim DMA | 变化量 | DMA时间变化(ms) | 总时延变化(ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 无 | - | - | - | - | - | - |

解释：`mnist` 的绝对 DMA 次数很小，单层增减不足以支撑复杂机制判断，更适合作为轻量 sanity check，而不是主要论证样本。

## 6. 结果解读与论文写作建议

### 6.1 可以直接写进论文的稳定结论

- `cost model` 的价值非常明确，而且跨模型稳定：禁用后会显著放大 DMA 次数、DMA 时间和等待时间，进而拉高 NPU 时延。
- `resnet50` 上的 `DMA elimination / 层融合` 收益同样明确：它显著压低了 `mvout` 次数，减少了残差相加节点的中间结果写回。
- 从性能分解看，`DMA时间` 和 `wait_irq` 是比 `compute_ms` 更敏感的指标，因此论文中应把“数据搬运与同步开销”作为优化收益的主叙事，而不是只写总时延。

### 6.2 必须如实记录的异常/反常现象

- `bert_base` 与 `vgg16` 的 `no_dma_elim` 结果优于或接近 baseline，这与实验命名对应的直觉预期不一致。
- 这类现象不应在论文中回避。更合理的写法是：当前 DMA elimination / layer fusion 实现对不同网络的收益不一致，尤其在 attention/GEMM-heavy 与部分 Conv+Relu 场景下，可能因为更细碎的 tile/quantize 边界处理而引入额外 DMA。
- 由于 `profile_report.json` 只提供执行层面的统计，不能直接证明 pass 内部做了哪一种变换；因此对根因的表述应使用“推测”“可能”“表明需要进一步验证”等措辞。

### 6.3 建议在正文和附录中的组织方式

- 正文主结论建议重点展示 `resnet50` 与 `bert_base`。前者适合证明 DMA elimination 在残差网络上的收益，后者适合证明 cost model 对 GEMM/attention 分块的重要性。
- `vgg16` 适合放在补充实验，强调优化策略并非对所有网络都是单调收益。
- `mnist` 可作为小规模 sanity check，证明趋势在小模型上仍可观察，但不宜承载主要论点。

## 7. 可引用的摘要结论

若需要在论文中用一段话概括本节实验，可以直接改写为：

> 在板级实测中，cost model 驱动的 Conv/GEMM 分块对 NPU 性能具有决定性影响。与 baseline 相比，禁用 cost model 会在四个模型上统一导致 DMA 次数大幅增加，其中 BERT Base 从 57,228 次增至 342,690 次，VGG16 从 22,339 次增至 285,966 次，表明劣化的分块策略会显著放大数据搬运与同步开销。另一方面，DMA elimination / 层融合优化在不同网络上的收益具有模型相关性：在 ResNet50 上，baseline 将 DMA 次数从 58,250 次降至 42,699 次，并显著减少残差相加节点的中间结果回写；但在 BERT Base 和 VGG16 上，当前实现并未表现出统一优势，说明该优化仍存在进一步按网络结构细化的空间。

## 8. 结论

- 若论文需要一个“强结论”，优先突出 `cost model` 的稳定收益。
- 若论文需要展示系统优化的复杂性，则应保留 `no_dma_elim` 的反常结果，因为它真实反映了优化 pass 的模型依赖性。
- 从本批数据看，未来最值得继续优化的方向不是单纯增加算子支持，而是进一步降低注意力/GEMM 路径中的 DMA 碎片化和不必要回写。
