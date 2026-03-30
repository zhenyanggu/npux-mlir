# Decoder Custom Op Task

## 1. 目标

当前目标不是继续给 `onnx-mlir` 补 `onnx.Custom` 支持，而是把
`aicas_run/decoder_model_merged_fp16.onnx` 里的微软域自定义算子离线展开为
标准 ONNX 子图，尽量落到后续 NPU 更容易接的标准算子上。

重点方向：

- `SimplifiedLayerNormalization`
- `SkipSimplifiedLayerNormalization`
- `RotaryEmbedding`
- `MultiHeadAttention`

期望结果：

- 新导出的 decoder 模型不再包含 `onnx.Custom`
- 能被当前 `onnx-mlir` 正常 ingest / shape inference / lowering
- 后续更容易映射到 NPU 常见 primitive，如 `MatMul/Gemm`

注意：标准化后不可能只剩 `MatMul/Gemm`，通常还会保留
`Reshape/Transpose/Add/Mul/Softmax/Gather/Concat/Slice/Cast` 等辅助算子。

## 2. 当前已完成的事情

### 2.1 vision encoder 已打通

`vision_encoder_fp16.onnx` 原先不能编译，现已可以直接完整编译成功。

已经修复的编译器问题：

1. `Flatten(axis == rank)` 的 shape helper 边界错误
2. `GatherND` lowering 假设切片尾部只有 1 维，无法处理当前模型里的高维 slice

相关代码修改：

- `src/Dialect/ONNX/ONNXOps/Tensor/Flatten.cpp`
- `src/Conversion/ONNXToKrnl/Tensor/GatherND.cpp`

验证命令：

```bash
build/Release/bin/onnx-mlir aicas_run/vision_encoder_fp16.onnx -o aicas_run/tmp/vision_encoder_fp16_compiled
```

当前产物：

- `aicas_run/tmp/vision_encoder_fp16_compiled.so`
- `aicas_run/tmp/vision_encoder_fixed_compiled.so`

### 2.2 vision 修模脚本已可用

`aicas_run/scripts/fix_vision_encoder.py` 已改成只依赖 `onnx`，不再需要
`onnx_graphsurgeon`。它会把 `/vision_model/embeddings/Flatten_1`
从 `Flatten(axis=2)` 改成 `Reshape([-1, 1])`。

虽然现在原始 `vision_encoder_fp16.onnx` 已经能直接编译，但这个脚本仍然可作为
模型修补参考。

## 3. decoder 当前阻塞点

### 3.1 失败现象

当前命令：

```bash
build/Release/bin/onnx-mlir aicas_run/decoder_model_merged_fp16.onnx -o aicas_run/tmp/decoder_model_merged_fp16
```

失败日志核心内容：

```text
Implementation of shape helper for op onnx.Custom is not currently available
missing implementation for shape inference
UNREACHABLE executed at src/Interface/ShapeHelperOpInterface.hpp:124
```

### 3.2 根因不是单个 MultiHeadAttention

这个 decoder 里并不是“只含有一个 MultiHeadAttention 不支持”，而是存在大量
`onnx.Custom`。

通过 `--EmitONNXBasic` 导出的 IR 统计结果如下：

- `RotaryEmbedding`: 64 个
- `SkipSimplifiedLayerNormalization`: 64 个
- `MultiHeadAttention`: 32 个
- `SimplifiedLayerNormalization`: 1 个

生成 IR 的命令：

```bash
mkdir -p aicas_run/tmp
build/Release/bin/onnx-mlir --EmitONNXBasic \
  aicas_run/decoder_model_merged_fp16.onnx \
  -o aicas_run/tmp/decoder_model_merged_fp16_basic
```

生成文件：

- `aicas_run/tmp/decoder_model_merged_fp16_basic.onnx.mlir`

## 4. decoder 模型里的自定义算子实例

下面是第一层里的典型样例，足够说明问题结构：

- `SimplifiedLayerNormalization`
- `RotaryEmbedding`
- `MultiHeadAttention`
- `SkipSimplifiedLayerNormalization`

对应位置（`aicas_run/tmp/decoder_model_merged_fp16_basic.onnx.mlir`）：

- 行 323：`SimplifiedLayerNormalization`
- 行 338：`RotaryEmbedding`
- 行 407：`MultiHeadAttention`
- 行 409：`SkipSimplifiedLayerNormalization`

可用命令：

```bash
nl -ba aicas_run/tmp/decoder_model_merged_fp16_basic.onnx.mlir | sed -n '320,410p'
```

## 5. ONNX Runtime 为什么能跑

ONNX Runtime 不是把这些东西自动拆成标准 ONNX 再跑，而是直接支持
`com.microsoft` 域里的 contrib/custom 算子，并在 graph optimization /
partition 之后调用相应 kernel。

对当前任务的重要结论：

- ORT 能跑，不代表模型已经是标准 ONNX
- ORT 的可运行路径不能直接拿来当 NPU 编译路径
- 如果后续要走自研 NPU，离线 defuse / rewrite 成标准 ONNX 是更对的方向

## 6. 为什么当前 onnx-mlir 会卡住

仓库里虽然有 `ONNXCustomOp`，但支持非常有限：

- `CustomOp` 语义默认未知
- 没有额外 shape inference 属性时，形状推理基本无法进行
- lowering 也只是把它转成 `KrnlCallOp` 调外部函数，不是内建语义实现

相关代码：

- `src/Dialect/ONNX/AdditionalONNXOps.td`
- `src/Dialect/ONNX/ONNXOps/Additional/Custom.cpp`
- `src/Conversion/ONNXToKrnl/Additional/Custom.cpp`

所以如果继续沿“给 onnx-mlir 补 Custom 支持”这条路走，需要同时补：

- shape inference
- lowering
- runtime entry / kernel

并且要覆盖 4 类自定义算子，不只是一类 MHA。

## 7. 建议的新方向

建议切换到“离线重写模型”的方向，即：

把 `decoder_model_merged_fp16.onnx` 中的自定义算子重写成标准 ONNX 子图。

推荐顺序：

1. `SimplifiedLayerNormalization`
2. `SkipSimplifiedLayerNormalization`
3. `RotaryEmbedding`
4. `MultiHeadAttention`

这样做的原因：

- 前两类结构最简单，容易快速清掉一大批 `Custom`
- `RotaryEmbedding` 逻辑明确，通常是固定图模式展开
- `MultiHeadAttention` 最复杂，放最后做更稳

## 8. 每类算子的预期展开方向

### 8.1 SimplifiedLayerNormalization

本质接近 RMSNorm。

典型展开：

- `Pow(x, 2)` 或 `Mul(x, x)`
- `ReduceMean` over last dim
- `Add(epsilon)`
- `Sqrt` / `Rsqrt`
- `Div` 或 `Mul`
- 再乘 `weight`

### 8.2 SkipSimplifiedLayerNormalization

本质是 residual add + RMSNorm，并且当前模型里它是多输出。

需要先确认：

- 第 0 个输出是什么
- 第 3 个输出是什么
- 中间 `none` 输出是否只是占位

当前 IR 看起来后续常用的是 `#0` 和 `#3`。

### 8.3 RotaryEmbedding

大概率可展开成标准张量变换：

- position index gather
- 对 q/k 做偶奇维拆分
- 与 sin/cos 做 `Mul/Add`
- 再 `Concat/Reshape/Transpose`

这个要按当前导出模型的具体输入布局来写，不能照搬别的实现。

### 8.4 MultiHeadAttention

理论上可以展开成标准 attention：

- Q, K, V reshape/transpose
- `Q * K^T`
- scale
- add mask
- softmax
- `Softmax(QK^T) * V`
- transpose/reshape 回输出

但当前模型里它接的是已经做过 rotary 和 cache 拼接之后的 Q/K/V，
所以 rewrite 时不要重复做前面的逻辑。

## 9. 推荐实施方式

建议新开一份离线 rewrite 脚本，而不是继续改 `onnx-mlir` 前端。

推荐脚本位置：

- `aicas_run/scripts/rewrite_decoder_custom_ops.py`

推荐依赖优先级：

1. 纯 `onnx`
2. 如果图编辑太麻烦，再考虑 `onnx_graphsurgeon`

理由：

- 当前环境里默认不一定有 `onnx_graphsurgeon`
- 纯 `onnx` 更稳，复现更容易

## 10. 实施步骤建议

### 阶段 1：先把 Custom 全部枚举清楚

目标：

- 把每种 `function_name`
- 输入个数
- 输出个数
- 关键属性
- 典型 shape

都提取成表，确认只有这 4 类，没有遗漏。

### 阶段 2：先重写最简单的 LN 类

先做：

- `SimplifiedLayerNormalization`
- `SkipSimplifiedLayerNormalization`

验证：

- 导出后的模型里这两类 `onnx.Custom` 数量应为 0

### 阶段 3：再做 RotaryEmbedding

验证：

- 和原模型在同一输入下逐层或最终输出对比

### 阶段 4：最后做 MultiHeadAttention

验证：

- rewrite 后模型不再含 `MultiHeadAttention`
- 尽量保持子图仍然是标准 attention 结构

### 阶段 5：重新跑 onnx-mlir

目标：

- 新模型不再因为 `onnx.Custom` 崩掉
- 如果出现新问题，再针对标准 ONNX 图做处理

## 11. 实用命令

### 11.1 查看 decoder 的 ONNXBasic IR

```bash
build/Release/bin/onnx-mlir --EmitONNXBasic \
  aicas_run/decoder_model_merged_fp16.onnx \
  -o aicas_run/tmp/decoder_model_merged_fp16_basic
```

### 11.2 编译 vision encoder（已通过）

```bash
build/Release/bin/onnx-mlir \
  aicas_run/vision_encoder_fp16.onnx \
  -o aicas_run/tmp/vision_encoder_fp16_compiled
```

### 11.3 重新测试 decoder（当前仍会因为 Custom 失败）

```bash
build/Release/bin/onnx-mlir \
  aicas_run/decoder_model_merged_fp16.onnx \
  -o aicas_run/tmp/decoder_model_merged_fp16
```

## 12. 当前工作区里与任务相关的修改

已经改过的文件：

- `aicas_run/scripts/fix_vision_encoder.py`
- `src/Conversion/ONNXToKrnl/Tensor/GatherND.cpp`
- `src/Dialect/ONNX/ONNXOps/Tensor/Flatten.cpp`

这些改动和 decoder rewrite 任务不冲突，但要注意：

- 当前 `aicas_run/tmp/` 下有中间产物
- 工作区本身不是干净状态
- 不要误删用户已有文件

## 13. 新窗口继续做时的建议起点

建议直接从下面这几步开始：

1. 读取 `aicas_run/tmp/decoder_model_merged_fp16_basic.onnx.mlir`
2. 写一个脚本先统计全部 `onnx.Custom(function_name=...)`
3. 先实现 `SimplifiedLayerNormalization` 的标准化重写
4. 再实现 `SkipSimplifiedLayerNormalization`
5. 每完成一种就重新导出模型，并确认剩余 `onnx.Custom` 数量下降

不要一开始就直接写 `MultiHeadAttention` 重写，复杂度太高，调试成本也最高。
