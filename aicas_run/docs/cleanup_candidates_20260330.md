# aicas_run 清理候选清单（2026-03-30）

以下清单是“建议删除/归档候选”，**未自动删除**，用于人工确认。

## 1. 可直接删除（纯中间产物或缓存）

1. Python 缓存目录
   - `aicas_run/__pycache__/`（约 68K）
   - `aicas_run/scripts/__pycache__/`（约 248K）
2. Windows 下载附带元数据
   - `aicas_run/models/*:Zone.Identifier`（6 个，总计 150B）
3. 编译中间文件（可再生）
   - `aicas_run/tmp/*.tmp`
   - `aicas_run/tmp/*.onnx.mlir`

## 2. 建议按需删除（体积大，但可能用于回归对比）

`aicas_run/tmp/` 当前约 `6.2G`，包括：

- `decoder_model_merged_fp16_basic.onnx.mlir`（约 1.4G）
- `decoder_model_merged_fp16_rewritten.onnx`（约 693M）
- `decoder_model_merged_fp16_rewritten.so`（约 693M）
- `decoder_model_merged_fp32_rewritten.constants.bin`（约 1.4G）
- `vision_encoder_*` 一组 `.so/.onnx.mlir/.tmp`
- `tmp/compile_check/`（约 971M，主要是旧 `.so` 备份）

如果当前只保留“最新 FP32 编译结果”，通常只需保留：

- `aicas_run/tmp/decoder_model_merged_fp32_rewritten.so`
- `aicas_run/tmp/decoder_model_merged_fp32_rewritten.constants.bin`（若运行时需要外置常量）

## 3. 根目录模型文件整理建议（先确认再动）

当前根目录有以下 FP16 模型：

- `aicas_run/decoder_model_merged_fp16.onnx`（约 692M）
- `aicas_run/embed_tokens_fp16.onnx`（约 91M）
- `aicas_run/vision_encoder_fp16.onnx`（约 188M）
- `aicas_run/vision_encoder_fixed.onnx`（约 188M）

其中两份与 `aicas_run/models/` 完全重复（哈希一致）：

- `embed_tokens_fp16.onnx`（root 与 models 相同）
- `vision_encoder_fixed.onnx`（root 与 models 相同）

建议：

1. 将仍在 root 独有的 FP16 模型先迁移到 `aicas_run/models/`（如 `decoder_model_merged_fp16.onnx`, `vision_encoder_fp16.onnx`）。
2. 统一验证脚本后，再删除 root 的重复副本。

## 4. 文档与脚本状态建议

1. `aicas_run/docs/decoder_custom_op_rewrite_compile_guide.md`：建议保留（最新操作手册）。
2. `aicas_run/decoder_custom_op_task.md`：建议保留但标记为“历史排障记录”（可后续移到 `docs/archive/`）。
