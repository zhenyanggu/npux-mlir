统一单算子测试方案

目标：
- 所有单算子 case 使用同一套输入规模基准，避免不同模型 shape 混入比较。
- 基准来自当前 `softmax` case 的外部输入规模 `1x128x768`。

shape 约定：
- 3D 算子直接使用 `1x128x768`
  - `unified_gelu`
  - `unified_gemm`
  - `unified_layernorm`
  - `unified_matmul`
  - `unified_matrix_add`
  - `unified_softmax`
  - `unified_transpose`
- 4D 空间算子使用同元素规模派生 shape `1x64x32x48`
  - `unified_conv`
  - `unified_maxpool`
  - `unified_relu`

说明：
- `unified_matmul` 按论文表项生成真正的 `QLinearMatMul`。
- 其余算子延续当前 `model_test/models/*` 的 QDQ 风格和输入输出文件命名。
- 所有 case 固定随机种子 `2026`，便于复现。
