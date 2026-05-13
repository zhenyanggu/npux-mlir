# Paper Model Tests

This folder is the paper-only test entry. It intentionally includes only four full-network models:

- `bert_base`
- `resnet50`
- `vgg16`
- `mnist`

Layer and block cases such as `bert_base_layer_*`, `resnet50_layer_*`, `resnet50_block_*`, and `vgg16_layer_*` are not part of this test set.

## Build

```bash
cd model_test
make paper_models
```

For ablation experiments, pass an experiment name to isolate both intermediate build files and final outputs:

```bash
cd model_test
make paper_models EXPERIMENT_NAME=baseline
make paper_models EXPERIMENT_NAME=no_costmodel GEMM_TILING_STRATEGY=legacy
make paper_models EXPERIMENT_NAME=no_dma_elim NPU_REMOVE_REDUNDANT_DMA=0
```

The default baseline options are:

```text
GEMM_TILING_STRATEGY=costmodel
CONV_TILING_STRATEGY=costmodel
NPU_REMOVE_REDUNDANT_DMA=1
```

`CONV_TILING_STRATEGY` defaults to the same value as `GEMM_TILING_STRATEGY`, so
`GEMM_TILING_STRATEGY=legacy` disables both GEMM and Conv costmodel tiling unless
`CONV_TILING_STRATEGY` is set explicitly.

The outputs will be written under:

```text
model_test/output/baseline/NPU/
model_test/output/no_costmodel/NPU/
model_test/output/no_dma_elim/NPU/
```

The corresponding build stamps are also isolated under `model_test/build/<EXPERIMENT_NAME>/`, so rebuilding the same model with different compiler options will not reuse the previous experiment's stamps.

Equivalent script entry:

```bash
cd model_test/paper_tests
./run_paper_models.sh build
```

The script accepts the same make-style variables:

```bash
./run_paper_models.sh build EXPERIMENT_NAME=baseline
./run_paper_models.sh build EXPERIMENT_NAME=no_costmodel GEMM_TILING_STRATEGY=legacy
./run_paper_models.sh build EXPERIMENT_NAME=no_dma_elim NPU_REMOVE_REDUNDANT_DMA=0
```

## CPU Baseline

```bash
cd model_test/paper_tests
./run_paper_models.sh cpu
```

## Run Generated NPU Executables

```bash
cd model_test/paper_tests
./run_paper_models.sh run
./run_paper_models.sh run EXPERIMENT_NAME=baseline
./run_paper_models.sh run EXPERIMENT_NAME=no_dma_elim
```
