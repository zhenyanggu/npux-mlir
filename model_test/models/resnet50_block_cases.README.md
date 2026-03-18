# ResNet50 Residual Block Cases

These testcases cover bottleneck residual blocks (projection + identity variants).
They are disabled by default to avoid affecting `make all`.

Residual block cases:
- resnet50_block_01_stage1_proj
- resnet50_block_02_stage1_identity
- resnet50_block_03_stage2_proj
- resnet50_block_04_stage2_identity
- resnet50_block_05_stage3_proj
- resnet50_block_06_stage3_identity
- resnet50_block_07_stage4_proj
- resnet50_block_08_stage4_identity

Enable one case:
- `rm -f models/<case_name>/.disabled`
- `make <case_name>`

Or run directly by model name even if disabled:
- `make run_one MODEL=<case_name>`
