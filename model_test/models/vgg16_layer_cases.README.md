# VGG16 Layer Cases

These testcases are generated from unique VGG16 layer signatures (op + shape),
filtered by hardware-supported ops: Conv / MaxPool / Gemm.
They are disabled by default to avoid affecting `make all`.

Enable one case:
- `rm -f models/<case_name>/.disabled`
- `make <case_name>`

Or run directly by model name even if disabled:
- `make run_one MODEL=<case_name>`
