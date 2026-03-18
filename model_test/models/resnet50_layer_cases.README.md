# ResNet50 Layer Cases

These testcases are generated from ResNet50 ONNX-MLIR and deduplicated by
(op + shape + attributes). Standalone Relu cases are removed.
When a Conv is directly followed by Relu, it is generated as one `ConvRelu` case.
They are disabled by default to avoid affecting `make all`.

Layer op counts:
- Add: 4
- Conv: 7
- ConvRelu: 13
- Flatten: 1
- Gemm: 1
- MaxPool: 1
- ReduceMean: 1

Enable one case:
- `rm -f models/<case_name>/.disabled`
- `make <case_name>`

Or run directly by model name even if disabled:
- `make run_one MODEL=<case_name>`
