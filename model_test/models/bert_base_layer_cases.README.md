# BERT Base Layer Cases

These testcases are generated from unique BERT base NPU-relevant layer signatures
(op + shape + attributes), then deduplicated to keep only representative cases.
They focus on the encoder-side NPU-accelerated parts:
- Add
- Gemm
- LayerNormalization
- MatMul
- Softmax
- Transpose

Small scalar helper ops introduced by activation decomposition are intentionally
not expanded into separate cases. The goal here is layer-level isolation for
the main NPU-relevant BERT subgraphs, not every internal arithmetic fragment.

The full-model testcase is intentionally not used as the correctness metric here.
Each layer case is standalone and compares numerical output against ORT golden data.

All `bert_base_layer_*` cases are disabled by default to avoid affecting `make all`.

Enable one case:
- `rm -f models/<case_name>/.disabled`
- `make <case_name>`

Or run directly by model name even if disabled:
- `make run_one MODEL=<case_name>`

Run the whole set:
- `make bert_base_layers_all`
