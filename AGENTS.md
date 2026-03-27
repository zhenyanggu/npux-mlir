# Repository Guidelines

## Project Structure & Module Organization
`npux-mlir` is a CMake-based compiler project built on MLIR/LLVM.
- `src/`: compiler, runtime, tools (for example `src/Tools/onnx-mlir-opt`).
- `include/`: public headers.
- `test/`: lit tests, backend tests, numerical/unit/perf tests, and NPU pipeline tests (for example `test/mlir`, `test/backend`, `test/numerical`, `test/npu_test.sh`).
- `docs/`: build, workflow, and testing documentation.
- `scripts/` and `env/`: environment bootstrap/activation for WSL-native development.
- `model_test/`: model-level workflows for host (`llvm`) and cross (`zcu102`) targets.

## Build, Test, and Development Commands
Typical local flow:
```bash
./scripts/bootstrap_env.sh
conda activate npux-mlir
source scripts/activate_env.sh
cmake --build build --target onnx-mlir onnx-mlir-opt
```
Useful test commands:
```bash
cmake --build build --target check-onnx-lit
cmake --build build --target check-onnx-backend
cmake --build build --target check-onnx-backend-dynamic
cmake --build build --target check-onnx-backend-constant
cmake --build build --target check-onnx-numerical
```
NPU/model workflows:
```bash
conda activate npux-mlir
source scripts/activate_env.sh
bash test/npu_test.sh <input.mlir>
make -C model_test -B llvm MODEL=gelu
make -C model_test -B zcu102 MODEL=gelu
```

## Coding Style & Naming Conventions
- C/C++ style follows LLVM conventions; `.clang-format` is based on LLVM.
- Python formatting uses `black`.
- Prefer C++ casts (`static_cast<...>`) over C-style casts.
- Keep naming consistent with nearby code and MLIR pass naming (`--npu-*`, `--convert-*`).
- Add new tests next to the relevant feature area (`test/mlir/...`, `test/backend/...`, etc.).

## Testing Guidelines
- Before running tests, always activate the project environment:
```bash
conda activate npux-mlir
source scripts/activate_env.sh
make -C model_test -B llvm MODEL=gelu
```
- For compiler pass changes, add/update lit + FileCheck tests in `test/mlir/**`.
- For runtime/op correctness, run backend and numerical targets.
- For NPU pipeline changes, run `test/npu_test.sh` or the relevant `test/*/npu_test.sh` flow.
- Prefer narrow repro runs during development, then run full affected suites before PR.

## Agent-Specific Instructions
- Default response language: Simplified Chinese (`简体中文`), unless the user explicitly asks for another language.

## Commit & Pull Request Guidelines
- Commit messages in this repo are short, action-oriented, and often Chinese (for example `修复...`, `添加...`); keep subject lines concise and specific.
- Sign commits with DCO: `git commit -s`.
- PRs should include: problem statement, scope, key design notes, and exact validation commands/results.
- Link related issues, and include logs/artifacts when behavior changes are hard to inspect from code alone.
