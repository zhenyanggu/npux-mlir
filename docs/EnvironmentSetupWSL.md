# WSL Native Environment Setup

本项目当前推荐的主开发路径是 **WSL 原生环境 + 用户目录前缀隔离**，而不是继续依赖固定名称的 Docker 容器。

## 设计目标

- 能在当前宿主机直接初始化成功。
- 不把 `LLVM/MLIR`、`protobuf`、`absl`、交叉工具链装进 `/opt` 或 `/usr/local`。
- Python 依赖与项目专属环境变量只在当前项目会话中生效，不污染别的项目。

## 当前默认实现

- Python 隔离：`conda` 环境 `npux-mlir`
- 本地前缀：`~/.local/npux-env`
- LLVM 源码：`~/.local/npux-env/src/llvm-project`
- LLVM/MLIR build tree：`~/.local/npux-env/build/llvm-project`
- AArch64 工具链：`~/.local/npux-env/toolchains/aarch64`

版本锁定见 [`env/toolchain.lock`](/home/gugugu2404/research/npu-mlir/npux-mlir/env/toolchain.lock)。

## 一键初始化

在仓库根目录执行：

```bash
./scripts/bootstrap_env.sh
```

这个脚本会完成：

1. 检查当前宿主机版本和基础工具。
2. 创建/复用 `conda` 环境。
3. 安装 Python 依赖。
4. 在用户目录前缀下构建 `absl`、`protobuf`、`llvm-project`。
5. 安装本地 patched `third_party/onnx`。
6. 安装 AArch64 交叉工具链。
7. 配置并构建 host `build/` 与 cross `build-zcu102/`。
8. 写入 conda activate/deactivate hook。

## 激活环境

两种方式都可以：

```bash
conda activate npux-mlir
```

或：

```bash
source scripts/activate_env.sh
```

激活后会自动导出：

- `MLIR_DIR`
- `LLVM_DIR`
- `absl_DIR`
- `LLVM_SRC_ROOT`
- `CROSS_CXX`
- `TRANSLATE_TOOL`
- `LLC_TOOL`
- `RUNTIME_LIB_DIR`

退出当前项目环境：

```bash
source scripts/activate_env.sh --deactivate
```

## 常用验证命令

```bash
which onnx-mlir onnx-mlir-opt mlir-translate llc
python -c "import onnx, google.protobuf, numpy"
cmake --build build --target onnx-mlir onnx-mlir-opt
cmake --build build-zcu102 --target cruntime OMTensorUtils OMExecutionSession
```

## model_test 工作流

激活环境后直接运行：

```bash
make -C model_test -B llvm MODEL=gelu
make -C model_test -B zcu102 MODEL=gelu
```

`model_test` 里的 Python 生成脚本会默认优先使用 `CUDAExecutionProvider`，并在失败时回退到 `CPUExecutionProvider`。如果只想强制走 CPU，可以显式覆盖：

```bash
make -C model_test -B llvm MODEL=gelu MODEL_TEST_ORT_FORCE_CPU=1
```

如果要手动调整 provider 顺序，可以覆盖：

```bash
make -C model_test -B llvm MODEL=gelu MODEL_TEST_ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
```

在 WSL 中使用 GPU 时，项目激活脚本还会自动把 `/usr/lib/wsl/lib`、`torch/lib` 和 `site-packages/nvidia/*/lib` 加入 `LD_LIBRARY_PATH`，让 `onnxruntime-gpu` 与 `torch` 能复用同一套 CUDA/cuDNN 运行库。

## Docker 说明

历史文档中的 `my-npux-dev` 仍然可作为参考调试环境，但不再是当前主路径。`my-mpux-env` 不是当前仓库约定的容器名。
