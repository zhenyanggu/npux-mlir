#!/bin/bash

set -euo pipefail

# ================= Configuration =================
# 目录定位
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_TEST_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${MODEL_TEST_DIR}/.." && pwd)
ROOT_DIR=$(pwd)

# 定义工具名称（可通过环境变量覆盖）
TRANSLATE_TOOL="${TRANSLATE_TOOL:-mlir-translate}"
LLC_TOOL="${LLC_TOOL:-llc}"

# ZCU102 交叉编译器 (AArch64，可通过环境变量覆盖)
CROSS_CXX="${CROSS_CXX:-aarch64-linux-gnu-g++}"

# [LLVM 源码路径 - 关键]
# 指向 llvm-project 根目录，可通过环境变量覆盖
LLVM_SRC_ROOT="${LLVM_SRC_ROOT:-/opt/llvm-project}"

# [关键路径配置]
# 1. 库文件路径 (ONNX-MLIR 的运行时库)
RUNTIME_LIB_DIR="${RUNTIME_LIB_DIR:-${PROJECT_ROOT}/build-zcu102/Release/lib}"

# 2. NPU Runtime 源码目录
RUNTIME_SRC_DIR="${RUNTIME_SRC_DIR:-${MODEL_TEST_DIR}/runtime}"

# 3. 头文件路径
INC_FLAGS="-I${PROJECT_ROOT}/include/onnx-mlir/Runtime \
           -I${PROJECT_ROOT}/include \
           -I${RUNTIME_SRC_DIR} \
           -I${LLVM_SRC_ROOT}/llvm/include \
           -I${LLVM_SRC_ROOT}/mlir/include \
           -I${LLVM_SRC_ROOT}/build/include"

# 4. 链接参数
LIB_FLAGS="-L${RUNTIME_LIB_DIR}"

# 5. 指定需要链接的库名称
# 注意：移除了 -lmlir_c_runner_utils，因为我们改为静态编译源码
LINK_LIBS="-lOMExecutionSession -lOMTensorUtils -lcruntime -lpthread"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ================= Helper Functions =================

STAGE_COUNT=4
CURRENT_INPUT=""
CURRENT_DIR=""

error_exit() {
    echo -e "${RED}>>> [ERROR] $1 Failed!${NC}"
    exit 1
}

enter_stage() {
    local folder_name="$1"
    STAGE_COUNT=$((STAGE_COUNT + 1))
    CURRENT_DIR="${ROOT_DIR}/${folder_name}"
    
    echo ""
    echo -e "${YELLOW}==========================================${NC}"
    echo -e "${YELLOW} $STAGE_COUNT. Folder: $folder_name${NC}"
    echo -e "${YELLOW}==========================================${NC}"
    
    mkdir -p "$CURRENT_DIR"
}

run_command() {
    local desc="$1"
    local cmd="$2"
    
    echo " -> $desc"
    if ! eval "$cmd"; then
        echo -e "${RED}xxx $desc fail! xxx${NC}"
        echo "Command executed: $cmd"
        exit 1
    fi
}

# ================= Main Pipeline =================

# 1. 检查输入
DEFAULT_INPUT="./NpuToLLVM/llvm.mlir"

if [ -z "${1:-}" ]; then
    if [ -f "$DEFAULT_INPUT" ]; then
        echo -e "${YELLOW}>>> No input file specified. Using default: ${DEFAULT_INPUT}${NC}"
        INPUT_FILE="$DEFAULT_INPUT"
    else
        echo "Usage: $0 [input_model.mlir] [output_binary_name]"
        exit 1
    fi
else
    INPUT_FILE="$1"
fi

OUTPUT_BIN_NAME="${2:-}"
if [ -z "$OUTPUT_BIN_NAME" ]; then
    OUTPUT_BIN_NAME="$(basename "$ROOT_DIR")_zcu102"
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: input file not found: ${INPUT_FILE}${NC}"
    exit 1
fi

CURRENT_INPUT=$(realpath "$INPUT_FILE")

echo ">>> Starting Cross-Compilation Pipeline for ZCU102"
echo ">>> Initial Input: $CURRENT_INPUT"
echo ">>> Output Binary Name: $OUTPUT_BIN_NAME"

if [ ! -d "$RUNTIME_LIB_DIR" ]; then
    echo -e "${RED}Error: runtime lib dir not found: ${RUNTIME_LIB_DIR}${NC}"
    exit 1
fi

if [ ! -d "${LLVM_SRC_ROOT}/mlir/lib/ExecutionEngine" ]; then
    echo -e "${RED}Error: invalid LLVM_SRC_ROOT: ${LLVM_SRC_ROOT}${NC}"
    exit 1
fi

# ------------------------------------------------
# Stage 5: Code Generation (LLVM IR -> Object)
# ------------------------------------------------
enter_stage "Codegen"

# 1. MLIR -> LLVM IR (.ll)
LL_FILE="${CURRENT_DIR}/model.ll"
CMD_TRANSLATE="$TRANSLATE_TOOL --mlir-to-llvmir \"$CURRENT_INPUT\" -o \"$LL_FILE\""
run_command "Translate to LLVM IR" "$CMD_TRANSLATE"

# 2. LLVM IR -> Object File (.o)
OBJ_FILE="${CURRENT_DIR}/model.o"
CMD_LLC="$LLC_TOOL \"$LL_FILE\" \
    -mtriple=aarch64-linux-gnu \
    -mcpu=cortex-a53 \
    -mattr=+neon \
    -relocation-model=pic \
    -O=2 \
    -filetype=obj \
    -o \"$OBJ_FILE\""
run_command "Compile to Object (.o)" "$CMD_LLC"

# ------------------------------------------------
# Stage 6: Linking (C++ -> Executable)
# ------------------------------------------------
enter_stage "Linking"

# [核心修改] 定位 MLIR 的运行时源码
MLIR_RUNNER_SRC="${LLVM_SRC_ROOT}/mlir/lib/ExecutionEngine/CRunnerUtils.cpp"

# 二次确认：有些 LLVM 版本是在 RunnerUtils.cpp 中
if [ ! -f "$MLIR_RUNNER_SRC" ]; then
    echo " -> CRunnerUtils.cpp not found, trying RunnerUtils.cpp..."
    MLIR_RUNNER_SRC="${LLVM_SRC_ROOT}/mlir/lib/ExecutionEngine/RunnerUtils.cpp"
fi

if [ ! -f "$MLIR_RUNNER_SRC" ]; then
    echo -e "${RED}Error: Could not find CRunnerUtils.cpp or RunnerUtils.cpp in ${LLVM_SRC_ROOT}/mlir/lib/ExecutionEngine${NC}"
    exit 1
fi

echo " -> Using MLIR Runtime Source: $MLIR_RUNNER_SRC"

# 检查项目源码
if [ ! -f "${ROOT_DIR}/main.cpp" ]; then
    echo -e "${RED}Error: main.cpp not found in ${ROOT_DIR}${NC}"
    exit 1
fi

NPU_RUNTIME_SRC="${ROOT_DIR}/npu_runtime.cpp"
if [ ! -f "$NPU_RUNTIME_SRC" ]; then
    NPU_RUNTIME_SRC="${RUNTIME_SRC_DIR}/npu_runtime.cpp"
fi

if [ ! -f "$NPU_RUNTIME_SRC" ]; then
    echo -e "${RED}Error: npu_runtime.cpp not found in ${ROOT_DIR} or ${RUNTIME_SRC_DIR}${NC}"
    exit 1
fi

# 输出文件
if [[ "$OUTPUT_BIN_NAME" = /* ]]; then
    TARGET_BIN="$OUTPUT_BIN_NAME"
else
    TARGET_BIN="${ROOT_DIR}/${OUTPUT_BIN_NAME}"
fi
mkdir -p "$(dirname "$TARGET_BIN")"

echo " -> Linking against runtime (Static)..."

# 构建 G++ 命令
# 将 MLIR_RUNNER_SRC 直接加入编译列表
CMD_LINK="$CROSS_CXX \"${ROOT_DIR}/main.cpp\" \"$NPU_RUNTIME_SRC\" \"$MLIR_RUNNER_SRC\" \"$OBJ_FILE\" \
    -o \"$TARGET_BIN\" \
    $INC_FLAGS \
    $LIB_FLAGS \
    -static \
    -O2 \
    -std=c++17 \
    -Wl,--allow-multiple-definition \
    $LINK_LIBS"

run_command "Link C++ Executable" "$CMD_LINK"

# 将编译期 profiling manifest 放到最终输出目录，便于运行期 report 直接补齐 layer_name/op_type
PROFILE_MANIFEST_SRC="${ROOT_DIR}/NpuPartition/profile_manifest.json"
PROFILE_MANIFEST_DST="$(dirname "$TARGET_BIN")/profile_manifest.json"
if [[ -f "$PROFILE_MANIFEST_SRC" ]]; then
    run_command "Copy Profile Manifest" "cp -f \"$PROFILE_MANIFEST_SRC\" \"$PROFILE_MANIFEST_DST\""
fi

# ================= Final Report =================

echo ""
echo -e "${GREEN}>>> Cross-Compilation Complete!${NC}"
echo -e ">>> Final Binary: ${GREEN}${TARGET_BIN}${NC}"
echo -e ">>> Target Board: ${YELLOW}Xilinx ZCU102 (AArch64)${NC}"
echo -e ">>> Deploy:"
echo -e "    scp ${TARGET_BIN} root@<board_ip>:~/"
