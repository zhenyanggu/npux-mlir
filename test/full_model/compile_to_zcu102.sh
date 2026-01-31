#!/bin/bash

# ================= Configuration =================
# 定义工具名称
TRANSLATE_TOOL="mlir-translate" # 建议用完整名称，防止混淆
LLC_TOOL="llc"

# ZCU102 交叉编译器 (AArch64)
CROSS_CXX="aarch64-linux-gnu-g++"

# [关键路径配置]
# 1. 库文件路径 (根据你的要求修改)
RUNTIME_LIB_DIR="/workspace/build-zcu102/Release/lib"

# 2. 头文件路径 
# 注意：头文件通常位于源码目录的 include 文件夹中，而不是 build 目录
# 假设你的源码在 /workspace/onnx-mlir，如果不同请修改此处
ONNX_MLIR_SRC="/workspace/onnx-mlir"
INC_FLAGS="-I/workspace/include/onnx-mlir/Runtime \
           -I/workspace/include"
# 3. 链接参数
# -L: 指定库查找路径
LIB_FLAGS="-L${RUNTIME_LIB_DIR}"

# 4. 指定需要链接的库名称
LINK_LIBS="-lOMExecutionSession -lOMTensorUtils -lcruntime -lpthread"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ================= Helper Functions =================

# 全局变量
STAGE_COUNT=4 
CURRENT_INPUT=""
CURRENT_DIR=""
ROOT_DIR=$(pwd)

# 报错并退出的函数
error_exit() {
    echo -e "${RED}>>> [ERROR] $1 Failed!${NC}"
    exit 1
}

# 进入一个新的阶段文件夹
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

# 运行通用命令的函数
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

# 1. 检查输入 (已修改：支持默认路径)
DEFAULT_INPUT="./NpuToLLVM/llvm.mlir"

if [ -z "$1" ]; then
    # 如果没有提供参数，检查默认文件是否存在
    if [ -f "$DEFAULT_INPUT" ]; then
        echo -e "${YELLOW}>>> No input file specified. Using default: ${DEFAULT_INPUT}${NC}"
        INPUT_FILE="$DEFAULT_INPUT"
    else
        echo "Usage: $0 [input_model.mlir]"
        echo -e "${RED}Error: Input file not provided and default '${DEFAULT_INPUT}' not found.${NC}"
        exit 1
    fi
else
    # 使用用户提供的参数
    INPUT_FILE="$1"
fi

CURRENT_INPUT=$(realpath "$INPUT_FILE")

echo ">>> Starting Cross-Compilation Pipeline for ZCU102"
echo ">>> Initial Input: $CURRENT_INPUT"
echo ">>> Lib Path: $RUNTIME_LIB_DIR"

# ------------------------------------------------
# Stage 5: Code Generation (LLVM IR -> Object)
# ------------------------------------------------
enter_stage "Codegen"

# 1. MLIR -> LLVM IR (.ll)
LL_FILE="${CURRENT_DIR}/model.ll"
CMD_TRANSLATE="$TRANSLATE_TOOL --mlir-to-llvmir \"$CURRENT_INPUT\" -o \"$LL_FILE\""

run_command "Translate to LLVM IR" "$CMD_TRANSLATE"

# 2. LLVM IR -> Object File (.o)
# ZCU102 (ZynqMP) 参数: Cortex-A53, AArch64
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

# 检查 C++ 源码
CPP_SRC="${ROOT_DIR}/main.cpp ${ROOT_DIR}/npu_runtime.cpp"

if [ ! -f "${ROOT_DIR}/main.cpp" ]; then
    echo -e "${RED}Error: main.cpp not found in ${ROOT_DIR}${NC}"
    exit 1
fi

# 输出可执行文件名称
TARGET_BIN="${ROOT_DIR}/gelu_exec_zcu102"

echo " -> Linking against runtime..."
echo " -> Lib Flags: $LIB_FLAGS"
echo " -> Libraries: $LINK_LIBS"

# 构建 G++ 命令
# 这里的顺序很重要：先源码，再 .o，再库文件
CMD_LINK="$CROSS_CXX $CPP_SRC \"$OBJ_FILE\" \
    -o \"$TARGET_BIN\" \
    $INC_FLAGS \
    $LIB_FLAGS \
    -static \
    -O2 \
    -Wl,--allow-multiple-definition \
    $LINK_LIBS"

run_command "Link C++ Executable" "$CMD_LINK"

# ================= Final Report =================

echo ""
echo -e "${GREEN}>>> Cross-Compilation Complete!${NC}"
echo -e ">>> Final Binary: ${GREEN}${TARGET_BIN}${NC}"
echo -e ">>> Target Board: ${YELLOW}Xilinx ZCU102 (AArch64)${NC}"
echo -e ">>> Deploy:"
echo -e "    scp ${TARGET_BIN} root@<board_ip>:~/"