#!/bin/bash

# ================= Configuration =================

# --- MLIR 工具 ---
# 定义工具名称（如果不在环境变量中，请改为绝对路径）
ONNX_MLIR_TOOL="onnx-mlir"  # [新增]
OPT_TOOL="onnx-mlir-opt"
TRANSLATE_TOOL="mlir-translate"
LLC_TOOL="llc"

# --- 交叉编译工具链设置 (请根据你的实际环境修改) ---
# 假设 TOOLCHAIN 已经在环境变量中，如果没有，请取消下面注释并修改路径
# TOOLCHAIN="/usr" 
if [ -z "$TOOLCHAIN" ]; then
    # 默认回退到一个常见的路径，或者提示用户
    TOOLCHAIN="/root/opt/arm-gnu-toolchain-13.2.Rel1-x86_64-arm-none-linux-gnueabihf" 
fi
CROSS_CXX="${TOOLCHAIN}/bin/arm-none-linux-gnueabihf-g++"

# --- 编译参数 ---
# 头文件路径
INC_FLAGS="-I /workspace/include/onnx-mlir/Runtime \
           -I /workspace/include"
# 库文件路径
LIB_FLAGS="-L /workspace/build-arm/Release/lib"
# 链接库
LINK_LIBS="-lOMExecutionSession -lOMTensorUtils -lcruntime -lpthread"

# --- 其他 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ================= Helper Functions =================

# 全局变量
STAGE_COUNT=0
CURRENT_INPUT=""
CURRENT_DIR=""
ROOT_DIR=$(pwd) # 记录脚本启动时的根目录

# 报错并退出的函数
error_exit() {
    echo -e "${RED}>>> [ERROR] $1 Failed!${NC}"
    exit 1
}

# 进入一个新的阶段文件夹
enter_stage() {
    local folder_name="$1"
    
    # 序号自增
    STAGE_COUNT=$((STAGE_COUNT + 1))
    
    echo ""
    echo -e "${YELLOW}==========================================${NC}"
    echo -e "${YELLOW} $STAGE_COUNT. Folder: $folder_name${NC}"
    echo -e "${YELLOW}==========================================${NC}"
    
    # 创建目录 (在根目录下创建)
    mkdir -p "$folder_name"
    CURRENT_DIR="$folder_name"
}

# 运行 MLIR Pass 的函数 (onnx-mlir-opt)
run_pass() {
    local desc="$1"
    local flags="$2"
    local output_file="$3"
    
    local output_path="${CURRENT_DIR}/${output_file}"
    
    echo " -> $desc"
    
    # 2>&1 把错误流也捕获
    if ! $OPT_TOOL $flags "$CURRENT_INPUT" -o "$output_path"; then
        echo -e "${RED}xxx $desc fail! xxx${NC}"
        echo "Command executed: $OPT_TOOL $flags $CURRENT_INPUT -o $output_path"
        exit 1
    fi
    
    # 更新 Current Input
    CURRENT_INPUT=$(realpath "$output_path")
}

# 运行通用命令的函数
# 用于 mlir-translate, llc, g++, onnx-mlir 等
run_command() {
    local desc="$1"
    local cmd_str="$2"
    
    echo " -> $desc"
    # echo "    [Cmd]: $cmd_str" # 调试时可以打开
    
    # 使用 eval 执行复杂命令字符串
    if ! eval "$cmd_str"; then
        echo -e "${RED}xxx $desc fail! xxx${NC}"
        echo "Command executed: $cmd_str"
        exit 1
    fi
}

# ================= Main Pipeline =================

# 1. 检查输入
if [ -z "$1" ]; then
    echo "Usage: $0 <input_model.mlir OR input_model.onnx>"
    exit 1
fi

RAW_INPUT=$(realpath "$1")
echo ">>> Starting NPU Compilation Pipeline"
echo ">>> Initial Input: $RAW_INPUT"

# [新增] 2. 判断文件类型并预处理
if [[ "$RAW_INPUT" == *.onnx ]]; then
    # --- Stage 0: Import ONNX ---
    enter_stage "ImportONNX"
    
    echo ">>> Detected ONNX file. Converting to MLIR..."
    
    # 执行 onnx-mlir --EmitONNXIR
    # 注意：onnx-mlir 默认会在当前执行目录下生成 output_file.onnx.mlir
    CMD_IMPORT="$ONNX_MLIR_TOOL --EmitONNXIR \"$RAW_INPUT\""
    run_command "Emit ONNX IR" "$CMD_IMPORT"
    
    # 计算生成的文件名：通常是 原文件名.mlir
    # 例如：input.onnx -> input.onnx.mlir
    BASE_NAME=$(basename "$RAW_INPUT")
    GENERATED_MLIR="${BASE_NAME}.mlir"
    
    # 为了保持文件夹整洁，将生成的文件移动到 ImportONNX 文件夹中
    if [ -f "$GENERATED_MLIR" ]; then
        mv "$GENERATED_MLIR" "${CURRENT_DIR}/"
        CURRENT_INPUT=$(realpath "${CURRENT_DIR}/${GENERATED_MLIR}")
        echo " -> Generated MLIR: $CURRENT_INPUT"
    else
        echo -e "${RED}Error: Expected generated file $GENERATED_MLIR not found!${NC}"
        exit 1
    fi

else
    # 如果不是 .onnx，假设是 .mlir，直接赋值
    CURRENT_INPUT="$RAW_INPUT"
fi


# ------------------------------------------------
# Stage 1: NpuPartition
# ------------------------------------------------
enter_stage "NpuPartition"

run_pass "Convert to Linalg" \
         "--convert-npu-onnx-to-linalg --npu-ops=Softmax,LayerNorm,Gelu" \
         "ConvertONNXToLinalgNpu.mlir"

run_pass "Op Merge" \
         "--npu-merge" \
         "NpuMerge.mlir"

run_pass "Region Extent" \
         "--npu-region-extension" \
         "NpuRegionExtension.mlir"

run_pass "Outline" \
         "--npu-outline" \
         "NpuOutline.mlir"


# ------------------------------------------------
# Stage 2: NpuTiling
# ------------------------------------------------
enter_stage "NpuTiling"

# 针对 2维 Gelu 使用 [1,2] 切分
run_pass "Tiling" \
         "--npu-tiling --gelu-tile-size=[1,8]" \
         "NpuTiling.mlir"

# ------------------------------------------------
# Stage 3: NpuBufferization
# ------------------------------------------------
enter_stage "NpuBufferization"

run_pass "Bufferize" \
         "--convert-onnx-to-krnl --target=npu --canonicalize --convert-krnl-to-affine --npu-dps-convert --cse --canonicalize" \
         "NpuBufferization.mlir"

# ------------------------------------------------
# Stage 4: NpuToLLVM
# ------------------------------------------------
enter_stage "NpuToLLVM"

run_pass "Npu Sram Promotion" \
        "--npu-sram-promotion" \
        "NpuSramPromotion.mlir"

run_pass "convert-vector-to-scf" \
        "--convert-vector-to-scf" \
        "convert-vector-to-scf.mlir"

run_pass "lower-affine" \
        "--lower-affine" \
        "lower-affine.mlir"

run_pass "lower-krnl-region" \
        "--lower-krnl-region" \
        "lower-krnl-region.mlir"

run_pass "buffer-loop-hoisting" \
        "--buffer-loop-hoisting" \
        "buffer-loop-hoisting.mlir"


run_pass "buffer-dealloc-test" \
        "--buffer-dealloc-test " \
        "buffer-dealloc-test.mlir"

run_pass "optimize-allocation-liveness" \
        "--optimize-allocation-liveness" \
        "optimize-allocation-liveness.mlir"

run_pass "convert-bufferization-to-memref" \
        "--convert-bufferization-to-memref" \
        "convert-bufferization-to-memref.mlir"

run_pass "fold-memref-alias-ops" \
        "--fold-memref-alias-ops" \
        "fold-memref-alias-ops.mlir"      

run_pass "Npux Conversion" \
         "--convert-linalg-to-npux --canonicalize" \
         "ConvertLinalgToNpux.mlir"

run_pass "Npu Memory Plan" \
        "--npu-memory-plan " \
        "NpuMemoryPlan.mlir"

run_pass "Npu Inline" \
        "--npu-inline --expand-strided-metadata" \
        "NpuInline.mlir"

run_pass "LLVM Lowering" \
        "--convert-krnl-to-llvm --target=npu --reconcile-unrealized-casts --canonicalize" \
        "llvm.mlir"

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
    -mtriple=armv7-linux-gnueabihf \
    -mcpu=cortex-a9 \
    -mattr=+neon,+vfp3 \
    -relocation-model=pic \
    -O=2 \
    -filetype=obj \
    -o \"$OBJ_FILE\""

run_command "Compile to Object (.o)" "$CMD_LLC"

# ------------------------------------------------
# Stage 6: Linking (C++ -> Executable)
# ------------------------------------------------
enter_stage "Linking"

# 检查 C++ 源码是否存在 (假设在脚本运行的根目录)
CPP_SRC="${ROOT_DIR}/main.cpp ${ROOT_DIR}/npu_runtime.cpp"
if [ ! -f "${ROOT_DIR}/main.cpp" ]; then
    echo -e "${RED}Error: main.cpp not found in ${ROOT_DIR}${NC}"
    exit 1
fi

# 输出可执行文件名称
TARGET_BIN="${ROOT_DIR}/gelu_exec" # 最终生成的可执行文件放在根目录，方便运行

# 构建 G++ 命令
# 注意：这里引用了 Stage 5 生成的 .o 文件
CMD_LINK="$CROSS_CXX $CPP_SRC \"$OBJ_FILE\" \
    -o \"$TARGET_BIN\" \
    $INC_FLAGS \
    $LIB_FLAGS \
    -static \
    -Wl,--allow-multiple-definition \
    $LINK_LIBS"

run_command "Link C++ Executable" "$CMD_LINK"

# ================= Final Report =================

echo ""
echo -e "${GREEN}>>> Compilation Complete!${NC}"
echo -e ">>> Final Binary: ${GREEN}${TARGET_BIN}${NC}"
echo -e ">>> To run (on board): ${YELLOW}./gelu_exec${NC}"