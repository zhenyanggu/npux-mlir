#!/bin/bash

# ================= Configuration =================
# 定义工具名称（如果不在环境变量中，请改为绝对路径）
OPT_TOOL="onnx-mlir-opt"

# 颜色定义，方便看日志
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ================= Helper Functions =================

# 全局变量
STAGE_COUNT=0
CURRENT_INPUT=""
CURRENT_DIR=""
TILING_CONFIG=""

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
    CURRENT_DIR="$folder_name"
    
    echo ""
    echo -e "${YELLOW}==========================================${NC}"
    echo -e "${YELLOW} $STAGE_COUNT. Folder: $folder_name${NC}"
    echo -e "${YELLOW}==========================================${NC}"
    
    # 创建目录 (在当前执行目录下)
    mkdir -p "$folder_name"
}

# 运行单个 Pass 的函数
# 参数 1: 描述 (用于日志)
# 参数 2: flag 参数
# 参数 3: 输出文件名
run_pass() {
    local desc="$1"
    local flags="$2"
    local output_file="$3"
    
    # 构建完整的输出路径
    local output_path="${CURRENT_DIR}/${output_file}"
    
    echo " -> $desc"
    
    # 执行命令
    # 注意：这里 $flags 没有加引号，是为了让参数正确展开
    # 2>&1 把错误流也捕获，方便调试
    if ! $OPT_TOOL $flags "$CURRENT_INPUT" -o "$output_path"; then
        echo -e "${RED}xxx $desc fail! xxx${NC}"
        echo "Command executed: $OPT_TOOL $flags $CURRENT_INPUT -o $output_path"
        exit 1
    fi
    
    # 更新 Current Input，供下一步使用
    CURRENT_INPUT=$(realpath "$output_path")
}

# ================= Main Pipeline =================

# 1. 检查输入
if [ -z "$1" ]; then
        echo "Usage: $0 <input_model.mlir> [tiling_config.json]"
    exit 1
fi

# 初始化输入文件
CURRENT_INPUT=$(realpath "$1")
if [ ! -f "$CURRENT_INPUT" ]; then
        error_exit "Input file not found: $CURRENT_INPUT"
fi

if [ -n "$2" ]; then
        TILING_CONFIG=$(realpath "$2")
else
        TILING_CONFIG="model.json"
fi

if [ ! -f "$TILING_CONFIG" ]; then
        error_exit "Tiling config not found: $TILING_CONFIG"
fi

echo ">>> Starting NPU Compilation Pipeline"
echo ">>> Initial Input: $CURRENT_INPUT"
echo ">>> Tiling Config: $TILING_CONFIG"

# ------------------------------------------------
#  NpuPartition
# ------------------------------------------------
enter_stage "NpuPartition"

run_pass "Convert to Linalg" \
         "--convert-npu-onnx-to-linalg --npu-ops=Conv,MatMul,LayerNorm,Softmax,Gelu,Gemm,Transpose,MaxPool --npu-tiling-config=$TILING_CONFIG" \
         "ConvertONNXToLinalgNpu.mlir"

# run_pass "Modify Scf Region Encoding" \
#         "--npu-modify-scf-encoding"\
#         "ModifyScfRegionEncoding.mlir"

# run_pass "Op Merge" \
#          "--npu-clean-pack --npu-merge" \
#          "NpuMerge.mlir"

# run_pass "Region Extent" \
#          "--npu-region-extension" \
#          "NpuRegionExtension.mlir"

# run_pass "Outline" \
#          "--npu-outline" \
#          "NpuOutline.mlir"

# ------------------------------------------------
# NpuFuse
# ------------------------------------------------

# enter_stage "NpuFuse"

# run_pass "Fusing" \
#          "--npu-fuse" \
#          "NpuFuse.mlir"

# # ------------------------------------------------
# #  NpuTiling
# # ------------------------------------------------
enter_stage "NpuTiling"


run_pass "Tiling " \
         "--npu-tiling --canonicalize --npu-tiling-config=model.json" \
         "NpuTiling.mlir"

run_pass "Insert Dma " \
         "--npu-insert-dma" \
         "NpuInsertDma.mlir"

run_pass "Op Splitting " \
         "--npu-op-splitting --npu-remove-redundant-dma" \
         "NpuOpSplitting.mlir"

# # # ------------------------------------------------
# # #  NpuBufferization
# # # ------------------------------------------------
enter_stage "NpuBufferization"

run_pass "Bufferize" \
         "--convert-onnx-to-krnl --target=npu --canonicalize --convert-krnl-to-affine --npu-dps-convert --cse --canonicalize" \
         "NpuBufferization.mlir"


# # # ------------------------------------------------
# # #  NpuToLLVM
# # # ------------------------------------------------
enter_stage "NpuToLLVM"

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
        "--custom-buffer-loop-hoisting" \
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

run_pass "Lower Subview" \
         "--npu-lower-subview" \
         "NpuLowerSubview.mlir"

run_pass "Npux Compute Fuse" \
         "--npux-compute-fusion" \
         "NpuxComputeFusion.mlir"

run_pass "convert-linalg-to-loops" \
        "--convert-linalg-to-loops" \
        "convert-linalg-to-loops.mlir"
        
run_pass "Npu Memory Plan" \
        "--npu-memory-plan --npu-tiling-config=model.json" \
        "NpuMemoryPlan.mlir"

# run_pass "Npu Inline" \
#         "--npu-inline  " \
#         "NpuInline.mlir"

run_pass "Erase Memoryspace"\
         "--npu-erase-memory-space --expand-strided-metadata "\
         "EraseNpuMemorySpace.mlir"

run_pass "LLVM Lowering" \
        "--convert-krnl-to-llvm --target=npu --reconcile-unrealized-casts --canonicalize" \
        "llvm.mlir"

# ================= Final Report =================

echo ""
echo -e "${GREEN}>>> Compilation Complete!${NC}"
# 直接使用 CURRENT_INPUT，因为它保存了最后一次 run_pass 的结果
echo -e ">>> Final Output: ${GREEN}${CURRENT_INPUT}${NC}"