#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_TEST_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${MODEL_TEST_DIR}/.." && pwd)

# 颜色定义，方便看日志
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ================= Helper Functions =================

resolve_tool() {
    local candidate=""
    for candidate in "$@"; do
        [[ -z "$candidate" ]] && continue
        if [[ "$candidate" == */* ]]; then
            if [[ -x "$candidate" ]]; then
                realpath "$candidate"
                return 0
            fi
        elif command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

# 全局变量
STAGE_COUNT=0
CURRENT_INPUT=""
CURRENT_DIR=""
TILING_CONFIG=""
PIPELINE_MODE="${PIPELINE_MODE:-${3:-npu}}"
PARTITION_STAGE=""
BUFFER_STAGE=""
LLVM_STAGE=""
PIPELINE_LABEL=""
NPU_OPS=""

# 报错并退出的函数
error_exit() {
    echo -e "${RED}>>> [ERROR] $1 Failed!${NC}"
    exit 1
}

OPT_TOOL=$(resolve_tool \
    "${OPT_TOOL:-}" \
    "${PROJECT_ROOT}/build/Release/bin/onnx-mlir-opt" \
    "${PROJECT_ROOT}/build/Debug/bin/onnx-mlir-opt" \
    "${PROJECT_ROOT}/install/bin/onnx-mlir-opt" \
    "onnx-mlir-opt") || error_exit "onnx-mlir-opt not found"

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
	        echo "Usage: $0 <input_model.mlir> [tiling_config.json] [pipeline_mode]"
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
        TILING_CONFIG="tile_config.json"
fi

if [ ! -f "$TILING_CONFIG" ]; then
	        error_exit "Tiling config not found: $TILING_CONFIG"
fi

PIPELINE_MODE="${PIPELINE_MODE,,}"
case "$PIPELINE_MODE" in
    npu)
        PARTITION_STAGE="NpuPartition"
        BUFFER_STAGE="NpuBufferization"
        LLVM_STAGE="NpuToLLVM"
        PIPELINE_LABEL="NPU"
        NPU_OPS="${NPU_OPS:-Conv,Add,MatMul,LayerNorm,Softmax,Gelu,Gemm,Transpose,MaxPool}"
        ;;
    cpu)
        PARTITION_STAGE="CpuPartition"
        BUFFER_STAGE="CpuBufferization"
        LLVM_STAGE="CpuToLLVM"
        PIPELINE_LABEL="CPU Baseline"
        NPU_OPS="${NPU_OPS:-None}"
        ;;
    *)
        error_exit "Unsupported pipeline mode: $PIPELINE_MODE (expected npu or cpu)"
        ;;
esac

echo ">>> Starting ${PIPELINE_LABEL} Compilation Pipeline"
echo ">>> Initial Input: $CURRENT_INPUT"
echo ">>> Tiling Config: $TILING_CONFIG"
echo ">>> Pipeline Mode: $PIPELINE_MODE"

# ------------------------------------------------
#  Partition + Profiling
# ------------------------------------------------
enter_stage "$PARTITION_STAGE"
PROFILE_MANIFEST_PATH="$(pwd)/${CURRENT_DIR}/profile_manifest.json"

run_pass "Convert to Linalg" \
         "--convert-npu-onnx-to-linalg --npu-ops=$NPU_OPS --npu-tiling-config=$TILING_CONFIG" \
         "ConvertONNXToLinalgNpu.mlir"

run_pass "Annotate Profile" \
         "--npu-profile-annotate=npu-profile-manifest=$PROFILE_MANIFEST_PATH" \
         "NpuProfileAnnotate.mlir"

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

if [[ "$PIPELINE_MODE" == "npu" ]]; then
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
             "--npu-tiling --canonicalize --npu-tiling-config=$TILING_CONFIG" \
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
    enter_stage "$BUFFER_STAGE"

    run_pass "Bufferize" \
             "--convert-onnx-to-krnl --target=npu --canonicalize --convert-krnl-to-affine --npu-dps-convert --cse --canonicalize" \
             "NpuBufferization.mlir"

    # # # ------------------------------------------------
    # # #  NpuToLLVM
    # # # ------------------------------------------------
    enter_stage "$LLVM_STAGE"

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
            "--npu-memory-plan --npu-tiling-config=$TILING_CONFIG" \
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
else
    # ------------------------------------------------
    #  CpuBufferization
    # ------------------------------------------------
    enter_stage "$BUFFER_STAGE"

    run_pass "Lower to Krnl/Affine" \
             "--convert-onnx-to-krnl --target=none --canonicalize --convert-krnl-to-affine --cse --canonicalize" \
             "CpuBufferization.mlir"

    # ------------------------------------------------
    #  CpuToLLVM
    # ------------------------------------------------
    enter_stage "$LLVM_STAGE"

    run_pass "cse" \
            "--cse" \
            "cse.mlir"

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

    run_pass "LLVM Lowering" \
            "--convert-krnl-to-llvm --target=none --reconcile-unrealized-casts --canonicalize" \
            "llvm.mlir"
fi

# ================= Final Report =================

echo ""
echo -e "${GREEN}>>> Compilation Complete!${NC}"
# 直接使用 CURRENT_INPUT，因为它保存了最后一次 run_pass 的结果
echo -e ">>> Final Output: ${GREEN}${CURRENT_INPUT}${NC}"
