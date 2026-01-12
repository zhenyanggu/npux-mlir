#!/bin/bash
set -e # 遇到错误立即停止

# 检查输入
if [ -z "$1" ]; then
    echo "Usage: $0 <input_model.mlir>"
    # 例如: ./run_npu_structured.sh ../model_onnx.onnx.mlir
    exit 1
fi

# 获取输入文件的绝对路径，防止 cd 切换目录后找不到文件
INPUT_FILE=$(realpath "$1")

echo ">>> Starting NPU Compilation Pipeline"
echo ">>> Input: $INPUT_FILE"

# ==========================================
# 1. Folder: NpuPartition
# ==========================================
echo "--- Entering NpuPartition ---"
mkdir -p NpuPartition
cd NpuPartition

echo "[1/14] Op Labeling"
onnx-mlir-opt --onnx-npu-op-label "$INPUT_FILE" -o ONNXOpLabel.mlir

echo "[2/14] Outline"
onnx-mlir-opt --onnx-npu-outline ONNXOpLabel.mlir -o NpuOutline.mlir

echo "[3/14] Convert to Linalg"
onnx-mlir-opt --convert-npu-onnx-to-linalg NpuOutline.mlir -o ConvertONNXToLinalgNpu.mlir

cd .. # 回到根目录

# ==========================================
# 2. Folder: NpuTiling
# ==========================================
echo "--- Entering NpuTiling ---"
mkdir -p NpuTiling
cd NpuTiling

echo "[4/14] Tiling Elemwise"
# 输入来自 NpuPartition 文件夹
onnx-mlir-opt --npu-tiling-elemwise ../NpuPartition/ConvertONNXToLinalgNpu.mlir -o ElemWise.mlir

cd ..

# ==========================================
# 3. Folder: NpuBufferization
# ==========================================
echo "--- Entering NpuBufferization ---"
mkdir -p NpuBufferization
cd NpuBufferization


echo "[5/14] One Shot Bufferize"
onnx-mlir-opt --npu-bufferize ../NpuTiling/ElemWise.mlir -o OneShotBufferize.mlir

echo "[6/14] Emit MLIR (Frontend)"
# 这步会在当前目录下生成 OneShotBufferize.onnx.mlir
onnx-mlir --EmitMLIR OneShotBufferize.mlir

echo "[7/14] Signature Rewrite (Post-Bufferization)"
onnx-mlir-opt --npu-signature-rewrite --canonicalize --reconcile-unrealized-casts ./OneShotBufferize.onnx.mlir -o SignatureRewrite.mlir

cd ..

# ==========================================
# 4. Folder: NpuToLLVM
# ==========================================
echo "--- Entering NpuToLLVM ---"
mkdir -p NpuToLLVM
cd NpuToLLVM



echo "[8/14] Memory Placement"
# 输入来自 NpuBufferization 文件夹
onnx-mlir-opt --npu-memory-placement ../NpuBufferization/SignatureRewrite.mlir -o NpuMemoryPlacement.mlir

echo "[9/14] SRAM Promotion"
onnx-mlir-opt --npu-sram-promotion NpuMemoryPlacement.mlir -o NpuSramPromotion.mlir

echo "[10/14] Memory Allocation"
onnx-mlir-opt --npu-memory-alloc NpuSramPromotion.mlir -o NpuAlloc.mlir

echo "[11/14] Instruction Lowering"
onnx-mlir-opt --npu-instruction-lowering NpuAlloc.mlir -o NpuInstructionLowering.mlir

echo "[12/14] Npu Inline"
onnx-mlir-opt --npu-inline ./NpuInstructionLowering.mlir -o NpuInline.mlir

echo "[13/14] convert to LLVM"
onnx-mlir-opt --convert-krnl-to-llvm ./NpuInline.mlir -o krnl_free.mlir

echo "[14/14] Finalize LLVM"
onnx-mlir-opt --npu-finalize-llvm ./krnl_free.mlir -o NpuFinalizeLLVM.mlir

onnx-mlir --EmitLLVMIR NpuFinalizeLLVM.mlir \
    -o model_armv7 \
    --mtriple=armv7-linux-gnueabihf \
    --mcpu=cortex-a9
mlir-translate --mlir-to-llvmir model_armv7.onnx.mlir -o model.ll

llc ./model.ll -mtriple=armv7-linux-gnueabihf -mcpu=cortex-a9 -mattr=+neon,+vfp3 \
 -relocation-model=pic -O=2 -filetype=obj \
 -o model.o
cp model.o ..
cd ..

echo ">>> Compilation Complete!"
echo ">>> Final Output: NpuToLLVM/NpuFinalizeLLVM.mlir"