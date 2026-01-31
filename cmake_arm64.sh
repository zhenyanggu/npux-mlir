#!/bin/bash

# ================= Configuration =================

# 1. 交叉编译器设置 (ZCU102 / AArch64)
CROSS_COMPILE_PREFIX="aarch64-linux-gnu"
CXX_COMPILER="/usr/bin/${CROSS_COMPILE_PREFIX}-g++"
C_COMPILER="/usr/bin/${CROSS_COMPILE_PREFIX}-gcc"

# 2. 依赖路径配置
HOST_ABSL_DIR="/usr/local/lib/cmake/absl"
HOST_MLIR_DIR="/opt/llvm-project/build/lib/cmake/mlir"
HOST_LLVM_DIR="/opt/llvm-project/build/lib/cmake/llvm"

# 3. Python 路径 (宿主机)
# 请根据实际情况调整，通常 Ubuntu 22.04 是 3.10
PY_VER="3.10"
HOST_PY_INC="/usr/include/python${PY_VER}"
# 注意：这里必须精确指向 .so 文件，不能只是目录
HOST_PY_LIB="/usr/lib/x86_64-linux-gnu/libpython${PY_VER}.so"

# 4. 构建目录
BUILD_DIR="build-zcu102"
TOOLCHAIN_FILE="zcu102_toolchain.cmake"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ================= Checks =================

echo ">>> Starting Final Fix for ZCU102 Cross-Compilation"

# 检查 1: 编译器
if [ ! -f "$CXX_COMPILER" ]; then
    echo -e "${RED}[Error] Compiler not found: $CXX_COMPILER${NC}"
    echo "Run: sudo apt install g++-aarch64-linux-gnu"
    exit 1
fi

# 检查 2: Python Header (关键!)
if [ ! -d "$HOST_PY_INC" ]; then
    echo -e "${RED}[Error] Python headers not found at $HOST_PY_INC${NC}"
    echo "You might be missing the dev package."
    echo "Run: sudo apt install python3-dev"
    exit 1
fi
echo -e " -> Found Python Headers: ${GREEN}$HOST_PY_INC${NC}"

# ================= Toolchain Generation =================

# 生成 Toolchain 文件
# 使用 BOTH 模式允许搜索宿主机
cat > $TOOLCHAIN_FILE <<EOF
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER "$C_COMPILER")
set(CMAKE_CXX_COMPILER "$CXX_COMPILER")

# 强制允许查找宿主机路径 (解决 Python 找不到的问题)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
EOF

echo -e " -> Toolchain file generated."

# ================= Build =================

# 清理旧的缓存，防止错误的 Python 路径残留
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo " -> Configuring CMake..."

# 关键修正：
# 使用 -DPython3_INCLUDE_DIR (单数) 和 -DPython3_LIBRARY (单数)
# 这告诉 CMake："不要找了，就在这里！"
CMD_CMAKE="cmake -G Ninja .. \
    -DCMAKE_TOOLCHAIN_FILE=../$TOOLCHAIN_FILE \
    -DCMAKE_BUILD_TYPE=Release \
    \
    -DONNX_MLIR_ENABLE_PYRUNTIME_LIGHT=ON \
    -DONNX_MLIR_ENABLE_JNI=OFF \
    -DONNX_MLIR_ENABLE_PYTHON_BINDING=OFF \
    \
    -Dabsl_DIR=$HOST_ABSL_DIR \
    -DMLIR_DIR=$HOST_MLIR_DIR \
    -DLLVM_DIR=$HOST_LLVM_DIR \
    \
    -DPython3_INCLUDE_DIR=$HOST_PY_INC \
    -DPython3_LIBRARY=$HOST_PY_LIB"

if ! eval $CMD_CMAKE; then
    echo -e "${RED}>>> CMake Configuration Failed!${NC}"
    echo "Check if libpython3.10.so exists in /usr/lib/x86_64-linux-gnu/"
    exit 1
fi

# ================= Compilation =================

echo " -> Compiling Runtime Libraries..."

# 只编译 Runtime，规避链接错误
TARGETS="cruntime OMTensorUtils OMExecutionSession"

if ! ninja $TARGETS; then
    echo -e "${RED}>>> Compilation Failed!${NC}"
    exit 1
fi

# ================= Report =================

LIB_DIR=$(pwd)/lib
echo ""
echo -e "${GREEN}>>> Build Complete!${NC}"
echo -e ">>> Libraries: ${GREEN}$LIB_DIR${NC}"
ls -lh $LIB_DIR/libonnx_mlir_cruntime_wrapper.* 2>/dev/null
ls -lh $LIB_DIR/libOMTensor.* 2>/dev/null