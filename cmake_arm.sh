# 1. 设置环境变量 (保持不变)
export TOOLCHAIN=$(ls -d $HOME/opt/arm-gnu-toolchain-*-x86_64-arm-none-linux-gnueabihf | head -n1)
# 2. 进入构建目录
mkdir -p build-arm
cd build-arm

# 3. 运行 CMake (增加了 Host 的依赖路径)
cmake -G Ninja .. \
    -DCMAKE_TOOLCHAIN_FILE=../arm_toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNX_MLIR_ENABLE_PYRUNTIME_LIGHT=ON \
    -DONNX_MLIR_ENABLE_JNI=OFF \
    -DONNX_MLIR_ENABLE_PYTHON_BINDING=OFF \
    \
    -Dabsl_DIR=/usr/local/lib/cmake/absl \
    -DMLIR_DIR=/opt/llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=/opt/llvm-project/build/lib/cmake/llvm \
    \
    -DPython3_INCLUDE_DIRS=/usr/include/python3.10 \
    -DPython3_LIBRARIES=/usr/lib/x86_64-linux-gnu/libpython3.10.so