# arm_toolchain.cmake

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# 获取环境变量中的 TOOLCHAIN 路径 (如果不习惯这种写法，也可以直接写绝对路径)
set(ARM_TOOLCHAIN_PATH "$ENV{TOOLCHAIN}")

# 指定编译器
set(CMAKE_C_COMPILER "${ARM_TOOLCHAIN_PATH}/bin/arm-none-linux-gnueabihf-gcc")
set(CMAKE_CXX_COMPILER "${ARM_TOOLCHAIN_PATH}/bin/arm-none-linux-gnueabihf-g++")

# 指定 Sysroot
set(CMAKE_SYSROOT "${ARM_TOOLCHAIN_PATH}/arm-none-linux-gnueabihf/libc")

# 查找库和头文件的行为控制
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)