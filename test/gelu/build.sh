$TOOLCHAIN/bin/arm-none-linux-gnueabihf-g++ \
  ./main.cpp ./npu_runtime.cpp ./model.o \
  -o gelu \
  -I /workspace/include/onnx-mlir/Runtime \
  -I /workspace/include \
  -L /workspace/build-arm/Release/lib \
  -static \
  -Wl,--allow-multiple-definition \
  -lOMExecutionSession \
  -lOMTensorUtils \
  -lcruntime \
  -lpthread