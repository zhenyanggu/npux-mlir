## 语法规范
1. 本项目使用的MLIR库在~/toolchain/llvm-project/mlir下，每次使用相关的语法务必查询该库确认使用是否正确。

## 项目架构
1. 本项目在ONNX-MLIR项目上修改，使用ONNX-MLIR导入.onnx文件到使用onnx dialect的.mlir，并且编译模型中fall-back到CPU处理的部分。
2. model_test/scripts/onnx_to_llvm.sh是项目测试脚本中用来将.onnx文件lowering到llvm方言 mlir的脚本，该脚本反映了项目的流水线。输出文件命名规则是{Pass定义文件名}.cpp定义的pass生成的文件命名为{Pass定义文件名}.mlir。每次修改流水线需要同步修改这个文件。
3. model_test/runtime是本项目使用的runtime，项目使用CPU调用CAPI控制NPU的形式，项目最后lowering得到的方言所有的NPU操作体现为llvm.call调用runtime里定义的CAPI
4. src/Conversion定义了本项目的pass。
5. src/Dialect/Npux定义的和runtime对应的中间方言，是为了最后lowering到llvm 方言的CAPI调用更方便
6. src/Dialect/ONNX 定义了ONNX方言，是使用ONNX-MLIR导入后得到的.mlir文件使用的方言，需要对ONNX方言进行lowering的时候先查询这里对onnx op的定义，特别需要查询这两个文件：src/Dialect/ONNX/AdditionalONNXOps.td，src/Dialect/ONNX/ONNXOps.td.inc。
7. pass的定义结构是这样的：在src/Pass声明对应函数便于调用，src/Conversion里定义该函数，src/Tools/onnx-mlir-opt/RegisterPasses.cpp调用该函数。

## 代码生成规范
1. 生成的代码需要在最开始写好注释，参考如下：
```cpp
//=============================================================================
// /src/Conversion/NpuPartition/ConvertONNXToLinalgNpu.cpp
// this file implements the NPU partitioning pass that labels
// ONNX operations for NPU execution based on a conversion registry.
//=============================================================================
```
2. 代码里定义的函数和数据结构需要写好注释。

## Debug规范
1. 不允许猜测原因修改代码，如果觉得可能是某个地方出错，请修改代码把对应的内容print出来，问题解决之后再把这段print删掉。