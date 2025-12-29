//=============================================================================
// /src/Conversion/NpuPartition/ONNXOpConversion.cpp
// this file implements the conversion of ONNX operations to linalg operations
// for NPU partitioning.
//=============================================================================

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/NPUConversionRegistry.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

using namespace mlir;

namespace npux {

struct GeluToLinalg : public OpRewritePattern<ONNXGeluOp> {
  using OpRewritePattern<ONNXGeluOp>::OpRewritePattern;

  // === 必须实现的静态检查函数 ===
  static bool isHardwareSupported(ONNXGeluOp op) {
    // 你的硬件检查逻辑
    return true;
  }

  // === 转换逻辑 ===
  LogicalResult matchAndRewrite(
      ONNXGeluOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getX();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    Type elementType = inputType.getElementType();

    // 1. 准备 Output (Empty Tensor)
    Value emptyTensor = rewriter.create<bufferization::AllocTensorOp>(
            loc, inputType, /*dynamicSizes=*/ValueRange{});

    // 2. Maps & Iterators (标准 Elementwise 配置)
    SmallVector<AffineMap, 2> indexingMaps = {
        rewriter.getMultiDimIdentityMap(inputType.getRank()),
        rewriter.getMultiDimIdentityMap(inputType.getRank())};
    SmallVector<utils::IteratorType> iteratorTypes(
        inputType.getRank(), utils::IteratorType::parallel);

    // 3. 创建 linalg.generic
    auto linalgOp = rewriter.create<linalg::GenericOp>(loc,
        /*resultTypes=*/inputType,
        /*inputs=*/input,
        /*outputs=*/emptyTensor, // 下面我们会换掉这个
        indexingMaps, iteratorTypes,
        /*bodyBuilder=*/[&](OpBuilder &b, Location loc, ValueRange args) {
          // 【关键修改】不要只 Yield args[0]
          // 加一点虚假的计算，防止被 Canonicalizer 优化成 Copy。
          // 比如：y = x + x (虽然数学不对，但后端只看 library_call，所以没关系)
          // 这样编译器就认为这是一个“加法操作”，不敢随便删了。
          Value dummyResult = b.create<arith::AddFOp>(loc, args[0], args[0]);
          b.create<linalg::YieldOp>(loc, dummyResult);
        });

    // === 4. 贴标签 (传家宝) ===

    // 标签 A: 身份识别 (给 CodeGen 看)
    // 告诉后端：虽然我 body 里写的是 yield x，但我其实是 Gelu！
    linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_gelu"));

    linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));

    // 标签 B: 硬件限制 (给 Tiling Pass 看)
    // 假设 Gelu 单元一次只能算 32x32，你就贴个 32
    // 这个值你可以从你的 NPU 硬件配置表里查
    // 如果是 Add 算子，可能这里就贴 64
    rewriter.replaceOp(op, linalgOp.getResults());

    // 也可以直接在这里把 tile size 以 attribute 形式挂上去，
    // 比如叫 "npux.tile_size"，这样通用的 tiling pass 读这个值就行了
    // 这里假设 Gelu 只能处理 rank 维度的切分建议
    linalgOp->setAttr("npux.max_tile_size",
        rewriter.getI32ArrayAttr({32, 32, 32, 32})); // 示例

    return success();
  }
};

// 注册！
// 这里也不会产生歧义，这就是一个 Registration
static npux::NPUOpRegistration<GeluToLinalg, ONNXGeluOp> registerGelu;

void registerNpuOpConversions() {};

} // namespace npux