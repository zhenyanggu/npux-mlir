#include "src/Conversion/NpuTiling/Strategy/Dispatcher.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "src/Conversion/NpuTiling/Strategy/Conv.hpp"
#include "src/Conversion/NpuTiling/Strategy/ElemWise.hpp"
#include "src/Conversion/NpuTiling/Strategy/Gemm.hpp"
#include "src/Conversion/NpuTiling/Strategy/LayerNorm.hpp"
#include "src/Conversion/NpuTiling/Strategy/Layout.hpp"
#include "src/Conversion/NpuTiling/Strategy/MaxPool.hpp"
#include "src/Conversion/NpuTiling/Strategy/Softmax.hpp"

using namespace mlir;

namespace npux {

LogicalResult tileSeedOp(
    const FusionCursor &cursor,
    PatternRewriter &rewriter) {
  auto seed = cursor.seed;
  auto root = cursor.tail;
  if (!seed || !root)
    return failure();

  auto libCall = seed->getAttrOfType<StringAttr>("library_call");
  if (!libCall)
    return failure();
  auto rootLibCall = root->getAttrOfType<StringAttr>("library_call");
  if (!rootLibCall)
    return failure();

  if (libCall.getValue() == "npu_conv")
    return tileConvWithRoot(cursor, rewriter);
  if (libCall.getValue() == "npu_gemm" || libCall.getValue() == "npu_matmul")
    return tileGemmWithRoot(cursor, rewriter);

  // In the two-pass pipeline every seed is materialized from a fusion_group.
  // A single-op group is still a degenerate fused chain, so it must follow
  // the same DMA-aware with-root path instead of the old standalone path.
  if (rootLibCall.getValue() == "npu_maxpool")
    return tileMaxPoolWithRoot(cursor, rewriter);
  if (rootLibCall.getValue() == "npu_layout_nchw_to_nchwc32" ||
      rootLibCall.getValue() == "npu_layout_nchwc32_to_nchw" ||
      rootLibCall.getValue() == "npu_transpose")
    return tileLayoutWithRoot(cursor, rewriter);
  if (rootLibCall.getValue() == "npu_softmax")
    return tileSoftmaxWithRoot(cursor, rewriter);
  if (rootLibCall.getValue() == "npu_layernorm")
    return tileLayerNormWithRoot(cursor, rewriter);
  if (libCall.getValue() == "npu_gelu" || libCall.getValue() == "npu_matadd" ||
      libCall.getValue() == "mv_acc_to_spm")
    return tileElemWiseWithRoot(cursor, rewriter);

  return failure();
}

} // namespace npux
