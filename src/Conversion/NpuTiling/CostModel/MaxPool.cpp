//======================================================
// src/Conversion/NpuTiling/CostModel/MaxPool.cpp
// This file implements the cost model for MaxPool operations,
// calculating tile sizes based on SRAM capacity constraints.
//======================================================

#include "src/Conversion/NpuTiling/CostModel/NpuCostModel.hpp"
#include <algorithm>

using namespace mlir;

namespace npux {

llvm::SmallVector<int64_t> NPUCostModel::getMaxPoolTileSizes(
    npucore::MaxPoolOp op) {
  auto outputType = cast<ShapedType>(op.getOutputs()[0].getType());
  if (outputType.getRank() != 4)
    return {1, 1, 1, 1};

  ArrayRef<int64_t> shape = outputType.getShape();
  int64_t n = shape[0];
  int64_t c = shape[1];
  int64_t oh = shape[2];
  int64_t ow = shape[3];

  int64_t bitWidth = outputType.getElementType().getIntOrFloatBitWidth();
  int64_t bytesPerElem = std::max<int64_t>(1, bitWidth / 8);
  int64_t spmSize = this->hw.spmSizeBytes;
  int64_t bytesPerOutPixel = 5 * bytesPerElem;
  int64_t maxPixels = spmSize / bytesPerOutPixel;
  if (maxPixels <= 0)
    return {1, 1, 1, 1};

  int64_t tN = 1;
  int64_t tC = 1;
  int64_t tOh = 1;
  int64_t tOw = 1;
  int64_t remainingPixels = maxPixels;

  if (ow != ShapedType::kDynamic && ow > 0) {
    tOw = std::min<int64_t>(ow, remainingPixels);
    remainingPixels = std::max<int64_t>(1, remainingPixels / tOw);
  }
  if (oh != ShapedType::kDynamic && oh > 0) {
    tOh = std::min<int64_t>(oh, remainingPixels);
    remainingPixels = std::max<int64_t>(1, remainingPixels / tOh);
  }
  if (c != ShapedType::kDynamic && c > 0)
    tC = std::min<int64_t>(c, remainingPixels);
  if (n != ShapedType::kDynamic && n > 0)
    tN = 1;

  llvm::errs() << "[CostModel] npucore.maxpool: Tile=[N:" << tN
               << ", C:" << tC << ", OH:" << tOh << ", OW:" << tOw << "]\n";
  return {tN, tC, tOh, tOw};
}

} // namespace npux
