// SPDX-License-Identifier: Apache-2.0

#include "src/Conversion/NpuToLLVM/VersaPDmaRegion.hpp"

#include "llvm/Support/Error.h"

namespace npux::versap {
namespace {

llvm::Error extractionError(llvm::StringRef detail) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
      "Versa-P DMA region extraction: %s", detail.str().c_str());
}

template <typename Op>
llvm::Error requireUnique(Op &result, Op candidate, llvm::StringRef name) {
  if (result)
    return extractionError(name);
  result = candidate;
  return llvm::Error::success();
}

} // namespace

llvm::Expected<GemmDmaRegion> extractGemmDmaRegion(ComputeRunOp compute) {
  if (compute.getOpType() != ComputeOpType::gemm)
    return extractionError("only GEMM compute_run is eligible");
  mlir::Block *block = compute->getBlock();
  DmaMvinOp loadA;
  DmaMvinOp loadW;
  DmaMvoutOp storeO;
  for (mlir::Operation &operation : *block) {
    if (auto mvin = mlir::dyn_cast<DmaMvinOp>(operation)) {
      if (!mvin->isBeforeInBlock(compute))
        continue;
      if (mvin.getDstMemref() == compute.getInputA()) {
        if (llvm::Error error = requireUnique(loadA, mvin, "A"))
          return std::move(error);
      }
      if (mvin.getDstMemref() == compute.getInputB()) {
        if (llvm::Error error = requireUnique(loadW, mvin, "W"))
          return std::move(error);
      }
    }
    if (auto mvout = mlir::dyn_cast<DmaMvoutOp>(operation)) {
      if (!compute->isBeforeInBlock(mvout))
        continue;
      if (mvout.getSrcMemref() == compute.getOutput()) {
        if (llvm::Error error = requireUnique(storeO, mvout, "output"))
          return std::move(error);
      }
    }
  }
  if (!loadA || !loadW || !storeO)
    return extractionError("complete A/W DMA-in and output DMA-out region is required");

  // A descriptor replaces an entire side-effecting region. Do not reach
  // across an intervening operation: doing so could leave a legacy DMA or
  // compute call between descriptor commands and silently change ordering.
  if (loadA == loadW)
    return extractionError("A and W must be loaded by distinct DMA operations");
  mlir::Operation *firstLoad = loadA.getOperation()->isBeforeInBlock(loadW.getOperation())
                                    ? loadA.getOperation()
                                    : loadW.getOperation();
  for (mlir::Operation *current = firstLoad;
       current != storeO.getOperation(); current = current->getNextNode()) {
    if (current == loadA.getOperation() || current == loadW.getOperation() ||
        current == compute.getOperation())
      continue;
    return extractionError("DMA-GEMM-DMA region must be contiguous");
  }
  return GemmDmaRegion{loadA, loadW, compute, storeO};
}

} // namespace npux::versap