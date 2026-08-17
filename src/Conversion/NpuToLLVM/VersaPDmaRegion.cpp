// SPDX-License-Identifier: Apache-2.0

#include "src/Conversion/NpuToLLVM/VersaPDmaRegion.hpp"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

llvm::Expected<GemmDmaRegion> extractGemmDmaRegion(ComputeRunOp compute,
    bool requireOutputDma) {
  if (compute.getOpType() != ComputeOpType::gemm &&
      compute.getOpType() != ComputeOpType::conv)
    return extractionError("only GEMM or CONV compute_run is eligible");
  mlir::Block *block = compute->getBlock();
  DmaMvinOp loadA;
  DmaMvinOp loadW;
  MvinBiasOp loadBias;
  MvinScaleOp loadScale;
  MvinMetadataOp loadMetadata;
  ResaddLoadOp loadResidual;
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
    if (auto bias = mlir::dyn_cast<MvinBiasOp>(operation)) {
      if (bias->isBeforeInBlock(compute)) {
        if (llvm::Error error = requireUnique(loadBias, bias, "bias"))
          return std::move(error);
      }
    }
    if (auto scale = mlir::dyn_cast<MvinScaleOp>(operation)) {
      if (scale->isBeforeInBlock(compute)) {
        if (llvm::Error error = requireUnique(loadScale, scale, "scale"))
          return std::move(error);
      }
    }
    if (auto metadata = mlir::dyn_cast<MvinMetadataOp>(operation)) {
      if (metadata->isBeforeInBlock(compute)) {
        if (llvm::Error error = requireUnique(loadMetadata, metadata, "metadata"))
          return std::move(error);
      }
    }
    if (auto residual = mlir::dyn_cast<ResaddLoadOp>(operation)) {
      if (residual->isBeforeInBlock(compute)) {
        if (llvm::Error error =
                requireUnique(loadResidual, residual, "residual"))
          return std::move(error);
      }
    }
    if (requireOutputDma &&
        mlir::isa<DmaMvoutOp>(operation)) {
      auto mvout = mlir::cast<DmaMvoutOp>(operation);
      if (!compute->isBeforeInBlock(mvout))
        continue;
      if (mvout.getSrcMemref() == compute.getOutput()) {
        if (llvm::Error error = requireUnique(storeO, mvout, "output"))
          return std::move(error);
      }
    }
  }
  if (!loadA || !loadW || (requireOutputDma && !storeO))
    return extractionError("complete A/W DMA-in and required output DMA-out region is required");

  // A descriptor replaces an entire side-effecting region. Do not reach
  // across an intervening operation: doing so could leave a legacy DMA or
  // compute call between descriptor commands and silently change ordering.
  if (loadA == loadW)
    return extractionError("A and W must be loaded by distinct DMA operations");
  mlir::Operation *firstLoad = loadA.getOperation()->isBeforeInBlock(loadW.getOperation())
                                    ? loadA.getOperation()
                                    : loadW.getOperation();
  mlir::Operation *last = storeO ? storeO.getOperation() : compute.getOperation();
  for (mlir::Operation *current = firstLoad; current != last;
       current = current->getNextNode()) {
    const bool isMetadataPack = [&] {
      if (auto pack = mlir::dyn_cast<PackMetadataOp>(current))
        return loadMetadata && pack.getDestination() == loadMetadata.getSource();
      return false;
    }();
    const bool isMetadataDealloc = [&] {
      if (auto dealloc = mlir::dyn_cast<mlir::memref::DeallocOp>(current))
        return loadMetadata && dealloc.getMemref() == loadMetadata.getSource();
      return false;
    }();
    const bool isMetadataAlloc = [&] {
      if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(current))
        return loadMetadata && alloc.getMemref() == loadMetadata.getSource();
      return false;
    }();
    const bool isMetadataPackOperand = [&] {
      if (!loadMetadata || current->getNumResults() == 0)
        return false;
      for (mlir::Value result : current->getResults()) {
        bool usedOnlyByMetadataPack = true;
        bool hasMetadataPackUser = false;
        for (mlir::Operation *user : result.getUsers()) {
          auto pack = mlir::dyn_cast<PackMetadataOp>(user);
          if (!pack || pack.getDestination() != loadMetadata.getSource()) {
            usedOnlyByMetadataPack = false;
            break;
          }
          hasMetadataPackUser = true;
        }
        if (!usedOnlyByMetadataPack || !hasMetadataPackUser)
          return false;
      }
      return true;
    }();
    if (current == loadA.getOperation() || current == loadW.getOperation() ||
        current == loadBias.getOperation() || current == loadScale.getOperation() ||
        current == loadMetadata.getOperation() ||
        current == loadResidual.getOperation() ||
        current == compute.getOperation() || isMetadataPack || isMetadataDealloc ||
        isMetadataAlloc || isMetadataPackOperand)
      continue;
    return extractionError("DMA-GEMM-DMA region must be contiguous");
  }
  return GemmDmaRegion{loadA, loadW, loadBias, loadScale, loadMetadata, loadResidual,
      compute, storeO};
}

llvm::Expected<GemvDmaRegion> extractGemvDmaRegion(GemvRunOp compute) {
  mlir::Block *block = compute->getBlock();
  DmaMvinOp loadA;
  DmaMvinOp loadW;
  DmaMvinOp loadMetadata;
  DmaMvoutOp storeO;
  for (mlir::Operation &operation : *block) {
    if (auto mvin = mlir::dyn_cast<DmaMvinOp>(operation)) {
      if (!mvin->isBeforeInBlock(compute))
        continue;
      if (mvin.getDstMemref() == compute.getInputA()) {
        if (llvm::Error error = requireUnique(loadA, mvin, "GEMV A"))
          return std::move(error);
      }
      if (mvin.getDstMemref() == compute.getInputW()) {
        if (llvm::Error error = requireUnique(loadW, mvin, "GEMV W"))
          return std::move(error);
      }
      if (mvin.getDstMemref() == compute.getMetadata()) {
        if (llvm::Error error = requireUnique(loadMetadata, mvin, "GEMV metadata"))
          return std::move(error);
      }
    }
    if (auto mvout = mlir::dyn_cast<DmaMvoutOp>(operation)) {
      if (compute->isBeforeInBlock(mvout) &&
          mvout.getSrcMemref() == compute.getOutput()) {
        if (llvm::Error error = requireUnique(storeO, mvout, "GEMV output"))
          return std::move(error);
      }
    }
  }
  if (!loadA || !loadW || !loadMetadata)
    return extractionError("complete GEMV A/W/metadata DMA-in region is required");
  mlir::Operation *firstLoad = loadA.getOperation();
  for (DmaMvinOp load : {loadW, loadMetadata})
    if (load.getOperation()->isBeforeInBlock(firstLoad))
      firstLoad = load.getOperation();
  mlir::Operation *last = storeO ? storeO.getOperation() : compute.getOperation();
  for (mlir::Operation *current = firstLoad; current;
       current = current->getNextNode()) {
    if (current == loadA.getOperation() || current == loadW.getOperation() ||
        current == loadMetadata.getOperation() || current == compute.getOperation())
      ;
    else if (current != storeO.getOperation())
      return extractionError("DMA-GEMV-DMA region must be contiguous");
    if (current == last)
      break;
  }
  return GemvDmaRegion{loadA, loadW, loadMetadata, compute, storeO};
}

llvm::Expected<VpuDmaRegion> extractVpuDmaRegion(VpuRunOp compute) {
  DmaMvoutOp storeO;
  for (mlir::Operation &operation : *compute->getBlock()) {
    auto mvout = mlir::dyn_cast<DmaMvoutOp>(operation);
    if (!mvout || !compute->isBeforeInBlock(mvout) ||
        mvout.getSrcMemref() != compute.getOutput())
      continue;
    if (llvm::Error error = requireUnique(storeO, mvout, "VPU output"))
      return std::move(error);
  }
  if (!storeO)
    return extractionError("VPU requires an output DMA-out operation");
  for (mlir::Operation *current = compute.getOperation();
       current != storeO.getOperation(); current = current->getNextNode()) {
    if (current != compute.getOperation())
      return extractionError("VPU and MVOUT must be contiguous");
  }
  return VpuDmaRegion{compute, storeO};
}

} // namespace npux::versap
