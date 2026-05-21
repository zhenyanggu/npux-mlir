//===============================================================
// src/Conversion/NpuTiling/FusionAnalysis/FusionEvaluator.hpp
// this file declares evaluator class of the profitability of
// fusion candidates
//===============================================================

#ifndef NPUX_CONVERSION_NPUTILING_FUSIONANALYSIS_FUSIONEVALUATOR_HPP
#define NPUX_CONVERSION_NPUTILING_FUSIONANALYSIS_FUSIONEVALUATOR_HPP

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {

class FusionCostEvaluator {
public:
  explicit FusionCostEvaluator(
      double cpuWaitIrqTime = 210.0, double dmaCallTime = 65.0);

  double evaluateFusionBenefit(mlir::Operation *tail, mlir::Operation *consumer,
      llvm::ArrayRef<int64_t> currentTailTileSizes,
      llvm::ArrayRef<int64_t> fusedConsumerTileSizes);

private:
  double waitIrqTime;
  double dmaTime;

  double calculateOpCostWithTile(
      mlir::Operation *op, llvm::ArrayRef<int64_t> outputTileSizes);
};

} // namespace mlir

#endif // NPUX_CONVERSION_NPUTILING_FUSIONANALYSIS_FUSIONEVALUATOR_HPP
