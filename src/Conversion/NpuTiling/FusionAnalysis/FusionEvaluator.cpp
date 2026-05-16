//===============================================================
// src/Conversion/NpuTiling/FusionAnalysis/FusionEvaluator.cpp
// this file is for evaluator class of the profitability of 
// fusion candidates
//===============================================================

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "src/Conversion/NpuTiling/FusionAnalysis/FusionEvaluator.hpp"

namespace mlir {

FusionCostEvaluator::FusionCostEvaluator(double cpuWaitIrqTime)
    : waitIrqTime(cpuWaitIrqTime) {}

double FusionCostEvaluator::calculateOpCost(linalg::LinalgOp op) {
  return waitIrqTime + calculateDMATime(op);
}

double FusionCostEvaluator::evaluateFusionBenefit(
    linalg::LinalgOp seed, linalg::LinalgOp consumer) {
  double costUnfused = calculateOpCost(seed) + calculateOpCost(consumer);
  double costFused = waitIrqTime + calculateFusedDMATime(seed, consumer);
  return costUnfused - costFused;
}

// TODO: 实现 FusionCostEvaluator 类的成员函数，计算 DMA 时间和融合后的 DMA 时间
double FusionCostEvaluator::calculateDMATime(linalg::LinalgOp op) {
  return 0.0;
}

double FusionCostEvaluator::calculateFusedDMATime(
    linalg::LinalgOp seed, linalg::LinalgOp consumer) {
  return 0.0;
}

} // namespace mlir
