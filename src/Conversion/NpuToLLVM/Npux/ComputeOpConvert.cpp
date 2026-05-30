//=======================================
// src/Conversion/NpuToLLVM/ComputeOpConvert.cpp
// This file implements convert linalg generic op (conv/gemm/matmul)
// to custom npux compute_run op
//=======================================
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Conversion/NpuToLLVM/NpuxConversionHelper.hpp"
#include "src/Dialect/Npucore/NpucoreOps.hpp"
#include "src/Dialect/Npux/NpuxOps.hpp"
#include "llvm/ADT/StringRef.h"

#include <cmath>
#include <cstdint>

using namespace mlir;
using namespace npux;

namespace {

// --- Helper Structures & Functions ---

struct FixedPointParams {
  int16_t multiplier;
  int16_t shift;
};

FixedPointParams getFixedPointParams(double scale) {
  if (std::abs(scale) < 1e-8)
    return {0, 0};

  // 核心修复：如果 scale 是 1.0 (允许微小浮点误差)，直接返回 multiplier=1, shift=0
  if (std::abs(scale - 1.0) < 1e-6)
    return {1, 0};

  int exponent;
  double mantissa = std::frexp(scale, &exponent);
  double mantissa_scaled = std::round(mantissa * 32768.0);
  if (mantissa_scaled >= 32768.0) {
    mantissa_scaled /= 2.0;
    exponent += 1;
  }
  return {static_cast<int16_t>(mantissa_scaled),
          static_cast<int16_t>(exponent - 15)};
}

int64_t getIntAttr(Operation *op, StringRef name, int64_t defaultVal) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name)) {
    return attr.getValue().getSExtValue();
  }
  return defaultVal;
}

double getFloatAttr(Operation *op, StringRef name, double defaultVal) {
  if (auto attr = op->getAttrOfType<FloatAttr>(name)) {
    return attr.getValueAsDouble();
  }
  return defaultVal;
}

int64_t getArrayAttr(
    Operation *op, StringRef name, int idx, int64_t defaultVal = 1) {
  if (auto attr = op->getAttrOfType<ArrayAttr>(name)) {
    if (idx < (int)attr.size()) {
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr[idx])) {
        return intAttr.getValue().getSExtValue();
      }
    }
  }
  return defaultVal;
}

class NpucoreMataddToNpuxPattern : public OpRewritePattern<npucore::MatAddOp> {
public:
  using OpRewritePattern<npucore::MatAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::MatAddOp op, PatternRewriter &rewriter) const override {
    if (op.hasTensorSemantics())
      return failure();
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return failure();

    Location loc = op.getLoc();
    Value inputAMemRef = op.getInputs()[0];
    Value inputBMemRef = op.getInputs()[1];
    Value outputMemRef = op.getOutputs()[0];

    auto checkSpace = [&](Value v, int expectedSpace) {
      if (!v)
        return true;
      auto type = mlir::dyn_cast<MemRefType>(v.getType());
      return type && type.getMemorySpaceAsInt() == expectedSpace;
    };

    if (!checkSpace(inputAMemRef, 3))
      return failure();
    if (!checkSpace(inputBMemRef, 3))
      return failure();
    if (!checkSpace(outputMemRef, 2))
      return failure();

    auto inAType = mlir::cast<MemRefType>(inputAMemRef.getType());
    ArrayRef<int64_t> inAShape = inAType.getShape();
    int64_t rankA = inAShape.size();
    if (rankA < 2)
      return failure();

    int64_t colNum = inAShape.back();
    int64_t rowNum = 1;
    for (int i = 0; i < rankA - 1; ++i)
      rowNum *= inAShape[i];

    auto c32 = [&](int64_t v) {
      return rewriter.create<arith::ConstantIntOp>(loc, v, 32);
    };

    Value vColNum = c32(colNum);
    Value vRowNum = c32(rowNum);

    double in1Scale = op.getIn1Scale().convertToDouble();
    double outScaleTarget = op.getOutScale().convertToDouble();
    int64_t outZp = op.getOutZp();
    double realMultiplier = in1Scale / outScaleTarget;
    auto quantParams = getFixedPointParams(realMultiplier);

    Value vOutZp = c32(outZp);
    Value vOutScale = c32(quantParams.multiplier);
    Value vOutScaleShift = c32(quantParams.shift);

    rewriter.replaceOpWithNewOp<MataddRunOp>(op, inputAMemRef, inputBMemRef,
        outputMemRef, vColNum, vRowNum, vOutZp, vOutScale, vOutScaleShift);
    return success();
  }
};

class NpucoreMatMulToNpuxPattern : public OpRewritePattern<npucore::MatMulOp> {
public:
  using OpRewritePattern<npucore::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::MatMulOp op, PatternRewriter &rewriter) const override {
    if (op.hasTensorSemantics())
      return failure();
    if (op.getInputs().size() < 2 || op.getInputs().size() > 3 ||
        op.getOutputs().size() != 1) {
      return failure();
    }

    Location loc = op.getLoc();
    Value inputAMemRef = op.getInputs()[0];
    Value inputBMemRef = op.getInputs()[1];
    Value outputMemRef = op.getOutputs()[0];

    Value psumMemRefForOp = nullptr;
    bool flagAccBias = false;
    bool flagDoAccum = false;

    StringRef loopStage = "single";
    if (auto attr = op->getAttrOfType<StringAttr>("npu.loop_stage"))
      loopStage = attr.getValue();

    StringRef splitStage = "single";
    if (auto attr = op->getAttrOfType<StringAttr>("npu.split_stage"))
      splitStage = attr.getValue();

    bool isLoopFirst = (loopStage == "head" || loopStage == "single");
    bool isSplitFirst = (splitStage == "head" || splitStage == "single");
    bool isFirstCalculation = (isLoopFirst && isSplitFirst);
    bool isLoopLast = (loopStage == "tail" || loopStage == "single");
    bool isSplitLast = (splitStage == "tail" || splitStage == "single");
    bool isLastCalculation = (isLoopLast && isSplitLast);
    bool originalDoRelu = (op.getDoRelu() != 0);
    bool doRelu = originalDoRelu && isLastCalculation;

    if (op.getInputs().size() == 3) {
      Value thirdInput = op.getInputs()[2];
      if (isFirstCalculation) {
        OpBuilder::InsertionGuard guard(rewriter);

        scf::ForOp targetLoop = nullptr;
        scf::ForOp outermostFor = nullptr;
        scf::ForOp parentFor = op->getParentOfType<scf::ForOp>();
        StringRef splitTargetDim = "N";
        StringRef loopTargetDim = "N";
        while (parentFor) {
          outermostFor = parentFor;
          if (auto splitDim =
                  parentFor->getAttrOfType<StringAttr>("npu.split_dim")) {
            if (splitDim.getValue() == splitTargetDim) {
              targetLoop = parentFor;
              break;
            }
          }
          if (auto loopDim =
                  parentFor->getAttrOfType<StringAttr>("npu.loop_dim")) {
            if (loopDim.getValue() == loopTargetDim) {
              targetLoop = parentFor;
              break;
            }
          }
          parentFor = parentFor->getParentOfType<scf::ForOp>();
        }

        scf::ForOp hoistAnchor = nullptr;
        if (targetLoop) {
          scf::ForOp child = op->getParentOfType<scf::ForOp>();
          while (child && child->getParentOp() != targetLoop)
            child = child->getParentOfType<scf::ForOp>();
          hoistAnchor = child;
        } else {
          hoistAnchor = outermostFor;
        }

        if (hoistAnchor) {
          std::function<void(Operation *)> hoistOps = [&](Operation *opToHoist) {
            for (Value operand : opToHoist->getOperands()) {
              if (Operation *defOp = operand.getDefiningOp()) {
                if (hoistAnchor->isAncestor(defOp))
                  hoistOps(defOp);
              }
            }
            opToHoist->moveBefore(hoistAnchor);
          };

          if (Operation *thirdDef = thirdInput.getDefiningOp()) {
            if (hoistAnchor->isAncestor(thirdDef))
              hoistOps(thirdDef);
          }
          rewriter.setInsertionPoint(hoistAnchor);
        }

        rewriter.create<MvinBiasOp>(loc, thirdInput);
        flagAccBias = true;
        flagDoAccum = true;
      } else if (loopStage == "body" || loopStage == "tail" ||
                 splitStage == "body" || splitStage == "tail") {
        psumMemRefForOp = outputMemRef;
        flagAccBias = false;
        flagDoAccum = true;
      } else {
        return failure();
      }
    } else {
      if (isFirstCalculation) {
        flagAccBias = false;
        flagDoAccum = false;
      } else if (loopStage == "body" || loopStage == "tail" ||
                 splitStage == "body" || splitStage == "tail") {
        psumMemRefForOp = outputMemRef;
        flagAccBias = false;
        flagDoAccum = true;
      } else {
        return failure();
      }
    }

    auto checkSpace = [&](Value v, int expectedSpace) {
      if (!v)
        return true;
      auto type = mlir::dyn_cast<MemRefType>(v.getType());
      return type && type.getMemorySpaceAsInt() == static_cast<unsigned>(expectedSpace);
    };

    if (!checkSpace(inputAMemRef, 2))
      return failure();
    if (!checkSpace(inputBMemRef, 2))
      return failure();

    auto outType = cast<MemRefType>(outputMemRef.getType());
    int outSpace = outType.getMemorySpaceAsInt();

    AccoutDest accDest;
    if (outSpace == 3) {
      accDest = AccoutDest::acc;
    } else if (outSpace == 2) {
      accDest = AccoutDest::spm;
    } else {
      return failure();
    }

    auto inAType = cast<MemRefType>(inputAMemRef.getType());
    auto inBType = cast<MemRefType>(inputBMemRef.getType());
    ArrayRef<int64_t> inAShape = inAType.getShape();
    ArrayRef<int64_t> inBShape = inBType.getShape();
    ArrayRef<int64_t> outShape = outType.getShape();

    SmallVector<int64_t, 4> inAStrides;
    SmallVector<int64_t, 4> inBStrides;
    SmallVector<int64_t, 4> outStrides;
    int64_t inAOffset, inBOffset, outOffset;

    if (failed(inAType.getStridesAndOffset(inAStrides, inAOffset)))
      return failure();
    if (failed(inBType.getStridesAndOffset(inBStrides, inBOffset)))
      return failure();
    if (failed(outType.getStridesAndOffset(outStrides, outOffset)))
      return failure();

    int64_t rankA = inAShape.size();
    int64_t rankB = inBShape.size();
    int64_t rankOut = outShape.size();

    int64_t aRow = inAShape[rankA - 2];
    int64_t aCol = inAShape[rankA - 1];
    int64_t aStride = inAStrides[rankA - 2];

    int64_t bRow = inBShape[rankB - 2];
    int64_t bCol = inBShape[rankB - 1];
    int64_t bStride = inBStrides[rankB - 2];

    int64_t outHeight = outShape[rankOut - 2];
    int64_t outWidth = outShape[rankOut - 1];
    int64_t outStride = outStrides[rankOut - 2];

    auto c32 = [&](int64_t v) {
      return rewriter.create<arith::ConstantIntOp>(loc, v, 32);
    };
    auto c8 = [&](int64_t v) {
      return rewriter.create<arith::ConstantIntOp>(loc, v, 8);
    };
    auto c1 = [&](bool v) {
      return rewriter.create<arith::ConstantIntOp>(loc, v, 1);
    };

    auto opTypeAttr =
        ComputeOpTypeAttr::get(rewriter.getContext(), ComputeOpType::gemm);
    auto dataflowModeAttr =
        DataflowModeAttr::get(rewriter.getContext(), DataflowMode::os);
    auto accoutDestAttr = AccoutDestAttr::get(rewriter.getContext(), accDest);
    Value vIntType = c8(0);

    auto safeM1 = [](int64_t v) { return v > 0 ? v - 1 : 0; };
    Value vZero32 = c32(0);
    Value vFalse = c1(false);

    Value vInAColM1 = c32(safeM1(aCol));
    Value vInARowM1 = c32(safeM1(aRow));
    Value vInAStride = c32(aStride);

    Value vInBColM1 = c32(safeM1(bCol));
    Value vInBRowM1 = c32(safeM1(bRow));
    Value vInBStride = c32(bStride);

    Value vBiasPsumWidth = c32(outWidth);
    Value vBiasPsumHeight = c32(outHeight);
    Value vBiasPsumStride = c32(outStride);
    Value vOutputStride = c32(outStride);

    double realMultiplier =
        (op.getLhsScale().convertToDouble() * op.getRhsScale().convertToDouble()) /
        op.getOutScale().convertToDouble();
    auto quantParams = getFixedPointParams(realMultiplier);

    Value vDoAccum = c1(flagDoAccum);
    Value vAccBias = c1(flagAccBias);
    Value vReluEnable = c1(doRelu);
    auto reluTypeAttr = ActivationTypeAttr::get(
        rewriter.getContext(),
        static_cast<ActivationType>(std::min<int64_t>(op.getReluType(), 4)));

    rewriter.replaceOpWithNewOp<ComputeRunOp>(op, opTypeAttr, dataflowModeAttr,
        accoutDestAttr, vIntType, inputAMemRef, inputBMemRef, psumMemRefForOp,
        outputMemRef, vZero32, vZero32, vZero32, vZero32, vZero32, vZero32,
        vZero32, vZero32, vFalse, vInAColM1, vInARowM1, vInAStride, vInBColM1,
        vInBRowM1, vInBStride, vBiasPsumWidth, vBiasPsumHeight,
        vBiasPsumStride, vOutputStride, vDoAccum, vReluEnable, reluTypeAttr,
        vAccBias, c32(op.getOutZp()), c32(quantParams.multiplier),
        c32(quantParams.shift), c32(op.getLhsZp()), c32(op.getRhsZp()));
    return success();
  }
};

class NpucoreConvToNpuxPattern : public OpRewritePattern<npucore::ConvOp> {
public:
  using OpRewritePattern<npucore::ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      npucore::ConvOp op, PatternRewriter &rewriter) const override {
    if (op.hasTensorSemantics())
      return failure();
    if (op.getInputs().size() < 2 || op.getInputs().size() > 3 ||
        op.getOutputs().size() != 1) {
      return failure();
    }

    Location loc = op.getLoc();
    Value inputAMemRef = op.getInputs()[0];
    Value inputBMemRef = op.getInputs()[1];
    Value outputMemRef = op.getOutputs()[0];

    Value psumMemRefForOp = nullptr;
    bool flagAccBias = false;
    bool flagDoAccum = false;

    StringRef loopStage = "single";
    if (auto attr = op->getAttrOfType<StringAttr>("npu.loop_stage"))
      loopStage = attr.getValue();

    StringRef splitStage = "single";
    if (auto attr = op->getAttrOfType<StringAttr>("npu.split_stage"))
      splitStage = attr.getValue();

    bool isLoopFirst = (loopStage == "head" || loopStage == "single");
    bool isSplitFirst = (splitStage == "head" || splitStage == "single");
    bool isFirstCalculation = (isLoopFirst && isSplitFirst);
    bool isLoopLast = (loopStage == "tail" || loopStage == "single");
    bool isSplitLast = (splitStage == "tail" || splitStage == "single");
    bool isLastCalculation = (isLoopLast && isSplitLast);
    bool originalDoRelu = (op.getDoRelu() != 0);
    bool doRelu = originalDoRelu && isLastCalculation;

    if (op.getInputs().size() == 3) {
      Value thirdInput = op.getInputs()[2];

      if (isFirstCalculation) {
        OpBuilder::InsertionGuard guard(rewriter);

        StringRef splitTargetDim = "cout";
        StringRef loopTargetDim = "OC";

        scf::ForOp targetLoop = nullptr;
        scf::ForOp outermostFor = nullptr;
        scf::ForOp parentFor = op->getParentOfType<scf::ForOp>();

        while (parentFor) {
          outermostFor = parentFor;

          if (auto splitDim =
                  parentFor->getAttrOfType<StringAttr>("npu.split_dim")) {
            if (splitDim.getValue() == splitTargetDim) {
              targetLoop = parentFor;
              break;
            }
          }
          if (auto loopDim =
                  parentFor->getAttrOfType<StringAttr>("npu.loop_dim")) {
            if (loopDim.getValue() == loopTargetDim) {
              targetLoop = parentFor;
              break;
            }
          }
          parentFor = parentFor->getParentOfType<scf::ForOp>();
        }

        scf::ForOp hoistAnchor = nullptr;
        if (targetLoop) {
          scf::ForOp child = op->getParentOfType<scf::ForOp>();
          while (child && child->getParentOp() != targetLoop)
            child = child->getParentOfType<scf::ForOp>();
          hoistAnchor = child;
        } else {
          hoistAnchor = outermostFor;
        }

        if (hoistAnchor) {
          std::function<void(Operation *)> hoistOps = [&](Operation *opToHoist) {
            for (Value operand : opToHoist->getOperands()) {
              if (Operation *defOp = operand.getDefiningOp()) {
                if (hoistAnchor->isAncestor(defOp))
                  hoistOps(defOp);
              }
            }
            opToHoist->moveBefore(hoistAnchor);
          };

          if (Operation *thirdDef = thirdInput.getDefiningOp()) {
            if (hoistAnchor->isAncestor(thirdDef))
              hoistOps(thirdDef);
          }
          rewriter.setInsertionPoint(hoistAnchor);
        }

        rewriter.create<MvinBiasOp>(loc, thirdInput);
        flagAccBias = true;
        flagDoAccum = true;
      } else if (loopStage == "body" || loopStage == "tail" ||
                 splitStage == "body" || splitStage == "tail") {
        psumMemRefForOp = outputMemRef;
        flagAccBias = false;
        flagDoAccum = true;
      } else {
        return failure();
      }
    } else {
      if (isFirstCalculation) {
        flagAccBias = false;
        flagDoAccum = false;
      } else if (loopStage == "body" || loopStage == "tail" ||
                 splitStage == "body" || splitStage == "tail") {
        psumMemRefForOp = outputMemRef;
        flagAccBias = false;
        flagDoAccum = true;
      } else {
        return failure();
      }
    }

    auto checkSpace = [&](Value v, int expectedSpace) {
      if (!v)
        return true;
      auto type = mlir::dyn_cast<MemRefType>(v.getType());
      return type && type.getMemorySpaceAsInt() == expectedSpace;
    };

    if (!checkSpace(inputAMemRef, 2))
      return failure();
    if (!checkSpace(inputBMemRef, 2))
      return failure();

    auto outType = cast<MemRefType>(outputMemRef.getType());
    int outSpace = outType.getMemorySpaceAsInt();

    AccoutDest accDest;
    if (outSpace == 3) {
      accDest = AccoutDest::acc;
    } else if (outSpace == 2) {
      accDest = AccoutDest::spm;
    } else {
      return failure();
    }

    auto inAType = mlir::cast<MemRefType>(inputAMemRef.getType());
    auto inBType = mlir::cast<MemRefType>(inputBMemRef.getType());
    ArrayRef<int64_t> inAShape = inAType.getShape();
    ArrayRef<int64_t> inBShape = inBType.getShape();
    ArrayRef<int64_t> outShape = outType.getShape();

    SmallVector<int64_t, 4> inAStrides;
    SmallVector<int64_t, 4> inBStrides;
    SmallVector<int64_t, 4> outStrides;
    int64_t inAOffset, inBOffset, outOffset;

    if (failed(inAType.getStridesAndOffset(inAStrides, inAOffset)))
      return failure();
    if (failed(inBType.getStridesAndOffset(inBStrides, inBOffset)))
      return failure();
    if (failed(outType.getStridesAndOffset(outStrides, outOffset)))
      return failure();

    int64_t aCol = inAShape[3];
    int64_t aRow = inAShape[2];
    int64_t aStride = inAStrides[2] / inAStrides[3];

    int64_t bRow = inBShape[4];
    int64_t bCol = inBShape[5];
    int64_t bStride = 0;

    int64_t outHeight = outShape[2];
    int64_t outWidth = outShape[3];
    int64_t outStride = outStrides[2] / outStrides[3];

    int64_t kernelSz = inBShape[2];
    int64_t strideVal = cast<IntegerAttr>(op.getStrides()[0]).getInt();
    int64_t dilationVal = cast<IntegerAttr>(op.getDilations()[0]).getInt();
    int64_t padModeVal = 0;
    bool isGroup = (op.getGroup() > 1);

    auto c32 = [&](int64_t v) {
      return rewriter.create<arith::ConstantIntOp>(loc, v, 32);
    };
    auto c8 = [&](int64_t v) {
      return rewriter.create<arith::ConstantIntOp>(loc, v, 8);
    };
    auto c1 = [&](bool v) {
      return rewriter.create<arith::ConstantIntOp>(loc, v, 1);
    };

    auto opTypeAttr =
        ComputeOpTypeAttr::get(rewriter.getContext(), ComputeOpType::conv);
    auto dataflowModeAttr =
        DataflowModeAttr::get(rewriter.getContext(), DataflowMode::ws);
    auto accoutDestAttr = AccoutDestAttr::get(rewriter.getContext(), accDest);
    Value vIntType = c8(0);

    Value vPadT = c32(0);
    Value vPadB = c32(0);
    Value vPadL = c32(0);
    Value vPadR = c32(0);
    Value vPadMode = c32(padModeVal);

    auto safeM1 = [](int64_t v) { return v > 0 ? v - 1 : 0; };
    Value vWeightShapeM1 = c32(safeM1(kernelSz));
    Value vWeightStrideM1 = c32(safeM1(strideVal));
    Value vWeightDilationM1 = c32(safeM1(dilationVal));
    Value vIsGroup = c1(isGroup);

    Value vInAColM1 = c32(safeM1(aCol));
    Value vInARowM1 = c32(safeM1(aRow));
    Value vInAStride = c32(aStride);

    Value vInBColM1 = c32(safeM1(bCol));
    Value vInBRowM1 = c32(safeM1(bRow));
    Value vInBStride = c32(bStride);

    Value vBiasPsumWidth = c32(outWidth);
    Value vBiasPsumHeight = c32(outHeight);
    Value vBiasPsumStride = c32(outStride);

    Value vOutputStride = c32(outStride);

    double inScale = op.getInScale().convertToDouble();
    double wScale = op.getWScale().convertToDouble();
    double outScaleTarget = op.getOutScale().convertToDouble();
    int64_t inZp = op.getInZp();
    int64_t wZp = op.getWZp();
    int64_t outZp = op.getOutZp();
    double realMultiplier = (inScale * wScale) / outScaleTarget;

    auto quantParams = getFixedPointParams(realMultiplier);
    Value vQuantScale = c32(quantParams.multiplier);
    Value vQuantShift = c32(quantParams.shift);
    Value vInAZp = c32(inZp);
    Value vInBZp = c32(wZp);
    Value vOutZp = c32(outZp);

    Value vDoAccum = c1(flagDoAccum);
    Value vAccBias = c1(flagAccBias);

    int64_t reluTypeVal = op.getReluType();
    ActivationType actType = static_cast<ActivationType>(reluTypeVal);
    if (reluTypeVal > 4)
      actType = ActivationType::relu;

    Value vReluEnable = c1(doRelu);
    auto reluTypeAttr = ActivationTypeAttr::get(rewriter.getContext(), actType);

    rewriter.replaceOpWithNewOp<ComputeRunOp>(op, opTypeAttr, dataflowModeAttr,
        accoutDestAttr, vIntType, inputAMemRef, inputBMemRef, psumMemRefForOp,
        outputMemRef, vPadT, vPadB, vPadL, vPadR, vPadMode, vWeightShapeM1,
        vWeightStrideM1, vWeightDilationM1, vIsGroup, vInAColM1, vInARowM1,
        vInAStride, vInBColM1, vInBRowM1, vInBStride, vBiasPsumWidth,
        vBiasPsumHeight, vBiasPsumStride, vOutputStride, vDoAccum,
        vReluEnable, reluTypeAttr, vAccBias, vOutZp, vQuantScale,
        vQuantShift, vInAZp, vInBZp);
    return success();
  }
};

} // namespace

void npux::populateNpucoreComputeToNpuxPattern(RewritePatternSet &patterns) {
  patterns.add<NpucoreMataddToNpuxPattern>(patterns.getContext());
  patterns.add<NpucoreMatMulToNpuxPattern>(patterns.getContext());
  patterns.add<NpucoreConvToNpuxPattern>(patterns.getContext());
}
