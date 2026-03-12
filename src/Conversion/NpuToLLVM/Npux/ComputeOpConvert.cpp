//=======================================
// src/Conversion/NpuToLLVM/ComputeOpConvert.cpp
// This file implements convert linalg generic op (conv/gemm/matmul)
// to custom npux compute_run op
//=======================================
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Conversion/NpuToLLVM/NpuxConversionHelper.hpp"
#include "src/Dialect/Npux/NpuxOps.hpp"

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

// --- Main Rewrite Pattern ---

class LinalgComputeToNpuxPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    // 1. Check Library Call Name
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr)
      return failure();

    StringRef libName = libCallAttr.getValue();

    if (libName == "npu_matadd") {
      if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
        return failure();

      Value inputAMemRef = op.getInputs()[0];
      Value inputBMemRef = op.getInputs()[1];
      Value outputMemRef = op.getOutputs()[0];

      auto inAType = dyn_cast<MemRefType>(inputAMemRef.getType());
      auto inBType = dyn_cast<MemRefType>(inputBMemRef.getType());
      auto outType = dyn_cast<MemRefType>(outputMemRef.getType());
      if (!inAType || !inBType || !outType)
        return failure();

      // MATADD contract: A/B in ACC, output in SPM.
      if (inAType.getMemorySpaceAsInt() != 3 || inBType.getMemorySpaceAsInt() != 3)
        return failure();
      if (outType.getMemorySpaceAsInt() != 2)
        return failure();

      auto outShape = outType.getShape();
      if (outShape.empty() || outShape.size() > 2)
        return failure();
      for (int64_t d : outShape) {
        if (d == ShapedType::kDynamic) {
          op.emitError("npu_matadd requires static shape after tiling");
          return failure();
        }
      }

      int64_t row = 1;
      int64_t col = outShape.back();
      if (outShape.size() == 2)
        row = outShape[0];

      if (col <= 0 || row <= 0)
        return failure();
      if (col > 255 || row > 255) {
        op.emitError()
            << "MATADD row/col exceed hardware 8-bit fields: row=" << row
            << ", col=" << col
            << ". Expected npu-tiling to split into row/col <= 255.";
        return failure();
      }

      double lhsScale = getFloatAttr(op, "lhs_scale", 1.0);
      double rhsScale = getFloatAttr(op, "rhs_scale", 1.0);
      if (std::abs(lhsScale - rhsScale) > 1e-6) {
        op.emitError()
            << "npu_matadd currently requires lhs_scale == rhs_scale "
            << "(got lhs=" << lhsScale << ", rhs=" << rhsScale << ")";
        return failure();
      }

      double outScaleTarget = getFloatAttr(op, "out_scale", 1.0);
      if (outScaleTarget <= 0.0)
        return failure();
      int64_t outZp = getIntAttr(op, "out_zp", 0);

      double realMultiplier = lhsScale / outScaleTarget;
      auto quantParams = getFixedPointParams(realMultiplier);

      Location loc = op.getLoc();
      auto c32 = [&](int64_t v) {
        return rewriter.create<arith::ConstantIntOp>(loc, v, 32);
      };

      rewriter.replaceOpWithNewOp<MataddRunOp>(op,
          inputAMemRef, inputBMemRef, outputMemRef, c32(col), c32(row),
          c32(outZp), c32(quantParams.multiplier), c32(quantParams.shift));
      return success();
    }

    ComputeOpType opType;

    if (libName == "npu_conv") {
      opType = ComputeOpType::conv;
    } else if (libName == "npu_matmul" || libName == "npu_gemm") {
      opType = ComputeOpType::gemm;
    } else {
      return failure();
    }

    Location loc = op.getLoc();

    // 2. Get Operands & Logic Selection
    if (op.getInputs().size() < 2 || op.getOutputs().size() != 1) {
      return failure();
    }

    Value inputAMemRef = op.getInputs()[0];
    Value inputBMemRef = op.getInputs()[1]; // Weight / RHS
    Value outputMemRef = op.getOutputs()[0];

    // Variables to be determined by logic path
    Value psumMemRefForOp = nullptr; // Passed to ComputeRunOp
    bool flagAccBias = false;
    bool flagDoAccum = false;

    // 获取 loop_stage 标签，如果没有打标签，默认作为 single 处理以保证安全
    StringRef loopStage = "single";
    if (auto attr = op->getAttrOfType<StringAttr>("npu.loop_stage")) {
      loopStage = attr.getValue();
    }

    // 获取 split_stage 标签，如果没有打标签，默认作为 single 处理
    StringRef splitStage = "single";
    if (auto attr = op->getAttrOfType<StringAttr>("npu.split_stage")) {
      splitStage = attr.getValue();
    }

    // 提取判断逻辑：判断当前是否是输出块的绝对“第一次计算”
    bool isLoopFirst = (loopStage == "head" || loopStage == "single");
    bool isSplitFirst = (splitStage == "head" || splitStage == "single");
    bool isFirstCalculation = (isLoopFirst && isSplitFirst);
    bool isLoopLast = (loopStage == "tail" || loopStage == "single");
    bool isSplitLast = (splitStage == "tail" || splitStage == "single");
    bool isLastCalculation = (isLoopLast && isSplitLast);
    bool originalDoRelu = (getIntAttr(op, "do_relu", 0) != 0);
    bool doRelu = originalDoRelu && isLastCalculation;

    if (op.getInputs().size() >= 3) {
      // ==========================================
      // 场景 A：带有 Bias 的情况
      // ==========================================
      Value thirdInput = op.getInputs()[2];

      if (isFirstCalculation) {
        // === Case A1: Head / Single (Bias Logic) ===
        OpBuilder::InsertionGuard guard(rewriter);
        scf::ForOp hoistAnchor = nullptr;

        if (opType == ComputeOpType::gemm) {
          auto parentFor = op->getParentOfType<scf::ForOp>();
          while (parentFor) {
            if (auto splitDim =
                    parentFor->getAttrOfType<StringAttr>("npu.split_dim")) {
              if (splitDim.getValue() == "M") {
                hoistAnchor = parentFor;
                break;
              }
            }
            parentFor = parentFor->getParentOfType<scf::ForOp>();
          }
        } else if (opType == ComputeOpType::conv) {
          scf::ForOp coutLoop = nullptr;
          scf::ForOp outermostFor = nullptr;
          auto parentFor = op->getParentOfType<scf::ForOp>();

          while (parentFor) {
            outermostFor = parentFor;
            if (auto splitDim =
                    parentFor->getAttrOfType<StringAttr>("npu.split_dim")) {
              StringRef dimVal = splitDim.getValue();
              if (dimVal == "cout" || dimVal == "Cout_c" || dimVal == "Cout") {
                coutLoop = parentFor;
              }
            }
            parentFor = parentFor->getParentOfType<scf::ForOp>();
          }

          if (coutLoop) {
            scf::ForOp child = op->getParentOfType<scf::ForOp>();
            while (child && child->getParentOp() != coutLoop) {
              child = child->getParentOfType<scf::ForOp>();
            }
            hoistAnchor = child;
          } else {
            hoistAnchor = outermostFor;
          }
        }

        if (hoistAnchor) {
          std::function<void(Operation *)> hoistOps =
              [&](Operation *opToHoist) {
                for (Value operand : opToHoist->getOperands()) {
                  if (Operation *defOp = operand.getDefiningOp()) {
                    if (hoistAnchor->isAncestor(defOp)) {
                      hoistOps(defOp);
                    }
                  }
                }
                opToHoist->moveBefore(hoistAnchor);
              };

          if (Operation *thirdDef = thirdInput.getDefiningOp()) {
            if (hoistAnchor->isAncestor(thirdDef)) {
              hoistOps(thirdDef);
            }
          }
          rewriter.setInsertionPoint(hoistAnchor);
        }

        // 1. Create the dedicated MvinBiasOp
        rewriter.create<MvinBiasOp>(loc, thirdInput);

        // 2. Configure ComputeOp Flags
        psumMemRefForOp =
            nullptr;        // Bias 已经在寄存器里了，不需要传入 psum buffer
        flagAccBias = true; // 启用加偏置

        // 【核心修改点】：因为需要加偏置，有加法操作，DoAccum 必须为 true
        flagDoAccum = true;

      } else if (loopStage == "body" || loopStage == "tail" ||
                 splitStage == "body" || splitStage == "tail") {
        // === Case A2: Body / Tail (Accumulation Logic) ===
        psumMemRefForOp = outputMemRef;
        flagAccBias = false; // 偏置已经在 head 阶段加过了
        flagDoAccum = true;  // 开启 Psum 累加模式
      } else {
        return failure();
      }

    } else {
      // ==========================================
      // 场景 B：没有 Bias 的情况
      // ==========================================
      if (isFirstCalculation) {
        // === Case B1: 无 Bias 且是第一次计算 ===
        psumMemRefForOp = nullptr;
        flagAccBias = false;

        // 【核心修改点】：既没有 Bias，也不需要累加 Psum，此时才是真正的 0
        flagDoAccum = false;

      } else if (loopStage == "body" || loopStage == "tail" ||
                 splitStage == "body" || splitStage == "tail") {
        // === Case B2: 无 Bias 但处于 Body/Tail 累加阶段 ===
        psumMemRefForOp = outputMemRef;
        flagAccBias = false;
        flagDoAccum =
            true; // 虽然没 Bias，但需要把之前的 Psum 加进来，所以是 true
      } else {
        return failure();
      }
    }

    // 4. Memory Space Validation & Destination Inference
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

    auto outType = mlir::cast<MemRefType>(outputMemRef.getType());
    int outSpace = outType.getMemorySpaceAsInt();

    AccoutDest accDest;
    if (outSpace == 3) {
      accDest = AccoutDest::acc;
    } else if (outSpace == 2) {
      accDest = AccoutDest::spm;
    } else {
      return failure();
    }

    // 5. Parse Geometry (Shapes & Strides)
    auto inAType = mlir::cast<MemRefType>(inputAMemRef.getType());
    auto inBType = mlir::cast<MemRefType>(inputBMemRef.getType());
    ArrayRef<int64_t> inAShape = inAType.getShape();
    ArrayRef<int64_t> inBShape = inBType.getShape();
    ArrayRef<int64_t> outShape = outType.getShape();

    SmallVector<int64_t, 4> inAStrides;
    SmallVector<int64_t, 4> inBStrides;
    SmallVector<int64_t, 4> outStrides;
    int64_t inAoffset, inBoffset, outOffset;

    if (failed(inAType.getStridesAndOffset(inAStrides, inAoffset))) {
      return failure();
    }
    if (failed(inBType.getStridesAndOffset(inBStrides, inBoffset))) {
      return failure();
    }
    if (failed(outType.getStridesAndOffset(outStrides, outOffset))) {
      return failure();
    }
    int64_t a_col = 1, a_row = 1, a_stride = 0;
    int64_t b_col = 1, b_row = 1, b_stride = 0;
    int64_t out_width = 1, out_height = 1, out_stride = 0;
    int64_t kernel_sz = 1, stride_val = 1, dilation_val = 1;
    int64_t pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    int64_t pad_mode_val = 0;
    bool is_group = false;

    if (opType == ComputeOpType::conv) {
      a_row = inAShape[2];
      a_col = inAShape[3];
      a_stride = inAStrides[2] / inAStrides[3];

      b_row = inBShape[4];
      b_col = inBShape[5];

      out_height = outShape[2];
      out_width = outShape[3];
      out_stride = outStrides[2] / outStrides[3];

      kernel_sz = inBShape[2];
      stride_val = getArrayAttr(op, "strides", 0, 1);
      dilation_val = getArrayAttr(op, "dilations", 0, 1);

      pad_mode_val = getIntAttr(op, "pad_mode", 0);
      is_group = (getIntAttr(op, "group", 1) > 1);
    } else {
      int64_t rankA = inAShape.size();
      int64_t rankB = inBShape.size();
      int64_t rankOut = outShape.size();

      a_row = inAShape[rankA - 2];
      a_col = inAShape[rankA - 1];
      a_stride = inAStrides[rankA - 2];

      b_row = inBShape[rankB - 2];
      b_col = inBShape[rankB - 1];
      b_stride = inBStrides[rankB - 2];

      out_height = outShape[rankOut - 2];
      out_width = outShape[rankOut - 1];
      out_stride = outStrides[rankOut - 2];
    }

    // 6. Create Constants
    auto c32 = [&](int64_t v) {
      return rewriter.create<arith::ConstantIntOp>(loc, v, 32);
    };
    auto c8 = [&](int64_t v) {
      return rewriter.create<arith::ConstantIntOp>(loc, v, 8);
    };
    auto c1 = [&](bool v) {
      return rewriter.create<arith::ConstantIntOp>(loc, v, 1);
    };

    // --- Operation Control ---
    auto opTypeAttr = ComputeOpTypeAttr::get(rewriter.getContext(), opType);
    auto dataflowMode =
        (opType == ComputeOpType::conv) ? DataflowMode::ws : DataflowMode::os;
    auto dataflowModeAttr =
        DataflowModeAttr::get(rewriter.getContext(), dataflowMode);
    auto accoutDestAttr = AccoutDestAttr::get(rewriter.getContext(), accDest);
    Value vIntType = c8(0);

    Value vPadT = c32(pad_t);
    Value vPadB = c32(pad_b);
    Value vPadL = c32(pad_l);
    Value vPadR = c32(pad_r);
    Value vPadMode = c32(pad_mode_val);

    auto safe_m1 = [](int64_t v) { return v > 0 ? v - 1 : 0; };

    if (opType == ComputeOpType::gemm) {
      // Hardware field width limits:
      //   SA_IN_A.COL_m1 : 11-bit (max 2047)
      //   SA_IN_B.ROW_m1 : 11-bit (max 2047)
      constexpr int64_t kMaxGemmKMinus1 = 2047;
      int64_t aColM1 = safe_m1(a_col);
      int64_t bRowM1 = safe_m1(b_row);
      if (aColM1 > kMaxGemmKMinus1 || bRowM1 > kMaxGemmKMinus1) {
        op.emitError()
            << "GEMM K dimension exceeds hardware field limit before lowering: "
            << "a_col_m1=" << aColM1 << ", b_row_m1=" << bRowM1
            << " (max 2047). "
            << "Expected K-splitting to keep each tile K<=2048.";
        return failure();
      }
    }

    Value vWeightShapeM1 = c32(safe_m1(kernel_sz));
    Value vWeightStrideM1 = c32(safe_m1(stride_val));
    Value vWeightDilationM1 = c32(safe_m1(dilation_val));
    Value vIsGroup = c1(is_group);

    Value vInAColM1 = c32(safe_m1(a_col));
    Value vInARowM1 = c32(safe_m1(a_row));
    Value vInAStride = c32(a_stride);

    Value vInBColM1 = c32(safe_m1(b_col));
    Value vInBRowM1 = c32(safe_m1(b_row));
    Value vInBStride = c32(b_stride);

    Value vBiasPsumWidth = c32(out_width);
    Value vBiasPsumHeight = c32(out_height);
    Value vBiasPsumStride = c32(out_stride);

    Value vOutputStride = c32(out_stride);

    // 7. Quantization Params
    double realMultiplier = 1.0;
    int64_t inZp = 0;
    int64_t wZp = 0;
    int64_t outZp = 0;
    double outScaleTarget = getFloatAttr(op, "out_scale", 1.0);
    outZp = getIntAttr(op, "out_zp", 0);
    if (opType == ComputeOpType::conv) {
      double inScale = getFloatAttr(op, "in_scale", 1.0);
      double wScale = getFloatAttr(op, "w_scale", 1.0);
      inZp = getIntAttr(op, "in_zp", 0);
      wZp = getIntAttr(op, "w_zp", 0);
      realMultiplier = (inScale * wScale) / outScaleTarget;
    } else {
      inZp = getIntAttr(op, "lhs_zp", 0);
      wZp = getIntAttr(op, "rhs_zp", 0);
      double lhsScale = getFloatAttr(op, "lhs_scale", 1.0);
      double rhsScale = getFloatAttr(op, "rhs_scale", 1.0);
      realMultiplier = (lhsScale * rhsScale) / outScaleTarget;
    }
    auto quantParams = getFixedPointParams(realMultiplier);
    Value vQuantScale = c32(quantParams.multiplier);
    Value vQuantShift = c32(quantParams.shift);
    Value vInAZp = c32(inZp);
    Value vInBZp = c32(wZp);
    Value vOutZp = c32(outZp);

    // 8. Flags (Accumulate, ReLU, Bias)
    Value vDoAccum = c1(flagDoAccum);
    Value vAccBias = c1(flagAccBias);

    int64_t reluTypeVal = getIntAttr(op, "relu_type", 0);
    ActivationType actType = static_cast<ActivationType>(reluTypeVal);
    if (reluTypeVal > 4)
      actType = ActivationType::relu;

    Value vReluEnable = c1(doRelu);
    auto reluTypeAttr = ActivationTypeAttr::get(rewriter.getContext(), actType);

    // 9. Create ComputeRunOp
    rewriter.replaceOpWithNewOp<ComputeRunOp>(op, opTypeAttr, dataflowModeAttr,
        accoutDestAttr, vIntType,

        inputAMemRef, inputBMemRef, psumMemRefForOp, outputMemRef,

        vPadT, vPadB, vPadL, vPadR, vPadMode,

        vWeightShapeM1, vWeightStrideM1, vWeightDilationM1, vIsGroup,

        vInAColM1, vInARowM1, vInAStride, vInBColM1, vInBRowM1, vInBStride,

        vBiasPsumWidth, vBiasPsumHeight, vBiasPsumStride,

        vOutputStride,

        vDoAccum, vReluEnable, reluTypeAttr, vAccBias,

        vOutZp, vQuantScale, vQuantShift, vInAZp, vInBZp);

    return success();
  }
};

} // namespace

void npux::populateLinalgConvToNpuxPattern(RewritePatternSet &patterns) {
  patterns.add<LinalgComputeToNpuxPattern>(patterns.getContext());
}
