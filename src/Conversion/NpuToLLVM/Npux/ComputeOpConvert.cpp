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

        StringRef split_targetDim = (opType == ComputeOpType::gemm) ? "N" : "cout";
        StringRef loop_targetDim = (opType == ComputeOpType::gemm) ? "N" : "OC";

        scf::ForOp targetLoop = nullptr;
        scf::ForOp outermostFor = nullptr;
        scf::ForOp parentFor = op->getParentOfType<scf::ForOp>();

        // 2. 从内向外遍历，寻找第一个匹配 targetDim 的循环，同时记录最外层循环
        while (parentFor) {
          outermostFor = parentFor; // 一直更新，循环结束时这就是最外层循环

          // 优先看当前层是不是 split_dim 匹配
          if (auto splitDim =
                  parentFor->getAttrOfType<StringAttr>("npu.split_dim")) {
            if (splitDim.getValue() == split_targetDim) {
              targetLoop = parentFor;
              break;
            }
          }
          // 再看当前层是不是 loop_dim 匹配
          if (auto loopDim =
                  parentFor->getAttrOfType<StringAttr>("npu.loop_dim")) {
            if (loopDim.getValue() == loop_targetDim) {
              targetLoop = parentFor;
              break;
            }
          }
          parentFor = parentFor->getParentOfType<scf::ForOp>();
        }

        // 3. 确定安全的插入锚点 (hoistAnchor)
        scf::ForOp hoistAnchor = nullptr;

        if (targetLoop) {
          // 情况 A：找到了 N/cout 循环。向下找到包含当前 op 的“直接子循环”。
          // 这样 moveBefore(child) 会把操作提上去，放在 targetLoop 内部、child
          // 的外部。
          scf::ForOp child = op->getParentOfType<scf::ForOp>();
          while (child && child->getParentOp() != targetLoop) {
            child = child->getParentOfType<scf::ForOp>();
          }
          hoistAnchor = child;
        } else {
          // 情况 B (你的核心逻辑)：没有找到 N/cout 循环。
          // 说明 Bias 计算对当前所有循环都是不变量，直接放在最外层循环的外面！
          hoistAnchor = outermostFor;
        }

        // 4. 执行 Hoist
        if (hoistAnchor) {
          std::function<void(Operation *)> hoistOps =
              [&](Operation *opToHoist) {
                for (Value operand : opToHoist->getOperands()) {
                  if (Operation *defOp = operand.getDefiningOp()) {
                    // 只上提当前 hoistAnchor 内部的操作，避免越界
                    if (hoistAnchor->isAncestor(defOp)) {
                      hoistOps(defOp);
                    }
                  }
                }
                opToHoist->moveBefore(hoistAnchor);
              };

          // 提取 Bias (第三个输入) 的定义链
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

// --- Matadd Rewrite Pattern ---

class LinalgMataddToNpuxPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {
    // 1. Check Library Call Name
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr || libCallAttr.getValue() != "npu_matadd")
      return failure();

    Location loc = op.getLoc();

    // 2. Check Operands
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1) {
      return failure();
    }

    Value inputAMemRef = op.getInputs()[0];
    Value inputBMemRef = op.getInputs()[1];
    Value outputMemRef = op.getOutputs()[0];

    // 3. Memory Space Validation
    // Constraints: Inputs A/B must reside in ACC (space 3), output resides in
    // SPM (space 2).
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

    // 4. Parse Geometry (Shapes)
    // 根据示例，输入 shape 是 1x1x32x1024，提取最后两维作为 row 和 col
    auto inAType = mlir::cast<MemRefType>(inputAMemRef.getType());
    ArrayRef<int64_t> inAShape = inAType.getShape();
    int64_t rankA = inAShape.size();
    if (rankA < 2)
      return failure();

    int64_t col_num = inAShape.back();
    int64_t row_num = 1;
    for (int i = 0; i < rankA - 1; ++i) {
      row_num *= inAShape[i];
    }

    // 5. Create Constants for Geometry
    auto c32 = [&](int64_t v) {
      return rewriter.create<arith::ConstantIntOp>(loc, v, 32);
    };

    Value vColNum = c32(col_num);
    Value vRowNum = c32(row_num);

    // 6. Quantization Params
    // Matadd 通常使用 output 的 scale 和 zeropoint
    double outScale = getFloatAttr(op, "out_scale", 1.0);
    int64_t outZp = getIntAttr(op, "out_zp", 0);

    auto quantParams = getFixedPointParams(outScale);
    Value vOutZp = c32(outZp);
    Value vOutScale = c32(quantParams.multiplier);
    Value vOutScaleShift = c32(quantParams.shift);

    // 7. Replace Op with MataddRunOp
    rewriter.replaceOpWithNewOp<MataddRunOp>(op, inputAMemRef, inputBMemRef,
        outputMemRef, vColNum, vRowNum, vOutZp, vOutScale, vOutScaleShift);

    return success();
  }
};

} // namespace

void npux::populateLinalgConvToNpuxPattern(RewritePatternSet &patterns) {
  patterns.add<LinalgComputeToNpuxPattern>(patterns.getContext());
  patterns.add<LinalgMataddToNpuxPattern>(patterns.getContext());
}