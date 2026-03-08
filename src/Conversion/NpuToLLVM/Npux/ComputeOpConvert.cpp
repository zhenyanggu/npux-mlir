//=======================================
// src/Conversion/NpuToLLVM/ComputeOpConvert.cpp
// This file implements convert linalg generic op (conv/gemm/matmul)
// to custom npux compute_run op
//=======================================
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Conversion/NpuToLLVM/NpuxConversionHelper.hpp"
#include "src/Dialect/Npux/NpuxOps.hpp"

#include <cmath>

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

    // Variables to be determined by logic path
    Value psumMemRefForOp = nullptr; // Passed to ComputeRunOp
    bool flagAccBias = false;
    bool flagDoAccum = false;

    // 获取 loop_stage 标签，如果没有打标签，默认作为 single 处理以保证安全
    StringRef loopStage = "single";
    if (auto attr = op->getAttrOfType<StringAttr>("npu.loop_stage")) {
      loopStage = attr.getValue();
    }

    if (op.getInputs().size() >= 3) {
      Value thirdInput = op.getInputs()[2];

      if (loopStage == "head" || loopStage == "single") {
        // === Case 1: Head / Single (Bias Logic) ===
        
        // 使用 InsertionGuard 保存当前的插入点
        OpBuilder::InsertionGuard guard(rewriter);
        
        scf::ForOp hoistAnchor = nullptr;

        if (opType == ComputeOpType::gemm) {
          // GEMM 的 MvinBias 逻辑保持不变
          auto parentFor = op->getParentOfType<scf::ForOp>();
          while (parentFor) {
            if (auto splitDim = parentFor->getAttrOfType<StringAttr>("npu.split_dim")) {
              if (splitDim.getValue() == "M") {
                hoistAnchor = parentFor;
                break;
              }
            }
            parentFor = parentFor->getParentOfType<scf::ForOp>();
          }
        } else if (opType == ComputeOpType::conv) {
          // Conv 的 MvinBias 逻辑
          scf::ForOp coutLoop = nullptr;
          scf::ForOp outermostFor = nullptr;
          auto parentFor = op->getParentOfType<scf::ForOp>();
          
          while (parentFor) {
            outermostFor = parentFor; // 一路记录，直到最外层
            if (auto splitDim = parentFor->getAttrOfType<StringAttr>("npu.split_dim")) {
              StringRef dimVal = splitDim.getValue();
              // 兼容可能的 Cout 标签命名
              if (dimVal == "cout" || dimVal == "Cout_c" || dimVal == "Cout") {
                coutLoop = parentFor;
              }
            }
            parentFor = parentFor->getParentOfType<scf::ForOp>();
          }
          
          if (coutLoop) {
            // 放到 Cout_c 这一层：即 cout 循环的内部，紧贴着下级内层循环（H）的外面
            scf::ForOp child = op->getParentOfType<scf::ForOp>();
            while (child && child->getParentOp() != coutLoop) {
              child = child->getParentOfType<scf::ForOp>();
            }
            hoistAnchor = child; 
          } else {
            // 如果没有任何 Cout_c 标签，放到最外层循环的外面
            hoistAnchor = outermostFor;
          }
        }

        if (hoistAnchor) {
          // ========================================================
          // 【核心修复】：消除 Dominance 错误
          // 将 thirdInput 的定义指令（及其依赖）连根拔起，一起提到锚点循环外
          // ========================================================
          std::function<void(Operation*)> hoistOps = [&](Operation* opToHoist) {
            for (Value operand : opToHoist->getOperands()) {
              if (Operation *defOp = operand.getDefiningOp()) {
                // 如果依赖的指令也在锚点循环内部，递归提取它
                if (hoistAnchor->isAncestor(defOp)) {
                  hoistOps(defOp);
                }
              }
            }
            // 将该指令移动到锚点循环之前
            opToHoist->moveBefore(hoistAnchor);
          };
          
          if (Operation *thirdDef = thirdInput.getDefiningOp()) {
            if (hoistAnchor->isAncestor(thirdDef)) {
              hoistOps(thirdDef);
            }
          }

          // 设置插入点为锚点循环前方
          rewriter.setInsertionPoint(hoistAnchor);
        }
        
        // 1. Create the dedicated MvinBiasOp
        rewriter.create<MvinBiasOp>(loc, thirdInput);

        // Guard 生命周期结束，插入点自动恢复到原来的 GenericOp 处
        // 后续的 ComputeRunOp 依然会正确生成在最内层

        // 2. Configure ComputeOp Flags
        psumMemRefForOp = nullptr;   // Bias 已经在寄存器里了，不需要传入 psum buffer
        flagAccBias = true;  // 启用加偏置
        flagDoAccum = false; // 不做 Psum 累加
      } else if (loopStage == "body" || loopStage == "tail") {
        // === Case 2: Body / Tail (Accumulation Logic) ===
        psumMemRefForOp = outputMemRef;
        flagAccBias = false; // 偏置已经在 head 阶段加过了
        flagDoAccum = true;  // 开启 Psum 累加模式
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

    SmallVector<int64_t, 4> inAStrides ;
    SmallVector<int64_t, 4> inBStrides ;
    SmallVector<int64_t, 4> outStrides ;
    int64_t inAoffset, inBoffset, outOffset;

    if(failed(inAType.getStridesAndOffset(inAStrides,inAoffset))) {
      return failure();
    }
    if(failed(inBType.getStridesAndOffset(inBStrides,inBoffset))) {
      return failure();
    }
    if(failed(outType.getStridesAndOffset(outStrides,outOffset))) {
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
      a_stride = inAStrides[2]/inAStrides[3];

      b_row = inBShape[4];
      b_col = inBShape[5];

      out_height = outShape[2];
      out_width = outShape[3];
      out_stride = outStrides[2]/outStrides[3];

      kernel_sz = outShape[2] ;
      stride_val = getArrayAttr(op, "strides", 0, 1);
      dilation_val = getArrayAttr(op, "dilations", 0, 1);

      pad_mode_val = getIntAttr(op, "pad_mode", 0);
      is_group = (getIntAttr(op, "group", 1) > 1);
    } else {
      if (inAShape.size() >= 2 && outShape.size() >= 2) {
        a_row = inAShape[0];
        a_col = inAShape[1];
        b_row = inAShape[1];
        b_col = outShape[1];
        out_height = a_row;
        out_width = b_col;
      }
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
    auto dataflowModeAttr =
        DataflowModeAttr::get(rewriter.getContext(), DataflowMode::ws);
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
      inZp = getIntAttr(op, "in_zp", 0);
      realMultiplier = inScale / outScaleTarget;
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

    bool doRelu = (getIntAttr(op, "do_relu", 0) != 0);
    int64_t reluTypeVal = getIntAttr(op, "relu_type", 0);
    ActivationType actType = static_cast<ActivationType>(reluTypeVal);
    if (reluTypeVal > 4)
      actType = ActivationType::relu;

    Value vReluEnable = c1(doRelu);
    auto reluTypeAttr = ActivationTypeAttr::get(rewriter.getContext(), actType);

    // 9. Create ComputeRunOp
    rewriter.replaceOpWithNewOp<ComputeRunOp>(op, opTypeAttr, dataflowModeAttr,
        accoutDestAttr, vIntType,

        inputAMemRef, inputBMemRef,
        psumMemRefForOp,
        outputMemRef,

        vPadT, vPadB, vPadL, vPadR, vPadMode,

        vWeightShapeM1, vWeightStrideM1, vWeightDilationM1, 
        vIsGroup,

        vInAColM1, vInARowM1, vInAStride,
         vInBColM1, vInBRowM1, vInBStride,

        vBiasPsumWidth, vBiasPsumHeight, vBiasPsumStride,

        vOutputStride,

        vDoAccum, 
        vReluEnable, reluTypeAttr,
        vAccBias, 

        vOutZp, vQuantScale, vQuantShift, 
        vInAZp, vInBZp);

    return success();
  }
};

} // namespace

void npux::populateLinalgConvToNpuxPattern(RewritePatternSet &patterns) {
  patterns.add<LinalgComputeToNpuxPattern>(patterns.getContext());
}