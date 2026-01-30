//=======================================
// src/Conversion/NpuToLLVM/ComputeOpConvert.cpp
// This file implements convert linalg generic op (conv/gemm/matmul)
// to custom npux compute_run op
//=======================================
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "src/Dialect/Npux/NpuxOps.hpp"
#include "src/Conversion/NpuToLLVM/NpuxConversionHelper.hpp"

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
  if (std::abs(scale) < 1e-8) return {0, 0};
  int exponent;
  double mantissa = std::frexp(scale, &exponent);
  double mantissa_scaled = std::round(mantissa * 32768.0);
  if (mantissa_scaled >= 32768.0) {
    mantissa_scaled /= 2.0;
    exponent += 1;
  }
  return {
    static_cast<int16_t>(mantissa_scaled),
    static_cast<int16_t>(exponent - 15)
  };
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

int64_t getArrayAttr(Operation *op, StringRef name, int idx, int64_t defaultVal = 1) {
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

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    // 1. Check Library Call Name
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr) return failure();
    
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

    // 2. Get Operands (Buffers)
    if (op.getInputs().size() < 2 || op.getOutputs().size() != 1) {
       return failure();
    }

    Value inputAMemRef = op.getInputs()[0];
    Value inputBMemRef = op.getInputs()[1]; // Weight / RHS
    Value outputMemRef = op.getOutputs()[0];

    // 3. Handle Optional Bias Input
    // Strategy: Pass the 3rd input if it exists. 
    // Flags (doAccum/accBias) are initialized to false and will be set by a later pass.
    Value biasMemRef; 
    if (op.getInputs().size() >= 3) {
        biasMemRef = op.getInputs()[2];
    }

    // 4. Memory Space Validation & Destination Inference
    auto checkSpace = [&](Value v, int expectedSpace) {
        if (!v) return true; // Skip null values
        auto type = mlir::dyn_cast<MemRefType>(v.getType());
        return type && type.getMemorySpaceAsInt() == expectedSpace;
    };

    if (!checkSpace(inputAMemRef, 2)) return failure(); // Input A -> SRAM
    if (!checkSpace(inputBMemRef, 2)) return failure(); // Input B -> SRAM
    if (!checkSpace(biasMemRef, 3)) return failure();   // Bias -> ACC (Space 3)
    
    // Determine accout_dest based on Output Memory Space
    // Space 2 = SRAM (SPM), Space 3 = ACC
    auto outType = mlir::cast<MemRefType>(outputMemRef.getType());
    int outSpace = outType.getMemorySpaceAsInt();
    
    AccoutDest accDest;
    if (outSpace == 3) {
        accDest = AccoutDest::acc;
    } else if (outSpace == 2) {
        accDest = AccoutDest::spm;
    } else {
        return failure(); // Output must be in SRAM or ACC
    }

    // 5. Parse Geometry (Shapes & Strides)
    // ---------------------------------------------------------
    auto inAType = mlir::cast<MemRefType>(inputAMemRef.getType());
    auto inBType = mlir::cast<MemRefType>(inputBMemRef.getType());

    ArrayRef<int64_t> inAShape = inAType.getShape();
    ArrayRef<int64_t> inBShape = inBType.getShape();
    ArrayRef<int64_t> outShape = outType.getShape();

    // Initialize defaults
    int64_t a_col = 1, a_row = 1, a_stride = 0;
    int64_t b_col = 1, b_row = 1, b_stride = 0;
    int64_t out_width = 1, out_height = 1, out_stride = 0; // Will be reused for biaspsum

    // Conv specific defaults
    int64_t kernel_sz = 1, stride_val = 1, dilation_val = 1;
    int64_t pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    int64_t pad_mode_val = 0;
    bool is_group = false;

    if (opType == ComputeOpType::conv) {
        // === CONV Shape Parsing ===
        // Mapping depends on your layout (NCHW vs NHWC). 
        // Assuming Logic: [N, C, H, W] or [N, H, W, C] -> extracting H/W
        // Based on previous context: inAShape[2]=H, inAShape[3]=W
        
        a_row = inAShape[2]; // H
        a_col = inAShape[3]; // W
        
        // Weight Geometry (Input B)
        b_row = inBShape[2]; 
        b_col = inBShape[3];

        // Output Geometry (reused for Loop Control)
        out_height = outShape[2];
        out_width = outShape[3];

        // Attributes
        kernel_sz = getArrayAttr(op, "kernel_shape", 0, 1);
        stride_val = getArrayAttr(op, "strides", 0, 1);
        dilation_val = getArrayAttr(op, "dilations", 0, 1);
        
        if (auto attr = op->getAttrOfType<ArrayAttr>("pads")) {
            if (attr.size() == 4) {
                // pad_t = mlir::cast<IntegerAttr>(attr[0]).getValue().getSExtValue();
                // pad_l = mlir::cast<IntegerAttr>(attr[1]).getValue().getSExtValue();
                // pad_b = mlir::cast<IntegerAttr>(attr[2]).getValue().getSExtValue();
                // pad_r = mlir::cast<IntegerAttr>(attr[3]).getValue().getSExtValue();
            }
        }
        pad_mode_val = getIntAttr(op, "pad_mode", 0);
        is_group = (getIntAttr(op, "group", 1) > 1);

    } else {
        // === GEMM / MATMUL Shape Parsing ===
        // M, K, N logic
        if (inAShape.size() >= 2 && outShape.size() >= 2) {
            int64_t M = inAShape[0];
            int64_t K = inAShape[1]; // Or inBShape[0]
            int64_t N = outShape[1];

            // Input A: [M, K]
            a_row = M; a_col = K;
            
            // Input B: [K, N] (assuming RHS is transposed or standard depending on impl)
            b_row = K; b_col = N;

            // Output: [M, N]
            out_height = M; out_width = N;
        }
    }

    // 6. Create Constants
    // ---------------------------------------------------------
    auto c32 = [&](int64_t v) { return rewriter.create<arith::ConstantIntOp>(loc, v, 32); };
    auto c8 = [&](int64_t v) { return rewriter.create<arith::ConstantIntOp>(loc, v, 8); };
    auto c1  = [&](bool v)    { return rewriter.create<arith::ConstantIntOp>(loc, v, 1); };

    // --- Operation Control ---
    auto opTypeAttr = ComputeOpTypeAttr::get(rewriter.getContext(), opType);
    auto dataflowModeAttr = DataflowModeAttr::get(rewriter.getContext(), DataflowMode::ws); // Default WS
    auto accoutDestAttr = AccoutDestAttr::get(rewriter.getContext(), accDest);
    
    Value vIntType = c8(0); // Default to INT8 (0)

    // --- Padding ---
    Value vPadT = c32(pad_t); Value vPadB = c32(pad_b);
    Value vPadL = c32(pad_l); Value vPadR = c32(pad_r);
    Value vPadMode = c32(pad_mode_val);

    // --- Weights & Inputs (Minus 1 Logic) ---
    // Note: Hardware registers for sizes usually expect Value-1.
    auto safe_m1 = [](int64_t v) { return v > 0 ? v - 1 : 0; };

    Value vWeightShapeM1    = c32(safe_m1(kernel_sz));
    Value vWeightStrideM1   = c32(safe_m1(stride_val));
    Value vWeightDilationM1 = c32(safe_m1(dilation_val));
    Value vIsGroup          = c1(is_group);

    Value vInAColM1  = c32(safe_m1(a_col));
    Value vInARowM1  = c32(safe_m1(a_row));
    Value vInAStride = c32(a_stride); // Stride usually raw

    Value vInBColM1  = c32(safe_m1(b_col));
    Value vInBRowM1  = c32(safe_m1(b_row));
    Value vInBStride = c32(b_stride);

    // --- Loop Control / Output (Reused Geometry) ---
    // Note: These define the execution loop limits. 
    // Usually raw width/height are passed here if API expects count, or -1 if 0-based index.
    // Based on API "biaspsum_width", assume raw value.
    Value vBiasPsumWidth  = c32(out_width);
    Value vBiasPsumHeight = c32(out_height);
    Value vBiasPsumStride = c32(0); // Placeholder for bias stride

    Value vOutputStride = c32(out_stride);

    // 7. Quantization Params
    // ---------------------------------------------------------
    double realMultiplier = 1.0;
    int64_t inZp = 0; int64_t wZp = 0; int64_t outZp = 0;

    double outScaleTarget = getFloatAttr(op, "out_scale", 1.0);
    outZp = getIntAttr(op, "out_zp", 0);

    if (opType == ComputeOpType::conv) {
        double inScale = getFloatAttr(op, "in_scale", 1.0);
        inZp = getIntAttr(op, "in_zp", 0);
        realMultiplier = inScale / outScaleTarget;
    } else {
        // GEMM / Matmul params
        inZp = getIntAttr(op, "lhs_zp", 0);
        wZp  = getIntAttr(op, "rhs_zp", 0);
        
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
    // ---------------------------------------------------------
    // Initialize Accumulate/Bias flags to False.
    // A subsequent compiler pass will analyze the graph and set these correctly.
    Value vDoAccum = c1(false);
    Value vAccBias = c1(false);

    // Activation
    bool doRelu = (getIntAttr(op, "do_relu", 0) != 0);
    int64_t reluTypeVal = getIntAttr(op, "relu_type", 0);
    ActivationType actType = static_cast<ActivationType>(reluTypeVal);
    if (reluTypeVal > 4) actType = ActivationType::relu;
    
    Value vReluEnable = c1(doRelu);
    auto reluTypeAttr = ActivationTypeAttr::get(rewriter.getContext(), actType);

    // 9. Create ComputeRunOp
    // ---------------------------------------------------------
    // Order MUST match the latest .td definition
    rewriter.replaceOpWithNewOp<ComputeRunOp>(op,
        opTypeAttr,
        dataflowModeAttr,
        accoutDestAttr,
        vIntType,
        
        inputAMemRef,
        inputBMemRef,
        biasMemRef,      // Optional Buffer (passed if exists)
        outputMemRef,

        vPadT, vPadB, vPadL, vPadR,
        vPadMode,

        vWeightShapeM1, vWeightStrideM1, vWeightDilationM1, vIsGroup,

        vInAColM1, vInARowM1, vInAStride,
        vInBColM1, vInBRowM1, vInBStride,

        vBiasPsumWidth, vBiasPsumHeight, vBiasPsumStride, // Reused Geometry

        vOutputStride,

        vDoAccum,        // Default False
        vReluEnable,
        reluTypeAttr,
        vAccBias,        // Default False

        vOutZp,          // Note: Output ZP first
        vQuantScale,
        vQuantShift,
        vInAZp,
        vInBZp
    );

    return success();
  }
};

} // namespace

void npux::populateLinalgConvToNpuxPattern(RewritePatternSet &patterns) {
  patterns.add<LinalgComputeToNpuxPattern>(patterns.getContext());
}