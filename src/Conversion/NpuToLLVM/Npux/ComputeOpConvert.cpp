//=======================================
// src/Conversion/NpuToLLVM/ComputeOpConvert.cpp
// This file implements convert linalg conv op
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

class LinalgConvToNpuxPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    // 1. 检查 Library Call 是否为 "npu_conv"
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr || libCallAttr.getValue() != "npu_conv") {
      return failure();
    }

    Location loc = op.getLoc();

    // 2. 获取操作数
    if (op.getInputs().size() < 2 || op.getOutputs().size() != 1) {
       return failure();
    }

    Value inputMemRef = op.getInputs()[0];
    Value weightMemRef = op.getInputs()[1];
    Value outputMemRef = op.getOutputs()[0];
    
    Value biasMemRef;
    if (op.getInputs().size() >= 3) {
        biasMemRef = op.getInputs()[2];
    } else {
        return failure(); 
    }

    // 3. 检查 Memory Space
    auto checkSram = [&](Value v) {
        auto type = mlir::dyn_cast<MemRefType>(v.getType());
        return type && type.getMemorySpaceAsInt() == 2;
    };

    if (!checkSram(inputMemRef) || !checkSram(outputMemRef)) {
        return failure();
    }

    // 4. 解析 Dimensions & Strides
    auto inType = mlir::cast<MemRefType>(inputMemRef.getType());
    auto outType = mlir::cast<MemRefType>(outputMemRef.getType());
    
    ArrayRef<int64_t> inShape = inType.getShape(); 
    ArrayRef<int64_t> outShape = outType.getShape(); 
    
    int64_t height = inShape[2];
    int64_t width  = inShape[3];
    int64_t cin_blk = inShape[1];
    int64_t cout_blk = outShape[1];
    int64_t tile_size = inShape[4]; 

    int64_t c_in_total = cin_blk * tile_size;
    int64_t c_out_total = cout_blk * tile_size;

    int64_t out_height = outShape[2];
    int64_t out_width = outShape[3];

    Value vDimH = rewriter.create<arith::ConstantIntOp>(loc, height, 32);
    Value vDimW = rewriter.create<arith::ConstantIntOp>(loc, width, 32);
    Value vDimCin = rewriter.create<arith::ConstantIntOp>(loc, c_in_total, 32);
    Value vDimCout = rewriter.create<arith::ConstantIntOp>(loc, c_out_total, 32);

    Value vOutH = rewriter.create<arith::ConstantIntOp>(loc, out_height, 32);
    Value vOutW = rewriter.create<arith::ConstantIntOp>(loc, out_width, 32);

    Value vStrideInA = rewriter.create<arith::ConstantIntOp>(loc, 0, 32); 
    Value vStrideInB = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value vStrideBias = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value vStrideOut = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

    // 5. 解析 Attributes (核心修复点)
    // -------------------------------------------------------------------------
    // [FIX] 使用 getValue().getSExtValue() 替代 getInt() 以支持 si64/ui64 类型
    // -------------------------------------------------------------------------
    auto getIntAttr = [&](StringRef name, int64_t defaultVal) -> int64_t {
        if (auto attr = op->getAttrOfType<IntegerAttr>(name)) {
            return attr.getValue().getSExtValue(); // SAFE
        }
        return defaultVal;
    };

    auto getArrayAttr = [&](StringRef name, int idx) -> int64_t {
        if (auto attr = op->getAttrOfType<ArrayAttr>(name)) {
            if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr[idx])) {
                return intAttr.getValue().getSExtValue(); // SAFE
            }
        }
        return 1;
    };

    // Kernel Shape
    int64_t kernel_h = getArrayAttr("kernel_shape", 0);
    Value vKernelSize = rewriter.create<arith::ConstantIntOp>(loc, kernel_h, 32);

    // Strides
    int64_t stride_h = getArrayAttr("strides", 0);
    Value vStride = rewriter.create<arith::ConstantIntOp>(loc, stride_h, 32);

    // Dilations
    int64_t dilation_h = getArrayAttr("dilations", 0);
    Value vDilation = rewriter.create<arith::ConstantIntOp>(loc, dilation_h, 32);

    // Pads
    int64_t pad_top = 0, pad_left = 0, pad_bottom = 0, pad_right = 0;
    if (auto attr = op->getAttrOfType<ArrayAttr>("pads")) {
        if (attr.size() == 4) {
            // [FIX] 使用 getValue().getSExtValue()
            pad_top = mlir::cast<IntegerAttr>(attr[0]).getValue().getSExtValue();
            pad_left = mlir::cast<IntegerAttr>(attr[1]).getValue().getSExtValue();
            pad_bottom = mlir::cast<IntegerAttr>(attr[2]).getValue().getSExtValue();
            pad_right = mlir::cast<IntegerAttr>(attr[3]).getValue().getSExtValue();
        }
    }
    Value vPadTop = rewriter.create<arith::ConstantIntOp>(loc, pad_top, 32);
    Value vPadBottom = rewriter.create<arith::ConstantIntOp>(loc, pad_bottom, 32);
    Value vPadLeft = rewriter.create<arith::ConstantIntOp>(loc, pad_left, 32);
    Value vPadRight = rewriter.create<arith::ConstantIntOp>(loc, pad_right, 32);

    // Group
    int64_t group = getIntAttr("group", 1);
    Value vIsGroup = rewriter.create<arith::ConstantIntOp>(loc, group > 1, 1);

    // 6. 解析 Quantization Params
    double inScale = 1.0;
    if (auto attr = op->getAttrOfType<FloatAttr>("in_scale")) inScale = attr.getValueAsDouble();
    
    double outScaleTarget = 1.0;
    if (auto attr = op->getAttrOfType<FloatAttr>("out_scale")) outScaleTarget = attr.getValueAsDouble();
    
    double realMultiplier = inScale / outScaleTarget;
    auto quantParams = getFixedPointParams(realMultiplier);

    Value vQuantScale = rewriter.create<arith::ConstantIntOp>(loc, quantParams.multiplier, 32);
    Value vQuantShift = rewriter.create<arith::ConstantIntOp>(loc, quantParams.shift, 32);

    // Zero Points ([FIX] 同样使用 safe get)
    int64_t inZp = getIntAttr("in_zp", 0);
    Value vInAZp = rewriter.create<arith::ConstantIntOp>(loc, inZp, 32);
    Value vInBZp = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

    int64_t outZp = getIntAttr("out_zp", 0);
    Value vOutZp = rewriter.create<arith::ConstantIntOp>(loc, outZp, 32);

    // 7. 其他配置
    auto opTypeAttr = ComputeOpTypeAttr::get(rewriter.getContext(), ComputeOpType::conv);
    Value vPrecision = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value vIsQuant = rewriter.create<arith::ConstantIntOp>(loc, true, 1);
    Value vDoAccum = rewriter.create<arith::ConstantIntOp>(loc, false, 1); 
    Value vDoRelu = rewriter.create<arith::ConstantIntOp>(loc, false, 1); 
    auto reluTypeAttr = ActivationTypeAttr::get(rewriter.getContext(), ActivationType::relu);

    // 8. Create ComputeRunOp
    rewriter.replaceOpWithNewOp<ComputeRunOp>(op,
        opTypeAttr,       
        vPrecision,      
        vIsQuant,         
        inputMemRef,      
        weightMemRef,     
        biasMemRef,      
        outputMemRef,     
        vDimH, vDimW, vDimCin, vDimCout, 
        vStrideInA, vStrideInB, vStrideBias, vStrideOut, 
        vKernelSize, vStride, vDilation, 
        vPadTop, vPadBottom, vPadLeft, vPadRight, 
        vIsGroup, 
        vOutW, vOutH, 
        vDoAccum, vDoRelu, reluTypeAttr, 
        vInAZp, vInBZp, vOutZp, vQuantScale, vQuantShift 
    );

    return success();
  }
};

} // namespace

void npux::populateLinalgConvToNpuxPattern(RewritePatternSet &patterns) {
  patterns.add<LinalgConvToNpuxPattern>(patterns.getContext());
}