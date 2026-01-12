//=======================================
//src/Conversion/NpuToLLVM/SfuOpConvert.cpp
//this file implements convert linalg sfu op
// to custom npux sfu run op
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

class LinalgSfuToNpuxPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr) return failure();
    StringRef opName = libCallAttr.getValue();

    if (opName == "npu_transpose") return failure();

    npux::SFUOpType sfuOpEnum;
    if (opName == "npu_gelu") sfuOpEnum = npux::SFUOpType::gelu;
    else if (opName == "npu_softmax") sfuOpEnum = npux::SFUOpType::softmax;
    else if (opName == "npu_layernorm") sfuOpEnum = npux::SFUOpType::layernorm;
    else return failure(); 

    Location loc = op.getLoc();

    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) {
       return failure();
    }
    
    Value inputMemRef = op.getInputs()[0];
    Value outputMemRef = op.getOutputs()[0];

    auto inType = mlir::dyn_cast<MemRefType>(inputMemRef.getType());
    if (!inType || inType.getMemorySpaceAsInt() != 2) {
       return failure();
    }


    ArrayRef<int64_t> shape = inType.getShape();
    int rank = shape.size();
    if (rank < 2) return failure();
    
    int64_t rows = shape[rank - 2];
    int64_t cols = shape[rank - 1];

    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows, 16);


    // 1. Input Quant
    double inScale = 1.0;
    if (auto attr = op->getAttrOfType<FloatAttr>("in_scale")) inScale = attr.getValueAsDouble();
    auto inFP = getFixedPointParams(inScale);
    
    Value vInScale = rewriter.create<arith::ConstantIntOp>(loc, inFP.multiplier, 16);
    Value vInShift = rewriter.create<arith::ConstantIntOp>(loc, inFP.shift, 16);
    
    int64_t inZp = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("in_zp")) inZp = attr.getInt();
    Value vInZp = rewriter.create<arith::ConstantIntOp>(loc, inZp, 32); // Input ZP use I32

    // 2. Output Quant
    double outScale = 1.0;
    if (auto attr = op->getAttrOfType<FloatAttr>("out_scale")) outScale = attr.getValueAsDouble();
    auto outFP = getFixedPointParams(outScale);

    Value vOutScale = rewriter.create<arith::ConstantIntOp>(loc, outFP.multiplier, 16);
    Value vOutShift = rewriter.create<arith::ConstantIntOp>(loc, outFP.shift, 16);

    int64_t outZp = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("out_zp")) outZp = attr.getInt();

    Value vOutZp = rewriter.create<arith::ConstantIntOp>(loc, outZp, 16); 

    // F. 其他配置
    // Int Type: 检查 MemRef ElementType
    int8_t intTypeVal = 8; // 默认
    if (inType.getElementType().isInteger(16)) intTypeVal = 16;
    else if (inType.getElementType().isInteger(32)) intTypeVal = 32;
    Value vIntType = rewriter.create<arith::ConstantIntOp>(loc, intTypeVal, 8);

    bool isQuant = op->hasAttr("in_zp");
    Value vIsQuant = rewriter.create<arith::ConstantIntOp>(loc, isQuant, 1);

    auto opTypeAttr = SFUOpTypeAttr::get(rewriter.getContext(), sfuOpEnum);

    rewriter.replaceOpWithNewOp<SfuRunOp>(op,
        opTypeAttr,
        vIntType,
        vIsQuant,
        inputMemRef, 
        vCol, 
        vRow,
        outputMemRef, 
        vInZp,
        vOutZp,
        vInScale,
        vInShift,
        vOutScale,
        vOutShift
    );

    return success();
  }
};


}// namespace

void npux::populateLinalgSfuToNpuxPattern(RewritePatternSet &patterns) {
  patterns.add<LinalgSfuToNpuxPattern>(patterns.getContext());
}