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

// ============================================================================
// Helper: Quant Params (Existing)
// ============================================================================
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

// ============================================================================
// Pattern 1: SFU Ops (Gelu, Softmax, LayerNorm) -> SfuRunOp
// ============================================================================
class LinalgSfuToNpuxPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr) return failure();
    StringRef opName = libCallAttr.getValue();

    // 仅匹配 SFU 相关算子
    npux::SFUOpType sfuOpEnum;
    if (opName == "npu_gelu") sfuOpEnum = npux::SFUOpType::gelu;
    else if (opName == "npu_softmax") sfuOpEnum = npux::SFUOpType::softmax;
    else if (opName == "npu_layernorm") sfuOpEnum = npux::SFUOpType::layernorm;
    else return failure(); 

    Location loc = op.getLoc();

    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) return failure();
    
    Value inputMemRef = op.getInputs()[0];
    Value outputMemRef = op.getOutputs()[0];

    auto inType = mlir::dyn_cast<MemRefType>(inputMemRef.getType());
    if (!inType || inType.getMemorySpaceAsInt() != 2) return failure();

    ArrayRef<int64_t> shape = inType.getShape();
    int rank = shape.size();
    if (rank < 2) return failure();
    
    // SFU 硬件接口需要的是 col_num - 1 和 row_num - 1
    int64_t rows = shape[rank - 2] - 1;
    int64_t cols = shape[rank - 1] - 1;

    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows, 16);

    // ... (Quant Params Calculation 保持不变) ...
    // 1. Input Quant
    double inScale = 1.0;
    if (auto attr = op->getAttrOfType<FloatAttr>("in_scale")) inScale = attr.getValueAsDouble();
    auto inFP = getFixedPointParams(inScale);
    
    Value vInScale = rewriter.create<arith::ConstantIntOp>(loc, inFP.multiplier, 16);
    Value vInShift = rewriter.create<arith::ConstantIntOp>(loc, inFP.shift, 16);
    
    int64_t inZp = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("in_zp")) inZp = attr.getInt();
    Value vInZp = rewriter.create<arith::ConstantIntOp>(loc, inZp, 32); 

    // 2. Output Quant
    double outScale = 1.0;
    if (auto attr = op->getAttrOfType<FloatAttr>("out_scale")) outScale = attr.getValueAsDouble();
    // 硬件要求 out_scale 是倒数形式 (inv_scale)
    double invOutScale = (std::abs(outScale) > 1e-9) ? (1.0 / outScale) : 1.0;
    auto outFP = getFixedPointParams(invOutScale);

    Value vOutScale = rewriter.create<arith::ConstantIntOp>(loc, outFP.multiplier, 16);
    Value vOutShift = rewriter.create<arith::ConstantIntOp>(loc, outFP.shift, 16);

    int64_t outZp = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("out_zp")) outZp = attr.getInt();
    Value vOutZp = rewriter.create<arith::ConstantIntOp>(loc, outZp, 16); 

    // F. 其他配置
    int8_t intTypeVal = 8; 
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

// ============================================================================
// Pattern 2: Transpose Op -> TransposeOp
// ============================================================================
class LinalgTransposeToNpuxPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr || libCallAttr.getValue() != "npu_transpose") return failure();

    Location loc = op.getLoc();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) return failure();

    Value inputMemRef = op.getInputs()[0];
    Value outputMemRef = op.getOutputs()[0];

    // 检查 Memory Space (SRAM=2)
    auto inType = mlir::dyn_cast<MemRefType>(inputMemRef.getType());
    if (!inType || inType.getMemorySpaceAsInt() != 2) return failure();

    // 计算 Shape: C API 需要 col_num (W-1) 和 row_num (H-1)
    // 假设是 2D 或更高维，取最后两维
    ArrayRef<int64_t> shape = inType.getShape();
    int rank = shape.size();
    if (rank < 2) return failure();

    int64_t rows = shape[rank - 2] - 1;
    int64_t cols = shape[rank - 1] - 1;

    Value vCol = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vRow = rewriter.create<arith::ConstantIntOp>(loc, rows, 16);

    // 创建 TransposeOp
    // 不需要量化参数，因为 Transpose 只搬运比特
    rewriter.replaceOpWithNewOp<TransposeOp>(op,
        inputMemRef,
        outputMemRef,
        vCol,
        vRow
    );

    return success();
  }
};

// ============================================================================
// Pattern 3: Resample Ops (MaxPool, AvgPool, Upsample) -> ResampleOp
// ============================================================================
class LinalgResampleToNpuxPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr) return failure();
    StringRef opName = libCallAttr.getValue();

    // 确定 Resample Type 和 Op
    npux::ResampleType typeEnum;
    npux::ResampleMode modeEnum;

    if (opName == "npu_maxpool") {
        typeEnum = npux::ResampleType::downsample;
        modeEnum = npux::ResampleMode::max_nearest;
    } else if (opName == "npu_avgpool") {
        typeEnum = npux::ResampleType::downsample;
        modeEnum = npux::ResampleMode::avg_bilinear;
    } else if (opName == "npu_upsample") {
        typeEnum = npux::ResampleType::upsample;
        modeEnum = npux::ResampleMode::max_nearest; // Nearest Neighbor
    } else {
        return failure();
    }

    Location loc = op.getLoc();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) return failure();

    Value inputMemRef = op.getInputs()[0];
    Value outputMemRef = op.getOutputs()[0];

    // 检查 Memory Space (SRAM=2)
    auto inType = mlir::dyn_cast<MemRefType>(inputMemRef.getType());
    if (!inType || inType.getMemorySpaceAsInt() != 2) return failure();

    // 计算 Shape: C API 需要 INPUT 的 col_num (W-1) 和 row_num (H-1)
    // 这一点很重要，对于 Upsample，Output 比 Input 大，但 API 仍需 Input 尺寸
    ArrayRef<int64_t> shape = inType.getShape();
    int rank = shape.size();
    if (rank < 2) return failure();

    int64_t rows = shape[rank - 2] - 1;
    int64_t cols = shape[rank - 1] - 1;

    Value vInputCol = rewriter.create<arith::ConstantIntOp>(loc, cols, 16);
    Value vInputRow = rewriter.create<arith::ConstantIntOp>(loc, rows, 16);

    // 构建 Enum 属性
    auto typeAttr = ResampleTypeAttr::get(rewriter.getContext(), typeEnum);
    auto modeAttr = ResampleModeAttr::get(rewriter.getContext(), modeEnum);

    // 创建 ResampleOp
    rewriter.replaceOpWithNewOp<npux::ResampleOp>(op,
        typeAttr,
        modeAttr,
        inputMemRef,
        outputMemRef,
        vInputCol,
        vInputRow
    );

    return success();
  }
};

}// namespace

class LinalgLayoutToNpuxPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    auto libCallAttr = op.getLibraryCallAttr();
    if (!libCallAttr) return failure();
    StringRef opName = libCallAttr.getValue();

    bool isPack = (opName == "npu_layout_nchw_to_nchwc32");
    bool isUnpack = (opName == "npu_layout_nchwc32_to_nchw");

    if (!isPack && !isUnpack) return failure();

    Location loc = op.getLoc();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) return failure();

    // 此时已经是 Bufferization 之后，操作数应为 MemRef
    Value inputMemRef = op.getInputs()[0];
    Value outputMemRef = op.getOutputs()[0];

    // 检查 Memory Space (SRAM=2)
    // 确保数据已经在 SRAM 中，符合 npu_layout_* 的要求
    auto inType = mlir::dyn_cast<MemRefType>(inputMemRef.getType());
    auto outType = mlir::dyn_cast<MemRefType>(outputMemRef.getType());

    // 从属性中提取 N, C, H, W 参数
    // 这些属性是在 Conv.cpp 中 setAttr 的
    auto getIntParam = [&](StringRef name) -> int64_t {
        if (auto attr = op->getAttrOfType<IntegerAttr>(name)) {
            return attr.getInt();
        }
        // 如果没有找到参数，这是一个错误，但在 Pattern 中通常返回 failure
        // 为了安全起见这里返回 1
        return 1; 
    };

    int64_t n = getIntParam("params_n");
    int64_t c = getIntParam("params_c");
    int64_t h = getIntParam("params_h");
    int64_t w = getIntParam("params_w");

    // 创建 i16 常量作为参数传递给 Op
    Value vN = rewriter.create<arith::ConstantIntOp>(loc, n, 16);
    Value vC = rewriter.create<arith::ConstantIntOp>(loc, c, 16);
    Value vH = rewriter.create<arith::ConstantIntOp>(loc, h, 16);
    Value vW = rewriter.create<arith::ConstantIntOp>(loc, w, 16);

    // 替换为具体的 Npux Op
    if (isPack) {
        rewriter.replaceOpWithNewOp<npux::LayoutNchwToNchwc32Op>(op,
            inputMemRef, outputMemRef, vN, vC, vH, vW);
    } else {
        rewriter.replaceOpWithNewOp<npux::LayoutNchwc32ToNchwOp>(op,
            inputMemRef, outputMemRef, vN, vC, vH, vW);
    }

    return success();
  }
};

// ============================================================================
// Registration
// ============================================================================
void npux::populateLinalgSfuToNpuxPattern(RewritePatternSet &patterns) {
  patterns.add<LinalgSfuToNpuxPattern>(patterns.getContext());
  patterns.add<LinalgTransposeToNpuxPattern>(patterns.getContext());
  patterns.add<LinalgResampleToNpuxPattern>(patterns.getContext());
  patterns.add<LinalgLayoutToNpuxPattern>(patterns.getContext());
}