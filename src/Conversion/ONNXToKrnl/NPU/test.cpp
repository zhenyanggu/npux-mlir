#include "src/Conversion/ONNXToKrnl/NPU/NpuxToNpuMiddleUtils.h"

using namespace mlir;

namespace npu_middle {

struct npux_mvinOpLowering : public OpConversionPattern<npux::mvinOp> {
  npux_mvinOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(npux::mvinOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    if (!convertedType)
      return op.emitError("Failed to convert type");

    Value input = adaptor.getInput();
    StringRef memspace = op.getMemspace();

    npu_middle::NpuMiddleBuilder create(rewriter, loc);
    Value result = create.mvin(input, memspace);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct npux_mvoutOpLowering : public OpConversionPattern<npux::mvoutOp> {
  npux_mvoutOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(npux::mvoutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    if (!convertedType)
      return op.emitError("Failed to convert type");

    Value input = adaptor.getInput();
    StringRef memspace = op.getMemspace();

    npu_middle::NpuMiddleBuilder create(rewriter, loc);
    Value result = create.mvout(input, memspace);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct npux_geluOpLowering : public OpConversionPattern<npux::geluOp> {
  npux_geluOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(npux::geluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    if (!convertedType)
      return op.emitError("Failed to convert type");

    Value input = adaptor.getInput();

    npu_middle::NpuMiddleBuilder create(rewriter, loc);
    Value result = create.gelu(input);

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace npu_middle

void npu_middle::populateLoweringNpuxTestPatterns(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
    mlir::MLIRContext *ctx) {
  patterns.insert<npu_middle::npux_mvinOpLowering,
      npu_middle::npux_mvoutOpLowering, npu_middle::npux_geluOpLowering>(
      typeConverter, ctx);
}

void npu_middle::populateLoweringNpuxToNpuMiddlePatterns(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
    mlir::MLIRContext *ctx) {
  populateLoweringNpuxTestPatterns(patterns, typeConverter, ctx);
}
