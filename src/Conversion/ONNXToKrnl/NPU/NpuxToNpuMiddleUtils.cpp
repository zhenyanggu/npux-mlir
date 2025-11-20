
#include "src/Conversion/ONNXToKrnl/NPU/NpuxToNpuMiddleUtils.h"






void npu_middle::populateLoweringNpuxTestPatterns(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
    mlir::MLIRContext *context) {}




void npu_middle::populateLoweringNpuxToNpuMiddlePatterns(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
    mlir::MLIRContext *context) {
  populateLoweringNpuxTestPatterns(patterns, typeConverter, context);
}
