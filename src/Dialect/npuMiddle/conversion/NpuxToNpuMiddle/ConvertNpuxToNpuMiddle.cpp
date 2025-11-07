
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

#include "src/Dialect/npux/ir/npuxOps.h"
#include "src/Dialect/npuMiddle/ir/npuMiddleOps.h"

using namespace mlir;
using namespace npux;

namespace npu_middle {

// ---------------------- Example: type converter ---------------------------
struct NOPXTypeConverter : public TypeConverter {
  NOPXTypeConverter() {
    // 默认把 RankedTensorType 保留为 RankedTensorType（示例）
    addConversion([](Type t) { return t; });

    // 若需要将 tensor -> memref，可以添加 conversion：
    // addConversion([](RankedTensorType t) -> Optional<Type> {
    //    return MemRefType::get(t.getShape(), t.getElementType());
    // });

    // function signature conversion helper
    addArgumentMaterialization([](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return nullptr; // implement if needed
    });
  }
};

// ---------------------- Example pattern ---------------------------
// 假设 NOPX::AddOp 要被降低为 npu_middle::BinaryOp (示意)
class NOPXAddLowering : public OpConversionPattern<nopx::AddOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(nopx::AddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // 获取转换后的 operands（TypeConverter 已替换类型）
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // 举例：创建一个 npu_middle 的二元加法 op
    // 你要根据 npu_middle 的 op 定义改这里
    auto resultType = typeConverter->convertType(op.getResult().getType()).cast<Type>();
    rewriter.replaceOpWithNewOp<npu_middle::AddOp>(op, resultType, lhs, rhs);
    return success();
  }
};

// ---------------------- populate patterns ---------------------------
void populateNOPXToNpuMiddlePatterns(RewritePatternSet &patterns,
                                     NOPXTypeConverter &typeConverter,
                                     MLIRContext *ctx) {
  patterns.add<NOPXAddLowering>(typeConverter, ctx);
  // TODO: 把所有 NOPX op 的 lowering 都加上
}

//===----------------------------------------------------------------------===//
// NPUX Dialect to NpuMiddle Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering of NPUX operations to the npu_middle dialect.
struct NPUXToNpuMiddlePass
    : public PassWrapper<NPUXToNpuMiddlePass, OperationPass<ModuleOp>> {

  // 1. 添加 Pass 身份宏
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NPUXToNpuMiddlePass)

  // 2. 添加 Pass 名称和描述
  StringRef getArgument() const override { return "convert-npux-to-npu-middle"; }

  StringRef getDescription() const override {
    return "Lower NPUX dialect ops to npu_middle dialect.";
  }

  // 3. 添加构造函数 (可以简化，或者保留配置选项的结构)
  // 默认构造函数
  NPUXToNpuMiddlePass() = default;
  // 拷贝构造函数
  NPUXToNpuMiddlePass(const NPUXToNpuMiddlePass &pass)
      : PassWrapper<NPUXToNpuMiddlePass, OperationPass<ModuleOp>>() {}
  
  // 示例：如果需要配置选项，可以添加带参数的构造函数
  /*
  NPUXToNpuMiddlePass(bool enableOptimization) {
    // Below, need explicit assignment to enable implicit conversion of bool to
    // Option<bool>.
    this->enableOptimization = enableOptimization;
  }
  */

  // 4. 添加 runOnOperation 的声明 (它已经在您的原代码中定义)
  void runOnOperation() final;

// 5. 添加可配置选项 (示例，如果您的 npux-to-npu_middle 转换有需要的话)
public:
  /*
  Option<bool> emitIntermediateIR{*this, "emit-intermediate-ir",
      llvm::cl::desc(
          "Emit intermediate IR rather than lowering to the npu_middle dialect."),
      llvm::cl::init(false)};*/
  
  /*
  Option<bool> enableOptimization{*this, "enable-opt",
      llvm::cl::desc("Enable some optimization for npux lowering"), 
      llvm::cl::init(false)};
  */
};


// runOnOperation 的实现保持您原有的逻辑
void NPUXToNpuMiddlePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = &getContext();

  // Type converter
  NPUXTypeConverter typeConverter;

  // Conversion target
  ConversionTarget target(*ctx);
  // 标记目标方言为 legal
  target.addLegalDialect<npu_middle::NpuMiddleDialect>();
  // 标记 std/other needed dialects
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<tensor::TensorDialect>(); // 视需要

  // NPUX dialect illegal（确保都被转换）
  target.addIllegalDialect<npux::NPUXDialect>();

  // Dynamically legal func::FuncOp if signature types converted
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  
  // 如果需要用到 emitIntermediateIR 选项来保留一些 NPUX ops
  if (emitIntermediateIR) {
    //keep it in case i wanna do some debugging
  }

  // patterns
  RewritePatternSet patterns(ctx);
  patterns.add<ConvertFuncOpSignaturePattern>(/*optional*/) ; // see helper if needed
  populateNPUXToNpuMiddlePatterns(patterns, typeConverter, ctx);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

// 辅助函数保持您原有的逻辑
std::unique_ptr<Pass> createNPUXToNpuMiddlePass() {
  return std::make_unique<NPUXToNpuMiddlePass>();
}

/*
// 示例：如果需要创建带参数的 Pass，可以添加这个辅助函数
std::unique_ptr<Pass> createNPUXToNpuMiddlePass(bool enableOptimization) {
  return std::make_unique<NPUXToNpuMiddlePass>(enableOptimization);
}
*/

} // namespace npux::lowering (或者您自定义的命名空间)


// Registration (在 Pass 注册文件中)
static PassRegistration<npux::lowering::NPUXToNpuMiddlePass>
    pass("convert-npux-to-npu-middle",
             "Lower NPUX dialect ops to npu_middle dialect");