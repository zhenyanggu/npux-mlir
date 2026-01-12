//================================================
//src/Conversion/NpuToLLVM/ConvertNpuxToLLVM.cpp
//this file declares npu to llvm conversion patterns
//which will be used in convert-krnl-to-llvm pass
//================================================



#include "src/Conversion/NpuToLLVM/ConvertNpuxToLLVM.hpp"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h" // 通常需要这个来支持 ConvertOpToLLVMPattern

using namespace mlir;
using namespace npux;

// ==========================================
// 辅助函数
// ==========================================

static FlatSymbolRefAttr getOrInsertExternFunc(PatternRewriter &rewriter, ModuleOp module,
                                               StringRef funcName, Type resultType,
                                               ArrayRef<Type> argTypes, bool isVarArg = false) {
  auto context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    return SymbolRefAttr::get(context, funcName);
  }

  auto llvmFnType = LLVM::LLVMFunctionType::get(resultType, argTypes, isVarArg);

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, llvmFnType);

  return SymbolRefAttr::get(context, funcName);
}

Value getFlatPtrFromMemRef(Location loc, Value memrefDescVal, 
                           Type elemType, ConversionPatternRewriter &rewriter) {
  MemRefDescriptor desc(memrefDescVal);
  Value alignedPtr = desc.alignedPtr(rewriter, loc);
  Value offset = desc.offset(rewriter, loc);
  
  // GEP i8* base + offset
  Value finalPtr = rewriter.create<LLVM::GEPOp>(
      loc, 
      alignedPtr.getType(), 
      rewriter.getI8Type(), 
      alignedPtr,           
      ArrayRef<Value>({offset}) 
  );

  return finalPtr;
}

Value getSramAddress(Location loc, Value sramMemRef, ConversionPatternRewriter &rewriter) {
  auto allocOp = sramMemRef.getDefiningOp<memref::AllocOp>();
  if (!allocOp || !allocOp->hasAttr("npu.offset")) {
    // 这里的错误处理最好加上，防止空指针崩溃
    return nullptr;
  }
  int32_t offsetVal = allocOp->getAttrOfType<IntegerAttr>("npu.offset").getInt();
  return rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(offsetVal));
}

// ==========================================
// 1. Lifecycle (Init/Destroy)
// ==========================================

class NpuxInitLowering : public ConvertOpToLLVMPattern<npux::InitOp> {
public:
  using ConvertOpToLLVMPattern<npux::InitOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(npux::InitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto i32Type = rewriter.getI32Type();
    
    // Init 返回 int，所以这里 CallOp 需要返回 i32Type，这是对的
    auto fnRef = getOrInsertExternFunc(rewriter, module, "npu_init", i32Type, {});
    auto callOp = rewriter.create<LLVM::CallOp>(op.getLoc(), i32Type, fnRef, ValueRange{});
    
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

class NpuxDestroyLowering : public ConvertOpToLLVMPattern<npux::DestroyOp> {
public:
  using ConvertOpToLLVMPattern<npux::DestroyOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(npux::DestroyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(getContext());
    
    // 函数声明需要 voidType
    auto fnRef = getOrInsertExternFunc(rewriter, module, "npu_destroy", voidType, {});
    
    // 【修复】CallOp 不能有 Result，传 TypeRange{}
    rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{}, fnRef, ValueRange{});
    
    rewriter.eraseOp(op);
    return success();
  }
};

// ==========================================
// 2. Memory (Alloc/Free)
// ==========================================

class NpuxAllocLowering : public ConvertOpToLLVMPattern<npux::AllocOp> {
public:
  using ConvertOpToLLVMPattern<npux::AllocOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(npux::AllocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    MemRefType memRefType = op.getType();
    auto loc = op.getLoc();
    auto i64Type = rewriter.getI64Type();

    if (!memRefType.hasStaticShape()) return failure(); 
    int64_t numElements = memRefType.getNumElements();
    int64_t elementSize = memRefType.getElementTypeBitWidth() / 8;
    if (elementSize == 0) elementSize = 1;
    int64_t totalBytes = numElements * elementSize;

    Value sizeVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(totalBytes));

    auto voidPtrType = LLVM::LLVMPointerType::get(getContext());
    // Alloc 返回 ptr，所以 CallOp 需要 result
    auto fnRef = getOrInsertExternFunc(rewriter, op->getParentOfType<ModuleOp>(), 
                                       "npu_mem_alloc", voidPtrType, {i64Type});
    
    auto callOp = rewriter.create<LLVM::CallOp>(op.getLoc(), voidPtrType, fnRef, ValueRange{sizeVal});
    Value rawPtr = callOp.getResult();

    auto targetType = getTypeConverter()->convertType(memRefType);
    if (!targetType) return failure();

    Value undefStruct = rewriter.create<LLVM::UndefOp>(loc, targetType);
    MemRefDescriptor desc(undefStruct);
    desc.setAllocatedPtr(rewriter, loc, rawPtr);
    desc.setAlignedPtr(rewriter, loc, rawPtr);
    
    Value zero = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(0));
    desc.setOffset(rewriter, loc, zero);

    for (int i = 0; i < memRefType.getRank(); ++i) {
      desc.setSize(rewriter, loc, i, rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(memRefType.getDimSize(i))));
      
      int64_t stride = 1;
      for(int j=i+1; j<memRefType.getRank(); ++j) stride *= memRefType.getDimSize(j);
      desc.setStride(rewriter, loc, i, rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(stride)));
    }

    rewriter.replaceOp(op, {desc}); 
    return success();
  }
};

class NpuxFreeLowering : public ConvertOpToLLVMPattern<npux::FreeOp> {
public:
  using ConvertOpToLLVMPattern<npux::FreeOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(npux::FreeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(getContext());
    auto voidPtrType = LLVM::LLVMPointerType::get(getContext());

    // 函数声明需要 voidType
    auto fnRef = getOrInsertExternFunc(rewriter, module, "npu_mem_free", voidType, {voidPtrType});
    
    MemRefDescriptor desc(adaptor.getMemref());
    Value ptr = desc.allocatedPtr(rewriter, op.getLoc());

    // 【修复】CallOp 传 TypeRange{}
    rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{}, fnRef, ValueRange{ptr});
    rewriter.eraseOp(op);
    return success();
  }
};

// ==========================================
// 3. Kernel (DMA/SFU)
// ==========================================

class NpuxDmaMvinLowering : public ConvertOpToLLVMPattern<DmaMvinOp> {
public:
  using ConvertOpToLLVMPattern<DmaMvinOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(DmaMvinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto dramMemRefType = cast<MemRefType>(op.getHostPtr().getType());
    Value hostPtr = getFlatPtrFromMemRef(loc, adaptor.getHostPtr(), 
                                         dramMemRefType.getElementType(), rewriter);

    Value sramMemRef = op.getSramMemref();
    Value sramAddr = getSramAddress(loc, sramMemRef, rewriter);
    if (!sramAddr) return failure();

    SmallVector<Value> args;
    args.push_back(hostPtr);   
    args.push_back(sramAddr); 
    args.push_back(adaptor.getColNum());
    args.push_back(adaptor.getRowNum());
    args.push_back(adaptor.getSramStride());
    args.push_back(adaptor.getDramStride());
    args.push_back(adaptor.getPrecision());
    args.push_back(adaptor.getInputType());
    args.push_back(adaptor.getDest());
    args.push_back(adaptor.getIsQuant());
    args.push_back(adaptor.getQuantZero());
    args.push_back(adaptor.getQuantScale());
    args.push_back(adaptor.getQuantShift());

    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(getContext());
    
    SmallVector<Type> argTypes;
    for(auto v : args) argTypes.push_back(v.getType());

    // 函数声明需要 voidType
    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(rewriter, module, "npu_dma_mvin", voidType, argTypes);
    
    // 【修复】CallOp 传 TypeRange{}
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnRef, args);
    
    return success();
  }
};

class NpuxSramAllocLowering : public ConvertOpToLLVMPattern<memref::AllocOp> {
public:
  using ConvertOpToLLVMPattern<memref::AllocOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getType().getMemorySpaceAsInt() != 2) return failure();
    for (Operation *user : op->getUsers()) {
      if (!isa<memref::DeallocOp>(user)) return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class NpuxSramDeallocLowering : public ConvertOpToLLVMPattern<memref::DeallocOp> {
public:
  using ConvertOpToLLVMPattern<memref::DeallocOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto type = cast<MemRefType>(op.getMemref().getType());
    if (type.getMemorySpaceAsInt() != 2) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

class NpuxSfuRunLowering : public ConvertOpToLLVMPattern<SfuRunOp> {
public:
  using ConvertOpToLLVMPattern<SfuRunOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(SfuRunOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value inSramAddr = getSramAddress(loc, op.getInputSramMemref(), rewriter);
    Value outSramAddr = getSramAddress(loc, op.getOutputSramMemref(), rewriter);
    if (!inSramAddr || !outSramAddr) return failure();

    SmallVector<Value> args;
    uint32_t opTypeVal = static_cast<uint32_t>(op.getOpType());
    args.push_back(rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(opTypeVal)));
    args.push_back(adaptor.getIntType());
    args.push_back(adaptor.getIsQuant());
    args.push_back(inSramAddr); 
    args.push_back(adaptor.getInputColNum());
    args.push_back(adaptor.getInputRowNum());
    args.push_back(outSramAddr); 
    args.push_back(adaptor.getInputZeropoint());
    args.push_back(adaptor.getOutputZeropoint());
    args.push_back(adaptor.getInputScale());
    args.push_back(adaptor.getInputScaleShift());
    args.push_back(adaptor.getOutputScale());
    args.push_back(adaptor.getOutputScaleShift());

    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(getContext());
    SmallVector<Type> argTypes;
    for(auto v : args) argTypes.push_back(v.getType());

    // 函数声明需要 voidType
    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(rewriter, module, "npu_sfu_run", voidType, argTypes);
    
    // 【修复】CallOp 传 TypeRange{}
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnRef, args);

    return success();
  }
};

class NpuxDmaMvoutLowering : public ConvertOpToLLVMPattern<DmaMvoutOp> {
public:
  using ConvertOpToLLVMPattern<DmaMvoutOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(DmaMvoutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto dramMemRefType = cast<MemRefType>(op.getHostPtr().getType());
    Value hostPtr = getFlatPtrFromMemRef(loc, adaptor.getHostPtr(), 
                                         dramMemRefType.getElementType(), rewriter);

    Value sramMemRef = op.getSramMemref();
    Value sramAddr = getSramAddress(loc, sramMemRef, rewriter);
    if (!sramAddr) return failure();

    SmallVector<Value> args;
    args.push_back(hostPtr);   
    args.push_back(sramAddr);  
    args.push_back(adaptor.getColNum());
    args.push_back(adaptor.getRowNum());
    args.push_back(adaptor.getSramStride());
    args.push_back(adaptor.getDramStride());
    args.push_back(adaptor.getPrecision());
    args.push_back(adaptor.getOutputType()); 
    args.push_back(adaptor.getSource());     
    args.push_back(adaptor.getIsQuant());
    args.push_back(adaptor.getQuantZero());
    args.push_back(adaptor.getQuantScale());
    args.push_back(adaptor.getQuantShift());

    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(getContext());
    SmallVector<Type> argTypes;
    for(auto v : args) argTypes.push_back(v.getType());

    // 函数声明需要 voidType
    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(rewriter, module, "npu_dma_mvout", voidType, argTypes);
    
    // 【修复】CallOp 传 TypeRange{}
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnRef, args);
    
    return success();
  }
};

void npux::populateNpuxToLLVMConversionPatterns(RewritePatternSet &patterns, 
                                                LLVMTypeConverter &typeConverter) {
  patterns.add<
        NpuxInitLowering,
        NpuxDestroyLowering,
        NpuxAllocLowering,
        NpuxFreeLowering,
        NpuxDmaMvinLowering,
        NpuxSramAllocLowering,
        NpuxSramDeallocLowering,
        NpuxSfuRunLowering,
        NpuxDmaMvoutLowering
  >(typeConverter);
}