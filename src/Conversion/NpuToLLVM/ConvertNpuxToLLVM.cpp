//================================================
//src/Conversion/NpuToLLVM/ConvertNpuxToLLVM.cpp
//this file declares npu to llvm conversion patterns
//which will be used in convert-krnl-to-llvm pass
//================================================

#include "src/Conversion/NpuToLLVM/ConvertNpuxToLLVM.hpp"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h" 

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

// 【修复】修改 getSramAddress 以支持 npux::SramAllocOp
Value getSramAddress(Location loc, Value sramMemRef, ConversionPatternRewriter &rewriter) {
  // 现在定义 Op 是 npux::SramAllocOp
  auto allocOp = sramMemRef.getDefiningOp<npux::SramAllocOp>();
  
  // 增加对 Function Argument 的兼容性（虽然目前主要是 alloc，但为了健壮性）
  // 如果不是 AllocOp 定义的，可能没有 npu.offset 属性，这通常是逻辑错误
  if (!allocOp) {
      return nullptr;
  }

  if (!allocOp->hasAttr("npu.offset")) {
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
    
    auto fnRef = getOrInsertExternFunc(rewriter, module, "npu_destroy", voidType, {});
    
    rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{}, fnRef, ValueRange{});
    
    rewriter.eraseOp(op);
    return success();
  }
};

// ==========================================
// 2. Memory (Alloc/Free) - DRAM Only
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

    auto fnRef = getOrInsertExternFunc(rewriter, module, "npu_mem_free", voidType, {voidPtrType});
    
    MemRefDescriptor desc(adaptor.getMemref());
    Value ptr = desc.allocatedPtr(rewriter, op.getLoc());

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

    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(rewriter, module, "npu_dma_mvin", voidType, argTypes);
    
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnRef, args);
    
    return success();
  }
};

// 【修复】匹配 npux::SramAllocOp
class NpuxSramAllocLowering : public ConvertOpToLLVMPattern<npux::SramAllocOp> {
public:
  using ConvertOpToLLVMPattern<npux::SramAllocOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(npux::SramAllocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // SRAM Alloc 在 LLVM 层面不需要任何指令。
    // 它的 offset 已经被 Plan Pass 计算并在 use-site (如 mvin, sfu_run) 解析了。
    // 所以这里直接擦除即可。
    rewriter.eraseOp(op);
    return success();
  }
};

// 【修复】匹配 npux::SramFreeOp
class NpuxSramFreeLowering : public ConvertOpToLLVMPattern<npux::SramFreeOp> {
public:
  using ConvertOpToLLVMPattern<npux::SramFreeOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(npux::SramFreeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // SRAM Free 只是 Planner 的逻辑指令，生码时无需任何操作。
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

    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(rewriter, module, "npu_sfu_run", voidType, argTypes);
    
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

    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(rewriter, module, "npu_dma_mvout", voidType, argTypes);
    
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnRef, args);
    
    return success();
  }
};

class NpuxComputeRunLowering : public ConvertOpToLLVMPattern<ComputeRunOp> {
public:
  using ConvertOpToLLVMPattern<ComputeRunOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(ComputeRunOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto i32Type = rewriter.getI32Type();

    Value addrA = getSramAddress(loc, op.getInputA(), rewriter);
    Value addrB = getSramAddress(loc, op.getInputB(), rewriter);
    Value addrBias = getSramAddress(loc, op.getBias(), rewriter);
    Value addrOut = getSramAddress(loc, op.getOutput(), rewriter);

    if (!addrA || !addrB || !addrBias || !addrOut) {
      return rewriter.notifyMatchFailure(op, "Failed to resolve SRAM address for buffers");
    }

    auto getI1AsI32 = [&](Value val) -> Value {
      return rewriter.create<LLVM::ZExtOp>(loc, i32Type, val);
    };

    auto getEnumAsI32 = [&](auto enumAttr) -> Value {
      uint32_t val = static_cast<uint32_t>(enumAttr);
      return rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(val));
    };

    SmallVector<Value> args;

    // 4.1 Operation Control
    args.push_back(getEnumAsI32(op.getOpType()));      
    args.push_back(adaptor.getPrecision());            
    args.push_back(getI1AsI32(adaptor.getIsQuant()));  

    // 4.2 Addresses (SRAM Offsets)
    args.push_back(addrA);
    args.push_back(addrB);
    args.push_back(addrBias);
    args.push_back(addrOut);

    // 4.3 Dimensions
    args.push_back(adaptor.getDimH());
    args.push_back(adaptor.getDimW());
    args.push_back(adaptor.getDimCIn());
    args.push_back(adaptor.getDimCOut());

    // 4.4 Strides
    args.push_back(adaptor.getStrideInA());
    args.push_back(adaptor.getStrideInB());
    args.push_back(adaptor.getStrideBias());
    args.push_back(adaptor.getStrideOut());

    // 4.5 Convolution Specifics
    args.push_back(adaptor.getKernelSize());
    args.push_back(adaptor.getStride());
    args.push_back(adaptor.getDilation());
    args.push_back(adaptor.getPadTop());
    args.push_back(adaptor.getPadBottom());
    args.push_back(adaptor.getPadLeft());
    args.push_back(adaptor.getPadRight());
    args.push_back(getI1AsI32(adaptor.getIsGroupConv())); 

    // 4.6 Output Config
    args.push_back(adaptor.getOutWidth());
    args.push_back(adaptor.getOutHeight());
    args.push_back(getI1AsI32(adaptor.getDoAccumulate())); 
    args.push_back(getI1AsI32(adaptor.getDoRelu()));      
    args.push_back(getEnumAsI32(op.getReluType()));        

    // 4.7 Quantization Params
    args.push_back(adaptor.getInputAZp());
    args.push_back(adaptor.getInputBZp());
    args.push_back(adaptor.getOutputZp());
    args.push_back(adaptor.getQuantScale());
    args.push_back(adaptor.getQuantShift());

    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(getContext());
    
    SmallVector<Type> argTypes(args.size(), i32Type);

    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(
        rewriter, module, "npu_compute_run", voidType, argTypes);

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
        NpuxSramAllocLowering, // 【修复】注册新的 Pattern
        NpuxSramFreeLowering,  // 【修复】注册新的 Pattern
        NpuxSfuRunLowering,
        NpuxComputeRunLowering,
        NpuxDmaMvoutLowering
  >(typeConverter);
}