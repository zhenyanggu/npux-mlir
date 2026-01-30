//================================================
// src/Conversion/NpuToLLVM/ConvertNpuxToLLVM.cpp
//================================================

#include "src/Conversion/NpuToLLVM/ConvertNpuxToLLVM.hpp"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h" 
#include "mlir/Dialect/Arith/IR/Arith.h"

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

// 【通用 NPU 地址获取】
// 支持 SRAM/ACC Alloc 和 npux.subview 的递归地址计算
Value getNpuOffsetAddress(Location loc, Value memRef, ConversionPatternRewriter &rewriter) {
  // [新增] 空指针检查，处理 Optional 参数
  if (!memRef) return nullptr;

  Operation* defOp = memRef.getDefiningOp();
  if (!defOp) return nullptr;
  
  auto i32Type = rewriter.getI32Type();

  // 1. Base Case: Alloc/SramAlloc/AccAlloc (读取 Plan 阶段的 Offset)
  if (defOp->hasAttr("npu.offset")) {
    int32_t offsetVal = defOp->getAttrOfType<IntegerAttr>("npu.offset").getInt();
    return rewriter.create<LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(offsetVal));
  }

  // 2. Recursive Case: npux.subview
  if (auto subviewOp = dyn_cast<npux::SubviewOp>(defOp)) {
    // 2.1 递归获取 Source 的基地址
    Value baseAddr = getNpuOffsetAddress(loc, subviewOp.getSource(), rewriter);
    if (!baseAddr) return nullptr;

    // 2.2 获取 Source MemRef 的 Strides 信息
    auto sourceMemRefType = cast<MemRefType>(subviewOp.getSource().getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    // NPU 场景通常是 Static Shape
    if (failed(sourceMemRefType.getStridesAndOffset(strides, offset))) {
      return nullptr; 
    }

    // 2.3 累加偏移量: addr = base + sum(offset[i] * stride[i])
    Value currentAddr = baseAddr;
    auto dynamicOffsets = subviewOp.getOffsets();

    for (size_t i = 0; i < dynamicOffsets.size(); ++i) {
      int64_t strideVal = strides[i];
      if (strideVal == 0) continue; 

      Value dimOffset = dynamicOffsets[i];
      Value dimOffsetI64 = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI64Type(), dimOffset);

      // [修复] 2. 再将 i64 截断为 i32
      Value dimOffsetI32 = rewriter.create<LLVM::TruncOp>(
          loc, i32Type, dimOffsetI64);

      Value strideConst = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(strideVal));

      Value offsetBytes = rewriter.create<LLVM::MulOp>(loc, dimOffsetI32, strideConst);
      currentAddr = rewriter.create<LLVM::AddOp>(loc, currentAddr, offsetBytes);
    }

    return currentAddr;
  }

  return nullptr;
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
    Type elementType = memRefType.getElementType();
    int64_t elementSize = elementType.isIndex() ? 8 : (elementType.getIntOrFloatBitWidth() / 8);
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
// 3. Kernel (Internal Memory Management)
// ==========================================

class NpuxSramAllocLowering : public ConvertOpToLLVMPattern<npux::SramAllocOp> {
public:
  using ConvertOpToLLVMPattern<npux::SramAllocOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(npux::SramAllocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class NpuxSramFreeLowering : public ConvertOpToLLVMPattern<npux::SramFreeOp> {
public:
  using ConvertOpToLLVMPattern<npux::SramFreeOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(npux::SramFreeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class NpuxAccAllocLowering : public ConvertOpToLLVMPattern<npux::AccAllocOp> {
public:
  using ConvertOpToLLVMPattern<npux::AccAllocOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(npux::AccAllocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class NpuxAccFreeLowering : public ConvertOpToLLVMPattern<npux::AccFreeOp> {
public:
  using ConvertOpToLLVMPattern<npux::AccFreeOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(npux::AccFreeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class NpuxSubviewLowering : public ConvertOpToLLVMPattern<npux::SubviewOp> {
public:
  using ConvertOpToLLVMPattern<npux::SubviewOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(npux::SubviewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};


// ==========================================
// 4. Data Movement & Compute
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

    Value dstMemRef = op.getDstMemref();
    Value dstAddr = getNpuOffsetAddress(loc, dstMemRef, rewriter);
    
    if (!dstAddr) return failure();

    SmallVector<Value> args;
    args.push_back(hostPtr);   
    args.push_back(dstAddr); 
    args.push_back(adaptor.getColNum());
    args.push_back(adaptor.getRowNum());
    args.push_back(adaptor.getSramStride());
    args.push_back(adaptor.getDramStride());
    args.push_back(adaptor.getPrecision());
    args.push_back(adaptor.getInputType());
    args.push_back(adaptor.getDest());
    args.push_back(adaptor.getIsBias()); // Pass IsBias

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
    Value sramAddr = getNpuOffsetAddress(loc, sramMemRef, rewriter);
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

class NpuxMovAccToSpmLowering : public ConvertOpToLLVMPattern<MvAccToSpmOp> {
public:
  using ConvertOpToLLVMPattern<MvAccToSpmOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(MvAccToSpmOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    Value accAddr = getNpuOffsetAddress(loc, op.getAccSrc(), rewriter);
    Value spmAddr = getNpuOffsetAddress(loc, op.getSpmDst(), rewriter);
    
    if (!accAddr || !spmAddr) return failure();

    SmallVector<Value> args;
    args.push_back(accAddr);
    args.push_back(spmAddr);
    args.push_back(adaptor.getColNum());
    args.push_back(adaptor.getRowNum());
    args.push_back(adaptor.getAccStride());
    args.push_back(adaptor.getSpmStride());

    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(getContext());
    SmallVector<Type> argTypes;
    for(auto v : args) argTypes.push_back(v.getType());

    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(rewriter, module, "npu_mv_acc_to_spm", voidType, argTypes);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnRef, args);
    return success();
  }
};

class NpuxSfuRunLowering : public ConvertOpToLLVMPattern<SfuRunOp> {
public:
  using ConvertOpToLLVMPattern<SfuRunOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(SfuRunOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value inSramAddr = getNpuOffsetAddress(loc, op.getInputSramMemref(), rewriter);
    Value outSramAddr = getNpuOffsetAddress(loc, op.getOutputSramMemref(), rewriter);
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

// 【重写】ComputeRun (Matching npu_conv_run C-API)
class NpuxComputeRunLowering : public ConvertOpToLLVMPattern<ComputeRunOp> {
public:
  using ConvertOpToLLVMPattern<ComputeRunOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(ComputeRunOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    // Types
    auto i8Type  = rewriter.getI8Type();
    auto i16Type = rewriter.getI16Type();
    auto i32Type = rewriter.getI32Type();

    // 1. Resolve Addresses
    Value addrA = getNpuOffsetAddress(loc, op.getInputA(), rewriter);
    Value addrB = getNpuOffsetAddress(loc, op.getInputB(), rewriter);
    Value addrOut = getNpuOffsetAddress(loc, op.getOutput(), rewriter);

    // BiasPsum Addr (Optional)
    Value addrBias;
    if (op.getBiaspsumMemref()) {
        addrBias = getNpuOffsetAddress(loc, op.getBiaspsumMemref(), rewriter);
        if (!addrBias) return failure();
    } else {
        addrBias = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
    }

    if (!addrA || !addrB || !addrOut) {
      return rewriter.notifyMatchFailure(op, "Failed to resolve NPU addresses");
    }

    // Helpers for casting to correct C-API type width
    auto getI8  = [&](Value v) { return rewriter.create<LLVM::TruncOp>(loc, i8Type, v); };
    auto getI16 = [&](Value v) { return rewriter.create<LLVM::TruncOp>(loc, i16Type, v); };
    
    // Enum converters
    auto getEnumI8 = [&](auto enumAttr) -> Value {
      uint8_t val = static_cast<uint8_t>(enumAttr);
      return rewriter.create<LLVM::ConstantOp>(loc, i8Type, rewriter.getI8IntegerAttr(val));
    };

    SmallVector<Value> args;

    // --- 1. Padding (Top, Bottom, Left, Right, Mode) ---
    // C-API: uint8_t
    args.push_back(getI8(adaptor.getPadTop()));
    args.push_back(getI8(adaptor.getPadBottom()));
    args.push_back(getI8(adaptor.getPadLeft()));
    args.push_back(getI8(adaptor.getPadRight()));
    args.push_back(getI8(adaptor.getPadMode())); 

    // --- 2. Weight (Shape_m1, Stride_m1, Dil_m1, IsGroup) ---
    // C-API: uint8_t
    args.push_back(getI8(adaptor.getWeightShapeM1()));
    args.push_back(getI8(adaptor.getWeightStrideM1()));
    args.push_back(getI8(adaptor.getWeightDilationM1()));
    args.push_back(getI8(adaptor.getIsGroupConv()));

    // --- 3. Control (IntType, OpType, Dataflow, Dest) ---
    // C-API: uint8_t
    args.push_back(getI8(adaptor.getIntType()));
    args.push_back(getEnumI8(op.getOpType()));
    args.push_back(getEnumI8(op.getDataflowMode()));
    args.push_back(getEnumI8(op.getAccoutDest()));

    // --- 4. Quant Input (InA_ZP, InB_ZP) ---
    // C-API: uint16_t
    args.push_back(getI16(adaptor.getInputAZeropoint()));
    args.push_back(getI16(adaptor.getInputBZeropoint()));

    // --- 5. Input A Geometry (Addr, ColM1, RowM1, Stride) ---
    // C-API: u32, u8, u8, u16
    args.push_back(addrA);
    args.push_back(getI8(adaptor.getInputAColNumM1()));
    args.push_back(getI8(adaptor.getInputARowNumM1()));
    args.push_back(getI16(adaptor.getInputAStride()));

    // --- 6. Input B Geometry (Addr, ColM1, RowM1, Stride) ---
    // C-API: u32, u8, u8, u16
    args.push_back(addrB);
    args.push_back(getI8(adaptor.getInputBColNumM1()));
    args.push_back(getI8(adaptor.getInputBRowNumM1()));
    args.push_back(getI16(adaptor.getInputBStride()));

    // --- 7. Bias/Psum Geometry (Width, Height, Addr, Stride) ---
    // C-API: u8, u8, u32, u16
    args.push_back(getI8(adaptor.getBiaspsumWidth()));
    args.push_back(getI8(adaptor.getBiaspsumHeight()));
    args.push_back(addrBias);
    args.push_back(getI16(adaptor.getBiaspsumStride()));

    // --- 8. Output Geometry (Addr, Stride) ---
    // C-API: u32, u16
    args.push_back(addrOut);
    args.push_back(getI16(adaptor.getOutputStride()));

    // --- 9. Post-Process (Accum, Relu, ReluType, AccBias) ---
    // C-API: u8
    args.push_back(getI8(adaptor.getIsAccumulate()));
    args.push_back(getI8(adaptor.getReluEnable()));
    args.push_back(getEnumI8(op.getReluType()));
    args.push_back(getI8(adaptor.getAccBias()));

    // --- 10. Quant Output (OutZP, Scale, Shift) ---
    // C-API: u32, u16, u16
    args.push_back(adaptor.getOutputZeropoint()); // i32
    args.push_back(getI16(adaptor.getQuantScale()));
    args.push_back(getI16(adaptor.getQuantScaleshift()));

    // Generate Call
    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(getContext());
    
    SmallVector<Type> argTypes;
    for (auto v : args) argTypes.push_back(v.getType());

    // Function name: npu_conv_run (matching C driver)
    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(
        rewriter, module, "npu_conv_run", voidType, argTypes);

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
        NpuxSramFreeLowering,
        NpuxAccAllocLowering,    
        NpuxAccFreeLowering,     
        NpuxSubviewLowering, 
        NpuxMovAccToSpmLowering, 
        NpuxSfuRunLowering,
        NpuxComputeRunLowering,
        NpuxDmaMvoutLowering
  >(typeConverter);
}