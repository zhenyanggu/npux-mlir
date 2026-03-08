//================================================
// src/Conversion/NpuToLLVM/ConvertNpuxToLLVM.cpp
//================================================

#include "src/Conversion/NpuToLLVM/ConvertNpuxToLLVM.hpp"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace npux;

// ==========================================
// 辅助函数
// ==========================================

static Value castToI8(
    Location loc, Value val, ConversionPatternRewriter &rewriter) {
  return rewriter.create<LLVM::TruncOp>(loc, rewriter.getI8Type(), val);
}

// 2. 将 i32 等截断为 i16 (用于 stride, zeropoint 等)
static Value castToI16(
    Location loc, Value val, ConversionPatternRewriter &rewriter) {
  return rewriter.create<LLVM::TruncOp>(loc, rewriter.getI16Type(), val);
}

// 3. [关键] 将 i8 (来自旧ODS) 截断为 i1 (用于新CAPI bool)
// 如果 ODS 定义是 I8，但 CAPI 改成了 bool，必须用这个转换
static Value castI8ToBool(
    Location loc, Value val, ConversionPatternRewriter &rewriter) {
  // 只有当 val 不是 i1 时才截断
  if (val.getType().isInteger(1))
    return val;
  return rewriter.create<LLVM::TruncOp>(loc, rewriter.getI1Type(), val);
}

// 4. 将 Enum 属性转为 i1 (bool) 常量
template <typename T>
static Value getEnumBool(
    Location loc, T enumVal, ConversionPatternRewriter &rewriter) {
  bool val = static_cast<bool>(enumVal);
  return rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI1Type(), rewriter.getBoolAttr(val));
}

// 5. 将 Enum 属性转为 i8 常量
template <typename T>
static Value getEnumI8(
    Location loc, T enumVal, ConversionPatternRewriter &rewriter) {
  uint8_t val = static_cast<uint8_t>(enumVal);
  return rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(val));
}

static FlatSymbolRefAttr getOrInsertExternFunc(PatternRewriter &rewriter,
    ModuleOp module, StringRef funcName, Type resultType,
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

Value getFlatPtrFromMemRef(Location loc, Value memrefDescVal, Type elemType,
    ConversionPatternRewriter &rewriter) {
  MemRefDescriptor desc(memrefDescVal);
  Value alignedPtr = desc.alignedPtr(rewriter, loc);
  Value offset = desc.offset(rewriter, loc);

  // GEP i8* base + offset
  Value finalPtr = rewriter.create<LLVM::GEPOp>(loc, alignedPtr.getType(),
      rewriter.getI8Type(), alignedPtr, ArrayRef<Value>({offset}));

  return finalPtr;
}

// 【通用 NPU 地址获取】
// 支持 SRAM/ACC Alloc 和 npux.subview 的递归地址计算
Value getNpuOffsetAddress(
    Location loc, Value memRef, ConversionPatternRewriter &rewriter) {
  // [新增] 空指针检查，处理 Optional 参数
  if (!memRef)
    return nullptr;

  Operation *defOp = memRef.getDefiningOp();
  if (!defOp)
    return nullptr;

  auto i32Type = rewriter.getI32Type();

  // 1. Base Case: Alloc/SramAlloc/AccAlloc (读取 Plan 阶段的 Offset)
  if (defOp->hasAttr("npu.offset")) {
    int32_t offsetVal =
        defOp->getAttrOfType<IntegerAttr>("npu.offset").getInt();
    return rewriter.create<LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(offsetVal));
  }

  // 2. Recursive Case: npux.subview
  if (auto subviewOp = dyn_cast<npux::SubviewOp>(defOp)) {
    // 2.1 递归获取 Source 的基地址
    Value baseAddr = getNpuOffsetAddress(loc, subviewOp.getSource(), rewriter);
    if (!baseAddr)
      return nullptr;

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
      if (strideVal == 0)
        continue;

      Value dimOffset = dynamicOffsets[i];
      Value dimOffsetI64 = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI64Type(), dimOffset);

      // [修复] 2. 再将 i64 截断为 i32
      Value dimOffsetI32 =
          rewriter.create<LLVM::TruncOp>(loc, i32Type, dimOffsetI64);

      Value strideConst = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(strideVal));

      Value offsetBytes =
          rewriter.create<LLVM::MulOp>(loc, dimOffsetI32, strideConst);
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

    auto fnRef =
        getOrInsertExternFunc(rewriter, module, "npu_init", i32Type, {});
    auto callOp = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i32Type, fnRef, ValueRange{});

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

    auto fnRef =
        getOrInsertExternFunc(rewriter, module, "npu_destroy", voidType, {});

    rewriter.create<LLVM::CallOp>(
        op.getLoc(), TypeRange{}, fnRef, ValueRange{});

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

    if (!memRefType.hasStaticShape())
      return failure();
    int64_t numElements = memRefType.getNumElements();
    Type elementType = memRefType.getElementType();
    int64_t elementSize =
        elementType.isIndex() ? 8 : (elementType.getIntOrFloatBitWidth() / 8);
    if (elementSize == 0)
      elementSize = 1;

    int64_t totalBytes = numElements * elementSize;

    Value sizeVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(totalBytes));

    auto voidPtrType = LLVM::LLVMPointerType::get(getContext());
    auto fnRef =
        getOrInsertExternFunc(rewriter, op->getParentOfType<ModuleOp>(),
            "npu_mem_alloc", voidPtrType, {i64Type});

    auto callOp = rewriter.create<LLVM::CallOp>(
        op.getLoc(), voidPtrType, fnRef, ValueRange{sizeVal});
    Value rawPtr = callOp.getResult();

    auto targetType = getTypeConverter()->convertType(memRefType);
    if (!targetType)
      return failure();

    Value undefStruct = rewriter.create<LLVM::UndefOp>(loc, targetType);
    MemRefDescriptor desc(undefStruct);
    desc.setAllocatedPtr(rewriter, loc, rawPtr);
    desc.setAlignedPtr(rewriter, loc, rawPtr);

    Value zero = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(0));
    desc.setOffset(rewriter, loc, zero);

    for (int i = 0; i < memRefType.getRank(); ++i) {
      desc.setSize(rewriter, loc, i,
          rewriter.create<LLVM::ConstantOp>(loc, i64Type,
              rewriter.getI64IntegerAttr(memRefType.getDimSize(i))));
      int64_t stride = 1;
      for (int j = i + 1; j < memRefType.getRank(); ++j)
        stride *= memRefType.getDimSize(j);
      desc.setStride(rewriter, loc, i,
          rewriter.create<LLVM::ConstantOp>(
              loc, i64Type, rewriter.getI64IntegerAttr(stride)));
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

    auto fnRef = getOrInsertExternFunc(
        rewriter, module, "npu_mem_free", voidType, {voidPtrType});

    MemRefDescriptor desc(adaptor.getMemref());
    Value ptr = desc.allocatedPtr(rewriter, op.getLoc());

    rewriter.create<LLVM::CallOp>(
        op.getLoc(), TypeRange{}, fnRef, ValueRange{ptr});
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

    // 1. 准备地址
    auto dramMemRefType = cast<MemRefType>(op.getHostPtr().getType());
    Value hostPtr = getFlatPtrFromMemRef(
        loc, adaptor.getHostPtr(), dramMemRefType.getElementType(), rewriter);
    Value dstAddr = getNpuOffsetAddress(loc, op.getDstMemref(), rewriter);
    if (!dstAddr)
      return failure();

    SmallVector<Value> args;
    args.push_back(hostPtr);
    args.push_back(dstAddr);
    args.push_back(adaptor.getColNum());     // i16
    args.push_back(adaptor.getRowNum());     // i16
    args.push_back(adaptor.getSramStride()); // i16
    args.push_back(adaptor.getDramStride()); // i16
    args.push_back(adaptor.getPrecision());  // i8
    args.push_back(adaptor.getInputType());  // i8

    // --- 修正重点 ---
    // ODS 定义是 I8:$dest，Adaptor 返回 i8。CAPI 需要 bool (i1)。
    // 动作：调用 castI8ToBool 进行截断。
    args.push_back(castI8ToBool(loc, adaptor.getDest(), rewriter));

    // ODS 定义是 I1:$is_bias，Adaptor 返回 i1。CAPI 需要 bool (i1)。
    // 动作：直接传。
    args.push_back(adaptor.getIsBias());

    // ODS 定义是 I1:$is_quant，直接传。
    args.push_back(adaptor.getIsQuant());

    args.push_back(adaptor.getQuantZero());  // i32
    args.push_back(adaptor.getQuantScale()); // i16
    args.push_back(adaptor.getQuantShift()); // i16

    // 2. 生成调用
    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(getContext());
    // 自动推导类型：现在 args 里包含 i1，生成的函数签名就是 (..., i1, i1, ...)
    SmallVector<Type> argTypes;
    for (auto v : args)
      argTypes.push_back(v.getType());

    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(
        rewriter, module, "npu_dma_mvin", voidType, argTypes);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnRef, args);
    return success();
  }
};

class NpuxMvinBiasLowering : public ConvertOpToLLVMPattern<MvinBiasOp> {
public:
  using ConvertOpToLLVMPattern<MvinBiasOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(MvinBiasOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // 1. 提取输入地址 (Source Pointer)
    auto memRefType = cast<MemRefType>(op.getSource().getType());
    Value hostPtr = getFlatPtrFromMemRef(
        loc, adaptor.getSource(), memRefType.getElementType(), rewriter);

    // 2. 创建 Dummy/默认常量
    // 根据你提供的 CAPI 参数顺序和类型进行填充
    Value c0_i1 =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI1Type(), 0);
    Value c1_i1 =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI1Type(), 1);
    Value c0_i8 =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI8Type(), 0);
    Value c0_i16 =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI16Type(), 0);
    Value c0_i32 =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 0);

    SmallVector<Value> args;
    // 参数顺序参考你提供的 DmaMvinLowering 逻辑
    args.push_back(hostPtr); // hostPtr (来自 source)
    args.push_back(c0_i32);  // dstAddr: 因为写到专用寄存器，传 0 即可
    args.push_back(c0_i32);  // colNum
    args.push_back(c0_i32);  // rowNum
    args.push_back(c0_i16);  // sramStride
    args.push_back(c0_i32);  // dramStride
    args.push_back(c0_i8);   // precision
    args.push_back(c0_i8);   // inputType
    args.push_back(c0_i1);   // dest

    // --- 核心设置 ---
    args.push_back(c1_i1); // is_bias: 必须设为 1

    args.push_back(c0_i1);  // is_quant
    args.push_back(c0_i32); // quantZero
    args.push_back(c0_i16); // quantScale
    args.push_back(c0_i16); // quantShift

    // 3. 准备 CAPI 函数签名
    auto voidType = LLVM::LLVMVoidType::get(getContext());
    SmallVector<Type> argTypes;
    for (auto v : args) {
      argTypes.push_back(v.getType());
    }

    // 依然复用 npu_dma_mvin 这个 CAPI
    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(
        rewriter, module, "npu_dma_mvin", voidType, argTypes);

    // 4. 生成调用并替换
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
    Value hostPtr = getFlatPtrFromMemRef(
        loc, adaptor.getHostPtr(), dramMemRefType.getElementType(), rewriter);
    Value sramAddr = getNpuOffsetAddress(loc, op.getSramMemref(), rewriter);
    if (!sramAddr)
      return failure();

    SmallVector<Value> args;
    args.push_back(hostPtr);
    args.push_back(sramAddr);
    args.push_back(adaptor.getColNum());
    args.push_back(adaptor.getRowNum());
    args.push_back(adaptor.getSramStride());
    args.push_back(adaptor.getDramStride());
    args.push_back(adaptor.getPrecision());
    args.push_back(adaptor.getOutputType());

    // --- 修正重点 ---
    // ODS I8:$source -> CAPI bool source
    args.push_back(castI8ToBool(loc, adaptor.getSource(), rewriter));

    // ODS I1:$is_quant -> CAPI bool is_quant
    args.push_back(adaptor.getIsQuant());

    args.push_back(adaptor.getQuantZero());
    args.push_back(adaptor.getQuantScale());
    args.push_back(adaptor.getQuantShift());

    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(getContext());
    SmallVector<Type> argTypes;
    for (auto v : args)
      argTypes.push_back(v.getType());

    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(
        rewriter, module, "npu_dma_mvout", voidType, argTypes);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnRef, args);
    return success();
  }
};

// class NpuxMovAccToSpmLowering : public ConvertOpToLLVMPattern<MvAccToSpmOp> {
// public:
//   using ConvertOpToLLVMPattern<MvAccToSpmOp>::ConvertOpToLLVMPattern;

//   LogicalResult matchAndRewrite(MvAccToSpmOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();

//     Value accAddr = getNpuOffsetAddress(loc, op.getAccSrc(), rewriter);
//     Value spmAddr = getNpuOffsetAddress(loc, op.getSpmDst(), rewriter);

//     if (!accAddr || !spmAddr)
//       return failure();

//     SmallVector<Value> args;
//     args.push_back(accAddr);
//     args.push_back(spmAddr);
//     args.push_back(adaptor.getColNum());
//     args.push_back(adaptor.getRowNum());
//     args.push_back(adaptor.getAccStride());
//     args.push_back(adaptor.getSpmStride());

//     auto module = op->getParentOfType<ModuleOp>();
//     auto voidType = LLVM::LLVMVoidType::get(getContext());
//     SmallVector<Type> argTypes;
//     for (auto v : args)
//       argTypes.push_back(v.getType());

//     FlatSymbolRefAttr fnRef = getOrInsertExternFunc(
//         rewriter, module, "npu_mv_acc_to_spm", voidType, argTypes);
//     rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnRef, args);
//     return success();
//   }
// };

class NpuxSfuRunLowering : public ConvertOpToLLVMPattern<SfuRunOp> {
public:
  using ConvertOpToLLVMPattern<SfuRunOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(SfuRunOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value inSramAddr =
        getNpuOffsetAddress(loc, op.getInputSramMemref(), rewriter);
    Value outSramAddr =
        getNpuOffsetAddress(loc, op.getOutputSramMemref(), rewriter);
    if (!inSramAddr || !outSramAddr)
      return failure();

    SmallVector<Value> args;
    uint32_t opTypeVal = static_cast<uint32_t>(op.getOpType());
    args.push_back(rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI8Type(), rewriter.getI8IntegerAttr(opTypeVal)));
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
    for (auto v : args)
      argTypes.push_back(v.getType());

    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(
        rewriter, module, "npu_sfu_run", voidType, argTypes);
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

    // 地址解析
    Value addrA = getNpuOffsetAddress(loc, op.getInputA(), rewriter);
    Value addrB = getNpuOffsetAddress(loc, op.getInputB(), rewriter);
    Value addrOut = getNpuOffsetAddress(loc, op.getOutput(), rewriter);
    Value addrPsum;
    if (op.getPsumMemref()) {
      addrPsum = getNpuOffsetAddress(loc, op.getPsumMemref(), rewriter);
      if (!addrPsum)
        return failure();
    } else {
      addrPsum = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(0));
    }
    if (!addrA || !addrB || !addrOut)
      return failure();

    SmallVector<Value> args;
    std::string funcName;

    if (op.getOpType() == npux::ComputeOpType::gemm) {
      funcName = "npu_gemm_run";

      // Enum -> Bool (i1)
      args.push_back(getEnumBool(loc, op.getDataflowMode(), rewriter));

      // i8 (ODS) -> i8 (CAPI) Direct
      args.push_back(castToI8(loc, adaptor.getIntType(), rewriter));

      // Enum -> i8
      args.push_back(getEnumI8(loc, op.getOpType(), rewriter));

      // Enum -> Bool (i1)
      args.push_back(getEnumBool(loc, op.getAccoutDest(), rewriter));

      // ZP (i32) -> i16
      args.push_back(castToI16(loc, adaptor.getInputAZeropoint(), rewriter));
      args.push_back(castToI16(loc, adaptor.getInputBZeropoint(), rewriter));
      args.push_back(adaptor.getOutputZeropoint()); // i32

      // Scale/Shift (i32) -> i16
      args.push_back(castToI16(loc, adaptor.getQuantScale(), rewriter));
      args.push_back(castToI16(loc, adaptor.getQuantScaleshift(), rewriter));

      // Bias/Psum
      args.push_back(addrPsum);
      args.push_back(castToI16(loc, adaptor.getBiaspsumStride(), rewriter));
      args.push_back(castToI8(loc, adaptor.getBiaspsumWidth(), rewriter));
      args.push_back(castToI8(loc, adaptor.getBiaspsumHeight(), rewriter));

      // Out
      args.push_back(addrOut);
      args.push_back(castToI16(loc, adaptor.getOutputStride(), rewriter));

      // Flags (ODS I1 -> CAPI Bool) - 直接传
      args.push_back(adaptor.getIsAccumulate());
      args.push_back(adaptor.getReluEnable());
      args.push_back(
          getEnumI8(loc, op.getReluType(), rewriter)); // ReluType是3bit，用i8
      args.push_back(adaptor.getAccBias());

      // Inputs
      args.push_back(addrA);
      args.push_back(castToI16(loc, adaptor.getInputAColNumM1(), rewriter));
      args.push_back(castToI8(loc, adaptor.getInputARowNumM1(), rewriter));
      args.push_back(castToI16(loc, adaptor.getInputAStride(), rewriter));

      args.push_back(addrB);
      args.push_back(castToI8(loc, adaptor.getInputBColNumM1(), rewriter));
      args.push_back(castToI16(loc, adaptor.getInputBRowNumM1(), rewriter));
      args.push_back(castToI16(loc, adaptor.getInputBStride(), rewriter));

    } else {
      // funcName = "npu_conv_tile_run";

      // // ==========================================
      // // 1. 准备辅助变量和类型
      // // ==========================================
      // auto weightType = cast<MemRefType>(op.getInputB().getType());
      // auto outType = cast<MemRefType>(op.getOutput().getType());
      // auto inType = cast<MemRefType>(op.getInputA().getType());

      // // ==========================================
      // // 2. 提取维度信息 (Layout Reads)
      // // ==========================================

      // // Weight Layout: [C/32, C/32, k, k, 32, 32] -> indices 2, 3 correspond
      // to
      // // k_h, k_w
      // int64_t kh_val = weightType.getDimSize(2);
      // int64_t kw_val = weightType.getDimSize(3);

      // // Output Layout: [N, C/32, H, W, 32] -> indices 2, 3 correspond to H,
      // W int64_t th_out_val = outType.getDimSize(2); int64_t tw_out_val =
      // outType.getDimSize(3);

      // int64_t tcout_val = outType.getDimSize(1) * outType.getDimSize(4);

      // // t_cin 逻辑: 输入的第二个维度(index 1) 乘 最里面的维度(index 4)
      // int64_t tcin_val = inType.getDimSize(1) * inType.getDimSize(4);

      // // ==========================================
      // // 3. 构建参数列表 (Strict Order)
      // // ==========================================

      // // --- Address Pointers ---
      // // uint32_t sram_addr_ifm (Input A)
      // args.push_back(addrA);

      // // uint32_t sram_addr_weight (Input B / Weight)
      // args.push_back(addrB);

      // // uint32_t sram_addr_ofm (Output)
      // args.push_back(addrOut);

      // // uint32_t acc_addr_psum (Bias/Psum)
      // args.push_back(addrPsum);

      // // --- Logic: i_cin & c_in ---
      // // 逻辑: i_cin 的取值, 当 isbias=1 (acc_bias) 时, i_cin=0
      // // 注意: 这里假设默认 i_cin 为 0 (因为 op 中似乎没有直接的 tile index
      // // 参数), 如果 op 中有对应的属性(例如 input_a_col_num
      // // 用于切分)，请在此处修改 default_icin。
      // Value i_cin = rewriter.create<LLVM::ConstantOp>(
      //     loc, i32Type, rewriter.getI32IntegerAttr(32));

      // // 如果 acc_bias (isbias) 为 true, i_cin 必须为 0
      // // 可以在运行时用 select
      // // 指令，也可以在编译时判断(如果是常量)。这里使用运行时的 Select
      // // 确保逻辑正确。
      // Value constZero = rewriter.create<LLVM::ConstantOp>(
      //     loc, i32Type, rewriter.getI32IntegerAttr(0));
      // Value isBiasVal = adaptor.getAccBias(); // i1
      // // 如果 isBias 为 true, 选 0, 否则选 i_cin (原值)
      // i_cin = rewriter.create<LLVM::SelectOp>(loc, isBiasVal, constZero,
      // i_cin);

      // // 逻辑: c_in 的取值
      // // 默认 c_in (这里设为0或者t_cin? 根据描述"取值不重要",
      // 只有特定情况重要) Value c_in = rewriter.create<LLVM::ConstantOp>(
      //     loc, i32Type, rewriter.getI32IntegerAttr(64));

      // // 当输出 dest 是 spm 时, c_in = i_cin
      // // 判断 dest 是否为 SPM (Enum value 0)
      // bool isDestSpm = (op.getAccoutDest() == npux::AccoutDest::spm);
      // if (isDestSpm) {
      //   i_cin = c_in;
      // }

      // // int32_t c_in
      // args.push_back(c_in);

      // // --- Convolution Parameters ---
      // // int32_t k_h
      // args.push_back(rewriter.create<LLVM::ConstantOp>(
      //     loc, i32Type, rewriter.getI32IntegerAttr(kh_val)));

      // // int32_t k_w
      // args.push_back(rewriter.create<LLVM::ConstantOp>(
      //     loc, i32Type, rewriter.getI32IntegerAttr(kw_val)));

      // Value convParamOne = rewriter.create<LLVM::ConstantOp>(
      //     loc, i32Type, rewriter.getI32IntegerAttr(1));

      // // int32_t stride (CAPI 需要原始值，因此对 *_m1 做 +1)
      // args.push_back(rewriter.create<LLVM::AddOp>(
      //     loc, adaptor.getWeightStrideM1(), convParamOne));

      // // int32_t dilation (CAPI 需要原始值，因此对 *_m1 做 +1)
      // args.push_back(rewriter.create<LLVM::AddOp>(
      //     loc, adaptor.getWeightDilationM1(), convParamOne));

      // // [注意] Padding 在你提供的新列表中被注释掉了，如果 CAPI 确实移除了
      // // padding 参数，这里就不传。 如果需要
      // // padding，请解开以下注释并按顺序加入:
      // /*
      // args.push_back(adaptor.getPadTop());
      // args.push_back(adaptor.getPadBottom());
      // args.push_back(adaptor.getPadLeft());
      // args.push_back(adaptor.getPadRight());
      // */

      // // --- Macro Tile / Computed Geometry ---
      // // int32_t i_cin
      // args.push_back(i_cin);

      // // int32_t t_cout
      // args.push_back(rewriter.create<LLVM::ConstantOp>(
      //     loc, i32Type, rewriter.getI32IntegerAttr(tcout_val)));

      // // int32_t t_h_out
      // args.push_back(rewriter.create<LLVM::ConstantOp>(
      //     loc, i32Type, rewriter.getI32IntegerAttr(th_out_val)));

      // // int32_t t_w_out
      // args.push_back(rewriter.create<LLVM::ConstantOp>(
      //     loc, i32Type, rewriter.getI32IntegerAttr(tw_out_val)));

      // // int32_t t_cin
      // args.push_back(rewriter.create<LLVM::ConstantOp>(
      //     loc, i32Type, rewriter.getI32IntegerAttr(tcin_val)));

      // // --- Quantization / Activation ---
      //   // uint16_t quant_scale
      // args.push_back(castToI16(loc, adaptor.getQuantScale(), rewriter));

      // // uint16_t quant_scaleshift
      // args.push_back(castToI16(loc, adaptor.getQuantScaleshift(), rewriter));

      // // bool relu_enable
      // args.push_back(adaptor.getReluEnable());

      // // uint8_t relu_type (3bit -> i8)
      // args.push_back(getEnumI8(loc, op.getReluType(), rewriter));

      //   // bool bias_enable
      //   args.push_back(adaptor.getAccBias());

      // // bool is_group_conv
      // args.push_back(adaptor.getIsGroupConv());

      funcName = "npu_conv_run";
      SmallVector<Value, 36> args;

      // ==========================================
      // 1-5. Padding 相关 (uint8_t)
      // ==========================================
      args.push_back(castToI8(loc, adaptor.getPadTop(), rewriter));
      args.push_back(castToI8(loc, adaptor.getPadBottom(), rewriter));
      args.push_back(castToI8(loc, adaptor.getPadLeft(), rewriter));
      args.push_back(castToI8(loc, adaptor.getPadRight(), rewriter));
      args.push_back(castToI8(loc, adaptor.getPadMode(), rewriter));

      // ==========================================
      // 6-9. Weight/Kernel 相关 (uint8_t, bool)
      // ==========================================
      args.push_back(castToI8(loc, adaptor.getWeightShapeM1(), rewriter));
      args.push_back(castToI8(loc, adaptor.getWeightStrideM1(), rewriter));
      args.push_back(castToI8(loc, adaptor.getWeightDilationM1(), rewriter));
      args.push_back(adaptor.getIsGroupConv()); // bool (i1)

      // ==========================================
      // 10-13. 基础控制 (uint8_t, bool)
      // ==========================================
      args.push_back(castToI8(loc, adaptor.getIntType(), rewriter));
      args.push_back(getEnumI8(loc, op.getOpType(), rewriter));

      args.push_back(
          getEnumBool(loc, op.getDataflowMode(), rewriter)); // WS/OS -> i1
      args.push_back(
          getEnumBool(loc, op.getAccoutDest(), rewriter)); // SPM/ACC -> i1

      // ==========================================
      // 14-15. ZeroPoints (uint16_t)
      // ==========================================
      args.push_back(castToI16(loc, adaptor.getInputAZeropoint(), rewriter));
      args.push_back(castToI16(loc, adaptor.getInputBZeropoint(), rewriter));

      // ==========================================
      // 16-19. Input A (addr, col, row, stride)
      // ==========================================
      args.push_back(addrA); // uint32_t
      args.push_back(castToI16(loc, adaptor.getInputAColNumM1(), rewriter));
      args.push_back(castToI8(loc, adaptor.getInputARowNumM1(), rewriter));
      args.push_back(castToI16(loc, adaptor.getInputAStride(), rewriter));

      // ==========================================
      // 20-23. Input B (addr, col, row, stride)
      // ==========================================
      args.push_back(addrB); // uint32_t
      args.push_back(castToI8(loc, adaptor.getInputBColNumM1(), rewriter));
      args.push_back(castToI16(loc, adaptor.getInputBRowNumM1(), rewriter));
      args.push_back(castToI16(loc, adaptor.getInputBStride(), rewriter));

      // ==========================================
      // 24-27. Bias/Psum (width, height, addr, stride)
      // ==========================================
      args.push_back(castToI8(loc, adaptor.getBiaspsumWidth(), rewriter));
      args.push_back(castToI8(loc, adaptor.getBiaspsumHeight(), rewriter));
      args.push_back(addrPsum); // uint32_t
      args.push_back(castToI16(loc, adaptor.getBiaspsumStride(), rewriter));

      // ==========================================
      // 28-29. Output (addr, stride)
      // ==========================================
      args.push_back(addrOut); // uint32_t
      args.push_back(castToI16(loc, adaptor.getOutputStride(), rewriter));

      // ==========================================
      // 30-33. Post-Processing (bool, uint8_t)
      // ==========================================
      args.push_back(adaptor.getIsAccumulate()); // bool
      args.push_back(adaptor.getReluEnable());   // bool
      args.push_back(getEnumI8(loc, op.getReluType(), rewriter));
      args.push_back(adaptor.getAccBias()); // is_bias (bool)

      // ==========================================
      // 34-36. Final Quant (uint32, uint16, uint16)
      // ==========================================
      args.push_back(adaptor.getOutputZeropoint()); // uint32_t
      args.push_back(castToI16(loc, adaptor.getQuantScale(), rewriter));
      args.push_back(castToI16(loc, adaptor.getQuantScaleshift(), rewriter));
    }
    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(getContext());
    SmallVector<Type> argTypes;
    for (auto v : args)
      argTypes.push_back(v.getType());

    FlatSymbolRefAttr fnRef =
        getOrInsertExternFunc(rewriter, module, funcName, voidType, argTypes);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnRef, args);

    return success();
  }
};

class NpuxTransposeRunLowering
    : public ConvertOpToLLVMPattern<npux::TransposeOp> {
public:
  using ConvertOpToLLVMPattern<npux::TransposeOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(npux::TransposeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // 1. 获取地址 (保持不变)
    Value inAddr = getNpuOffsetAddress(loc, op.getInputSram(), rewriter);
    Value outAddr = getNpuOffsetAddress(loc, op.getOutputSram(), rewriter);

    if (!inAddr || !outAddr)
      return failure();

    SmallVector<Value> args;
    args.push_back(inAddr);
    args.push_back(outAddr);

    // 2. 处理行列数 (确保是 i16)
    auto ensureI16 = [&](Value v) -> Value {
      Type t = v.getType();
      if (t.isInteger(16))
        return v;
      return rewriter.create<LLVM::TruncOp>(loc, rewriter.getI16Type(), v);
    };

    args.push_back(ensureI16(adaptor.getColNum()));
    args.push_back(ensureI16(adaptor.getRowNum()));

    // 3. 处理新增的两个 bool 参数 (out_padding_row, out_padding_col)
    // 在 LLVM Dialect 中，bool 通常对应 i1
    Value falseVal = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));

    args.push_back(falseVal); // out_padding_row = 0
    args.push_back(falseVal); // out_padding_col = 0

    // 4. 获取函数并生成调用
    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());

    // 更新参数类型列表
    SmallVector<Type> argTypes;
    for (auto v : args)
      argTypes.push_back(v.getType());

    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(
        rewriter, module, "npu_transpose_run", voidType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnRef, args);
    return success();
  }
};

// ============================================================================
// Resample Run Lowering
// ============================================================================
class NpuxResampleRunLowering
    : public ConvertOpToLLVMPattern<npux::ResampleOp> {
public:
  using ConvertOpToLLVMPattern<npux::ResampleOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(npux::ResampleOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value inAddr = getNpuOffsetAddress(loc, op.getInputSram(), rewriter);
    Value outAddr = getNpuOffsetAddress(loc, op.getOutputSram(), rewriter);
    if (!inAddr || !outAddr)
      return failure();

    SmallVector<Value> args;

    // 1. Enum -> uint8_t
    args.push_back(getEnumI8(loc, op.getResampleType(), rewriter));
    args.push_back(getEnumI8(loc, op.getResampleMode(), rewriter));

    // 2. Addrs
    args.push_back(inAddr);
    args.push_back(outAddr);

    // 3. Shapes (uint16_t)
    auto ensureI16 = [&](Value v) -> Value {
      Type t = v.getType();
      if (t.isInteger(16))
        return v;
      return rewriter.create<LLVM::TruncOp>(loc, rewriter.getI16Type(), v);
    };
    args.push_back(ensureI16(adaptor.getInputColNum()));
    args.push_back(ensureI16(adaptor.getInputRowNum()));

    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(getContext());
    SmallVector<Type> argTypes;
    for (auto v : args)
      argTypes.push_back(v.getType());

    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(
        rewriter, module, "npu_resample_run", voidType, argTypes);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnRef, args);
    return success();
  }
};

class NpuxLayoutNchwToNchwc32Lowering
    : public ConvertOpToLLVMPattern<npux::LayoutNchwToNchwc32Op> {
public:
  using ConvertOpToLLVMPattern<
      npux::LayoutNchwToNchwc32Op>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(npux::LayoutNchwToNchwc32Op op,
      OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // 1. 获取 SRAM 地址
    // 注意：Layout 转换 Op 的输入输出都在 SRAM (Memory Space 2)
    Value inAddr = getNpuOffsetAddress(loc, op.getInputSram(), rewriter);
    Value outAddr = getNpuOffsetAddress(loc, op.getOutputSram(), rewriter);

    if (!inAddr || !outAddr)
      return failure();

    // 2. 准备参数
    // void npu_layout_nchw_to_nchwc32(uint32_t sram_addr, uint32_t output_addr,
    //                                 uint16_t n, uint16_t c, uint16_t h,
    //                                 uint16_t w);
    SmallVector<Value> args;
    args.push_back(inAddr);
    args.push_back(outAddr);

    // 辅助 lambda：确保 i16 类型
    auto ensureI16 = [&](Value v) -> Value {
      Type t = v.getType();
      if (t.isInteger(16))
        return v;
      // 虽然 ODS 中已经是 I16，但为了稳健性，如果类型系统有变化则 Trunc
      return rewriter.create<LLVM::TruncOp>(loc, rewriter.getI16Type(), v);
    };

    args.push_back(ensureI16(adaptor.getN()));
    args.push_back(ensureI16(adaptor.getC()));
    args.push_back(ensureI16(adaptor.getH()));
    args.push_back(ensureI16(adaptor.getW()));

    // 3. 生成函数调用
    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    SmallVector<Type> argTypes;
    for (auto v : args)
      argTypes.push_back(v.getType());

    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(
        rewriter, module, "npu_layout_nchw_to_nchwc32", voidType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnRef, args);
    return success();
  }
};

class NpuxLayoutNchwc32ToNchwLowering
    : public ConvertOpToLLVMPattern<npux::LayoutNchwc32ToNchwOp> {
public:
  using ConvertOpToLLVMPattern<
      npux::LayoutNchwc32ToNchwOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(npux::LayoutNchwc32ToNchwOp op,
      OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value inAddr = getNpuOffsetAddress(loc, op.getInputSram(), rewriter);
    Value outAddr = getNpuOffsetAddress(loc, op.getOutputSram(), rewriter);

    if (!inAddr || !outAddr)
      return failure();

    // void npu_layout_nchwc32_to_nchw(uint32_t sram_addr, uint32_t output_addr,
    //                                 uint16_t n, uint16_t c, uint16_t h,
    //                                 uint16_t w);
    SmallVector<Value> args;
    args.push_back(inAddr);
    args.push_back(outAddr);

    auto ensureI16 = [&](Value v) -> Value {
      Type t = v.getType();
      if (t.isInteger(16))
        return v;
      return rewriter.create<LLVM::TruncOp>(loc, rewriter.getI16Type(), v);
    };

    args.push_back(ensureI16(adaptor.getN()));
    args.push_back(ensureI16(adaptor.getC()));
    args.push_back(ensureI16(adaptor.getH()));
    args.push_back(ensureI16(adaptor.getW()));

    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    SmallVector<Type> argTypes;
    for (auto v : args)
      argTypes.push_back(v.getType());

    FlatSymbolRefAttr fnRef = getOrInsertExternFunc(
        rewriter, module, "npu_layout_nchwc32_to_nchw", voidType, argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnRef, args);
    return success();
  }
};

void npux::populateNpuxToLLVMConversionPatterns(
    RewritePatternSet &patterns, LLVMTypeConverter &typeConverter) {
  patterns.add<NpuxInitLowering, NpuxDestroyLowering, NpuxAllocLowering,
      NpuxFreeLowering, NpuxMvinBiasLowering, NpuxDmaMvinLowering,
      NpuxSramAllocLowering, NpuxSramFreeLowering, NpuxAccAllocLowering,
      NpuxAccFreeLowering, NpuxSubviewLowering, NpuxSfuRunLowering,
      NpuxComputeRunLowering, NpuxDmaMvoutLowering, NpuxTransposeRunLowering,
      NpuxResampleRunLowering, NpuxLayoutNchwToNchwc32Lowering,
      NpuxLayoutNchwc32ToNchwLowering>(typeConverter);
}