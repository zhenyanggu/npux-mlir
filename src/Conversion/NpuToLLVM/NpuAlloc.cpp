//==============================================================
// src/Conversion/NpuToLLVM/NpuAlloc.cpp
// Unified Memory Allocation Pass for Host (DRAM) and NPU (SRAM)
//==============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h" // 必须包含这个
#include "src/Pass/Passes.hpp"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;

namespace {

// 定义内存空间常量 (必须与 NpuInstructionLowering.cpp 保持一致)
const int HOST_SPACE = 0;        // CPU 专用
const int NPU_SRAM_SPACE = 1;    // NPU 内部 SRAM (Space 1)
const int SHARED_DRAM_SPACE = 2; // CPU/NPU 共享 DRAM (Space 2)

const int64_t SRAM_ALIGNMENT = 64;

// ============================================================
// Part 1: Host DRAM Allocation Logic (Pattern Rewrite)
// ============================================================

// 检查是否被 NPU DMA 使用 (兼容旧逻辑)
static bool isNpuRelated(Value memref) {
    for (Operation *user : memref.getUsers()) {
        if (auto callOp = dyn_cast<func::CallOp>(user)) {
            StringRef funcName = callOp.getCallee();
            if (funcName.starts_with("npu_dma_")) return true;
        }
        if (auto viewOp = dyn_cast<ViewLikeOpInterface>(user)) {
            if (isNpuRelated(viewOp->getResult(0))) return true;
        }
    }
    return false;
}

// 插入 npu_mem_alloc 声明
static func::FuncOp getOrInsertAllocFunc(PatternRewriter &rewriter, ModuleOp module) {
    if (auto func = module.lookupSymbol<func::FuncOp>("npu_mem_alloc")) return func;
    
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());

    // 【修复 1】使用 ShapedType::kDynamic (防止 -1 导致的 crash)
    // 【修复 2】明确返回 Space 2 (Shared DRAM)，否则 memref.view 会报错
    auto returnType = MemRefType::get(
        {ShapedType::kDynamic},       // Shape: [?]
        rewriter.getI8Type(),         // Element: i8
        MemRefLayoutAttrInterface{},  // Layout: Identity
        rewriter.getI64IntegerAttr(SHARED_DRAM_SPACE) // Space: 2
    );

    auto funcType = FunctionType::get(rewriter.getContext(), 
                                      {rewriter.getIndexType()}, // Input: size
                                      {returnType});             // Output: memref<?xi8, 2>
    
    auto func = rewriter.create<func::FuncOp>(module.getLoc(), "npu_mem_alloc", funcType);
    func.setPrivate();
    return func;
}

struct LowerSharedAllocPattern : public OpRewritePattern<memref::AllocOp> {
    using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(memref::AllocOp op, PatternRewriter &rewriter) const override {
        auto type = mlir::cast<MemRefType>(op.getType());
        int space = type.getMemorySpaceAsInt();

        // 【判断条件】
        // 1. 如果是 Space 2 (明确标记为共享 DRAM) -> 必须处理
        // 2. 如果是 Space 0 (普通内存) 但被 NPU 用到了 -> 为了性能也处理 (可选)
        bool shouldLower = (space == SHARED_DRAM_SPACE) || 
                           (space == HOST_SPACE && isNpuRelated(op.getResult()));

        if (!shouldLower) return failure();

        // 1. 计算总字节数
        int64_t totalSize = (type.getNumElements() * type.getElementTypeBitWidth()) / 8;

        // 2. 生成 npu_mem_alloc 调用
        auto module = op->getParentOfType<ModuleOp>();
        auto allocFunc = getOrInsertAllocFunc(rewriter, module);
        Value sizeVal = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), totalSize);
        auto call = rewriter.create<func::CallOp>(op.getLoc(), allocFunc, ValueRange{sizeVal});
        
        // 3. 将返回的 void* (i8 memref) 转换回原来的类型
        // 这一步至关重要：它保证了类型的连续性，比如 affine.store 依然能操作它
        Value voidPtr = call.getResult(0);
        Value byteShift = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
        
        // 使用 memref.view 强转类型
        // 注意：这里生成的 ViewOp 会保留原有的 Space (比如 Space 2)
        Value casted = rewriter.create<memref::ViewOp>(op.getLoc(), type, voidPtr, byteShift, ValueRange{});
        
        rewriter.replaceOp(op, casted);
        return success();
    }
};

// ============================================================
// Part 2: NPU SRAM Allocation Logic (Global Analysis)
// ============================================================

class SmartAllocator {
    struct Interval {
        int64_t start; int64_t end; int64_t size; int64_t offset = -1; memref::AllocOp op;
    };
    std::vector<Interval> intervals;

public:
    void analyze(ModuleOp module) {
        int opIndex = 0;
        DenseMap<Operation*, int> opMap;
        module.walk([&](Operation *op) { opMap[op] = opIndex++; });

        module.walk([&](memref::AllocOp allocOp) {
            auto type = mlir::cast<MemRefType>(allocOp.getType());
            // 【关键】只处理 Space 1 (纯 SRAM)
            // 绝对不要处理 Space 2 或 Space 0
            if (type.getMemorySpaceAsInt() != NPU_SRAM_SPACE) return;

            int64_t sizeBytes = (type.getNumElements() * type.getElementTypeBitWidth()) / 8;
            int lastUseIndex = opMap[allocOp];
            for (auto user : allocOp.getResult().getUsers()) {
                if (opMap.count(user)) lastUseIndex = std::max(lastUseIndex, opMap[user]);
            }
            intervals.push_back({opMap[allocOp], lastUseIndex, sizeBytes, -1, allocOp});
        });

        std::sort(intervals.begin(), intervals.end(), [](const Interval &a, const Interval &b) {
            return a.start < b.start;
        });
    }

    void solve() {
        int64_t currentTop = 0;
        for (auto &live : intervals) {
            if (currentTop % SRAM_ALIGNMENT != 0) 
                currentTop += (SRAM_ALIGNMENT - currentTop % SRAM_ALIGNMENT);
            live.offset = currentTop;
            currentTop += live.size;
        }
    }

    int64_t getOffset(memref::AllocOp op) {
        for (const auto &inter : intervals) if (inter.op == op) return inter.offset;
        return -1;
    }
};

// ============================================================
// Unified Pass
// ============================================================

struct NpuMemoryAllocationPass : public PassWrapper<NpuMemoryAllocationPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuMemoryAllocationPass)
    StringRef getArgument() const override { return "npu-memory-allocation"; }
    StringRef getDescription() const override { return "Handle memory allocation for both Shared DRAM and NPU SRAM"; }
    
    void runOnOperation() override {
        ModuleOp module = getOperation();
        OpBuilder b(module.getContext());

        // ---------------------------------------------------------
        // Phase 1: Shared DRAM Allocation (Rewrite Patterns)
        // 目标：处理 Space 2 (Shared) -> npu_mem_alloc
        // ---------------------------------------------------------
        {
            RewritePatternSet patterns(&getContext());
            patterns.add<LowerSharedAllocPattern>(&getContext());
            if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
                signalPassFailure();
                return;
            }
        }

        // ---------------------------------------------------------
        // Phase 2: NPU SRAM Allocation (Analysis & Replacement)
        // 目标：处理 Space 1 (SRAM) -> i32 Constant Address
        // ---------------------------------------------------------
        SmartAllocator allocator;
        allocator.analyze(module);
        allocator.solve();

        DenseMap<Value, Value> addressMap;

        // 2.1 替换 Alloc 为 Constant Address
        module.walk([&](memref::AllocOp allocOp) {
            auto type = mlir::cast<MemRefType>(allocOp.getType());
            // 只替换 Space 1
            if (type.getMemorySpaceAsInt() != NPU_SRAM_SPACE) return;

            int64_t offset = allocator.getOffset(allocOp);
            // 如果没分配到 (例如 size=0 或被过滤掉)，跳过
            if (offset == -1) return;

            b.setInsertionPoint(allocOp);
            Value addrVal = b.create<arith::ConstantIntOp>(allocOp.getLoc(), offset, 32);
            addressMap[allocOp.getResult()] = addrVal;
        });

        // 2.2 计算 SubView 偏移
        module.walk([&](memref::SubViewOp subviewOp) {
            if (mlir::cast<MemRefType>(subviewOp.getType()).getMemorySpaceAsInt() != NPU_SRAM_SPACE) return;
            
            Value source = subviewOp.getSource();
            // 如果 source 没被替换成 i32，说明 source 是 DRAM 指针，那这个 subview 应该保留为指针运算
            if (addressMap.count(source) == 0) return;

            b.setInsertionPoint(subviewOp);
            Value baseAddr = addressMap[source];
            
            // 简化的偏移计算 (假设 offset=0)
            // 真正的工程代码需要在这里展开 subview 的 offsets/strides
            Value byteOffset = b.create<arith::ConstantIntOp>(subviewOp.getLoc(), 0, 32); 
            
            Value finalAddr = b.create<arith::AddIOp>(subviewOp.getLoc(), baseAddr, byteOffset);
            addressMap[subviewOp.getResult()] = finalAddr;
        });

        // 2.3 更新 Call 参数
        module.walk([&](func::CallOp call) {
            for (unsigned i = 0; i < call.getNumOperands(); ++i) {
                if (addressMap.count(call.getOperand(i))) {
                    call.setOperand(i, addressMap[call.getOperand(i)]);
                }
            }
        });

        // 2.4 清理 Space 1 的 MemRef 操作
        // 只有那些真正被替换了的 (在 addressMap 里的) 才删除
        module.walk([&](memref::SubViewOp op) {
             if (addressMap.count(op.getResult())) op.erase();
        });
        module.walk([&](memref::AllocOp op) {
             if (addressMap.count(op.getResult())) op.erase();
        });

        // 2.5 更新 Runtime 函数签名
        SymbolTable symbolTable(module);
        module.walk([&](func::FuncOp func) {
            if (!func.isPrivate() || !func.getName().starts_with("npu_")) return;
            
            auto type = func.getFunctionType();
            SmallVector<Type> newInputs;
            
            for (unsigned i = 0; i < type.getNumInputs(); ++i) {
                Type t = type.getInput(i);
                if (auto memrefT = mlir::dyn_cast<MemRefType>(t)) {
                     // 只有 Space 1 (SRAM) 才变 i32
                     // Space 2 (Shared) 和 Space 0 (Host) 保持为 MemRef (Pointer)
                     if (memrefT.getMemorySpaceAsInt() == NPU_SRAM_SPACE) {
                         newInputs.push_back(b.getI32Type());
                     } else {
                         newInputs.push_back(t);
                     }
                } else {
                    newInputs.push_back(t);
                }
            }
            func.setType(FunctionType::get(func.getContext(), newInputs, type.getResults()));
        });
    }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuMemoryAllocationPass() {
    return std::make_unique<NpuMemoryAllocationPass>();
}

static PassRegistration<NpuMemoryAllocationPass> pass;