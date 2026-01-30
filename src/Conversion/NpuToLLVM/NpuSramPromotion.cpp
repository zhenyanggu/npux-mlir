//======================================================
// src/Conversion/NpuToLLVM/NpuSramPromotion.cpp
// Promotes DRAM subviews to SRAM/ACC buffers for computation
//======================================================
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// =========================================================
// Helper 1: 创建量化/截断 Generic (ACC i32 -> SPM i8)
// 增加了 label 参数，方便后续识别
// =========================================================
void createQuantizeGeneric(OpBuilder &builder, Location loc, Value srcI32, Value dstI8, StringRef label) {
    auto srcType = cast<MemRefType>(srcI32.getType());
    int64_t rank = srcType.getRank();

    SmallVector<AffineMap, 2> indexingMaps(2, builder.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    auto genericOp = builder.create<linalg::GenericOp>(
        loc,
        /*resultTypes=*/TypeRange{}, 
        /*inputs=*/ValueRange{srcI32},
        /*outputs=*/ValueRange{dstI8},
        indexingMaps,
        iteratorTypes,
        /*bodyBuilder=*/[](OpBuilder &b, Location loc, ValueRange args) {
            Value input = args[0];
            // Value output = args[1]; // Unused

            // i32 -> i8 截断逻辑
            Value quantized = b.create<arith::TruncIOp>(loc, b.getI8Type(), input);
            b.create<linalg::YieldOp>(loc, quantized);
        }
    );

    // 打上 Label，方便后续 Pass 识别 (例如: "npu.quant_transfer")
    if (!label.empty()) {
        genericOp->setAttr("npu.pp_stage", builder.getStringAttr(label));
    }
}

// =========================================================
// Helper 2: 重写 Conv 的 Region
// 逻辑源自你提供的代码：根据 Acc 类型自动做 Cast
// =========================================================
void updateConvRegion(PatternRewriter &rewriter, linalg::GenericOp op, bool processedBias) {
    Region &region = op.getRegion();
    Block &block = region.front();
    
    // 1. 获取现有 Block 参数
    unsigned numArgs = block.getNumArguments();
    if (numArgs == 0) return;

    // 2. 修改 Accumulator 参数类型 (i8 -> i32)
    Type accType = rewriter.getI32Type(); 
    block.getArgument(numArgs - 1).setType(accType);

    while (!block.empty()) {
        rewriter.eraseOp(&block.back());
    }

    // 4. 重建 Body
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    
    Location loc = op.getLoc();
    ValueRange args = block.getArguments();

    Value acc = args.back();   // Accumulator (Output)
    Value input = args[0];     // Input Feature
    Value weight = args[1];    // Weight
    
    // ... 下面的逻辑保持不变 ...
    
    auto castToAccType = [&](Value val) -> Value {
        Type valType = val.getType();
        if (valType == accType) return val;
        
        unsigned accWidth = accType.getIntOrFloatBitWidth();
        unsigned valWidth = valType.getIntOrFloatBitWidth();
        
        if (valWidth > accWidth) {
            return rewriter.create<arith::TruncIOp>(loc, accType, val);
        } else {
            return rewriter.create<arith::ExtSIOp>(loc, accType, val);
        }
    };

    Value update;
    
    if (mlir::isa<FloatType>(accType)) {
        Value sum = rewriter.create<arith::MulFOp>(loc, input, weight); 
        if (processedBias && args.size() > 3) {
             sum = rewriter.create<arith::AddFOp>(loc, sum, args[2]);
        }
        update = rewriter.create<arith::AddFOp>(loc, acc, sum);
    } 
    else {
        Value castInput = castToAccType(input);
        Value castWeight = castToAccType(weight);
        
        Value prod = rewriter.create<arith::MulIOp>(loc, castInput, castWeight);
        
        Value sum = prod;
        if (processedBias && args.size() > 3) {
            Value castBias = castToAccType(args[2]);
            sum = rewriter.create<arith::AddIOp>(loc, sum, castBias);
        }
        
        update = rewriter.create<arith::AddIOp>(loc, acc, sum);
    }

    rewriter.create<linalg::YieldOp>(loc, update);
}


bool isElementWiseInPlace(linalg::GenericOp op) {
    if (op.getLibraryCallAttr()) return false;
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1) return false;
    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 2 || maps[0] != maps[1]) return false;
    for (auto type : op.getIteratorTypesArray()) {
        if (type != utils::IteratorType::parallel) return false;
    }
    return true;
}

class PromoteLinalgToSramPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, PatternRewriter &rewriter) const override {

    if (!op->hasAttr("npu.target")) return failure();

    Location loc = op.getLoc();
    
    // 1. 识别 NPU 计算类算子 (Conv, Gemm, Matmul)
    // 这些算子共享相同的 Promotion 策略：
    // - Inputs -> SRAM (Space 2)
    // - Bias (if any) -> ACC (Space 3)
    // - Output -> ACC (Space 3) -> Quantize -> SRAM (Space 2) -> DRAM
    bool isNpuCompute = false;
    if (auto libCall = op.getLibraryCallAttr()) {
        StringRef name = libCall.getValue();
        if (name == "npu_conv" || name == "npu_gemm" || name == "npu_matmul") {
            isNpuCompute = true;
        }
    }

    llvm::SmallDenseMap<unsigned, Value> operandsToReplace;
    bool promotedAny = false;
    llvm::SmallDenseMap<unsigned, Value> inputSramBuffers;

    // 2. Inputs Promotion
    for (auto item : llvm::enumerate(op.getInputs())) {
      unsigned idx = item.index(); 
      Value currentInput = item.value();
      auto type = dyn_cast<MemRefType>(currentInput.getType());

      if (!type) continue;
      // 如果已经在 SRAM (Space 2) 或 ACC (Space 3)，跳过
      if (type.getMemorySpaceAsInt() != 0) {
          inputSramBuffers[idx] = currentInput;
          continue;
      }

      int targetSpace = 2; // 默认搬运到 SPM (Space 2)
      
      // 特殊处理 Bias:
      // 对于 Conv 和 Gemm，通常 inputs[2] 是 Bias，需要放在 ACC (Space 3)
      // Matmul 通常只有 2 个输入，不会触发此逻辑
      if (isNpuCompute && idx == 2) {
          targetSpace = 3; 
      }

      auto internalMemType = MemRefType::get(type.getShape(), type.getElementType(), {}, targetSpace);
      Value internalBuf = rewriter.create<memref::AllocOp>(loc, internalMemType);
      rewriter.create<memref::CopyOp>(loc, currentInput, internalBuf);

      operandsToReplace[idx] = internalBuf;
      inputSramBuffers[idx] = internalBuf;
      promotedAny = true;
    }

    // 3. Outputs Promotion
    unsigned inputCount = op.getNumDpsInputs();
    SmallVector<std::pair<Value, Value>> copyChain; 
    
    Value accBufferI32 = nullptr;
    Value spmBufferI8 = nullptr;
    bool canReuse = isElementWiseInPlace(op);

    for (auto item : llvm::enumerate(op.getOutputs())) {
      unsigned outputIdxRel = item.index();
      unsigned operandIdx = inputCount + outputIdxRel; 
      Value currentOutput = item.value();
      auto type = dyn_cast<MemRefType>(currentOutput.getType());

      if (!type || type.getMemorySpaceAsInt() != 0) continue;

      Value computeResultBuf;

      if (isNpuCompute) {
          // === NPU Compute (Conv/Gemm/Matmul) ===
          // path: ACC(i32) -> SPM(i8) -> DRAM

          // 1. 创建 Accumulator Buffer (Space 3, i32)
          auto accType = MemRefType::get(type.getShape(), rewriter.getI32Type(), {}, 3);
          computeResultBuf = rewriter.create<memref::AllocOp>(loc, accType);
          accBufferI32 = computeResultBuf;

          // 2. 创建 SPM Buffer (Space 2, i8) 用于量化回传
          auto spmType = MemRefType::get(type.getShape(), type.getElementType(), {}, 2);
          Value spmBuf = rewriter.create<memref::AllocOp>(loc, spmType);
          spmBufferI8 = spmBuf;

          // 3. 记录 Copy 链: SPM -> DRAM
          // (ACC -> SPM 的转换由后续的 createQuantizeGeneric 处理)
          copyChain.push_back({spmBuf, currentOutput});

      } else {
          // === Elementwise Ops (Add, Sub, etc.) ===
          if (canReuse && outputIdxRel == 0 && inputSramBuffers.count(0)) {
              Value inputBuf = inputSramBuffers[0];
              auto inType = cast<MemRefType>(inputBuf.getType());
              if (inType.getElementType() == type.getElementType() && inType.getShape() == type.getShape()) {
                  computeResultBuf = inputBuf;
              }
          }
          if (!computeResultBuf) {
              auto sramType = MemRefType::get(type.getShape(), type.getElementType(), {}, 2);
              computeResultBuf = rewriter.create<memref::AllocOp>(loc, sramType);
          }
          copyChain.push_back({computeResultBuf, currentOutput});
      }

      operandsToReplace[operandIdx] = computeResultBuf;
      promotedAny = true;
    }

    if (!promotedAny) return failure();

    // 4. Clone Op 并替换 Operand
    auto newOp = cast<linalg::GenericOp>(rewriter.clone(*op.getOperation()));
    for (auto it : operandsToReplace) {
      newOp->setOperand(it.first, it.second);
    }

    // 5. Update Region (针对 NpuCompute 且 Accumulator 发生变化的情况)
    if (isNpuCompute && accBufferI32) {
        // 判断是否有 Bias: 输入数量 > 2 (Inputs[0], Inputs[1] 是数据, Inputs[2] 是 Bias)
        // Matmul (2 inputs) -> hasBias = false
        // Gemm/Conv (3 inputs) -> hasBias = true
        bool hasBias = (newOp.getNumDpsInputs() > 2);
        
        // 复用 updateConvRegion，它内部逻辑是通用的: Acc += A * B + (Bias?)
        updateConvRegion(rewriter, newOp, hasBias);
    }

    // 6. Insert Copy Back & Quantization Logic
    rewriter.setInsertionPointAfter(newOp);

    if (isNpuCompute && accBufferI32 && spmBufferI8) {
        // 插入 Quantize 节点: ACC(i32) -> SPM(i8)
        // label 设为 "quant_acc2spm"
        createQuantizeGeneric(rewriter, loc, accBufferI32, spmBufferI8, "quant_acc2spm");
    }

    // 执行内存拷贝回 DRAM
    for (auto pair : copyChain) {
      rewriter.create<memref::CopyOp>(loc, pair.first, pair.second);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct SramPromotionPass : public PassWrapper<SramPromotionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SramPromotionPass)
  StringRef getArgument() const override { return "npu-sram-promotion"; }
  StringRef getDescription() const override { return "Promote DRAM to SRAM/ACC for NPU Ops"; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    context->getOrLoadDialect<arith::ArithDialect>(); 
    
    func::FuncOp func = getOperation();
    auto targetAttr = func->getAttrOfType<StringAttr>("npu.target");
    if (!targetAttr || targetAttr.getValue() != "npu") return;

    patterns.add<PromoteLinalgToSramPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuSramPromotionPass() {
  return std::make_unique<SramPromotionPass>();
}