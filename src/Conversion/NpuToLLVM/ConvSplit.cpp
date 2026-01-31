//=============================================================================
// src/Conversion/NpuPartition/ConvSplit.cpp
//
// Split Convolution Loop on IC dimension for NPU optimization.
//=============================================================================
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/Npux/NpuxOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace npux;

namespace {

// ============================================================================
// 1. Analysis Helpers
// ============================================================================

struct OpChain {
  memref::SubViewOp dramSubview = nullptr;
  Value hostPtr = nullptr; // [新增]
  DmaMvinOp mvinOp = nullptr;
  Value sramBuffer = nullptr;
};

struct ConvLoopAnalysis {
  ComputeRunOp computeOp = nullptr;
  OpChain inputChain;
  OpChain weightChain;
  OpChain biasChain;

  MvAccToSpmOp accToSpmOp = nullptr;
  DmaMvoutOp mvoutOp = nullptr;
  memref::SubViewOp outputDramSubview = nullptr;

  bool isValid() const { return computeOp != nullptr; }
};

OpChain traceMvinChain(Value sramValue, Operation *scope) {
  OpChain chain;
  if (!sramValue)
    return chain;

  for (Operation *user : sramValue.getUsers()) {
    if (!scope->isAncestor(user))
      continue;

    if (auto mvin = dyn_cast<DmaMvinOp>(user)) {
      if (mvin.getDstMemref() == sramValue) {
        chain.mvinOp = mvin;
        chain.sramBuffer = sramValue;

        // [修改] 始终记录 Host Ptr，无论它是什么 Op 定义的
        Value hostPtr = mvin.getHostPtr();
        chain.hostPtr = hostPtr; // 需要在 struct OpChain 里加这个成员

        // 尝试向上查找 subview，找不到也没关系
        while (hostPtr) {
          if (auto sv = hostPtr.getDefiningOp<memref::SubViewOp>()) {
            chain.dramSubview = sv;
            break;
          }
          if (auto cast = hostPtr.getDefiningOp<memref::CastOp>()) {
            hostPtr = cast.getSource();
            continue;
          }
          break;
        }
        break;
      }
    }
  }
  return chain;
}

ConvLoopAnalysis analyzeLoop(scf::ForOp op) {
  ConvLoopAnalysis analysis;
  Operation *loopScope = op.getOperation();

  op.getBody()->walk([&](ComputeRunOp comp) { analysis.computeOp = comp; });

  if (!analysis.isValid())
    return analysis;

  analysis.inputChain =
      traceMvinChain(analysis.computeOp.getInputA(), loopScope);
  analysis.weightChain =
      traceMvinChain(analysis.computeOp.getInputB(), loopScope);

  if (analysis.computeOp.getBiaspsumMemref()) {
    analysis.biasChain =
        traceMvinChain(analysis.computeOp.getBiaspsumMemref(), loopScope);
  }

  op.getBody()->walk([&](DmaMvoutOp mvout) {
    analysis.mvoutOp = mvout;
    Value hostPtr = mvout.getHostPtr();
    while (hostPtr) {
      if (auto sv = hostPtr.getDefiningOp<memref::SubViewOp>()) {
        analysis.outputDramSubview = sv;
        break;
      }
      if (auto cast = hostPtr.getDefiningOp<memref::CastOp>()) {
        hostPtr = cast.getSource();
        continue;
      }
      break;
    }
  });

  op.getBody()->walk([&](MvAccToSpmOp acc) { analysis.accToSpmOp = acc; });

  return analysis;
}

// ============================================================================
// 2. Slicing Helpers
// ============================================================================

Value getVal(ImplicitLocOpBuilder &b, OpFoldResult ofr) {
  if (auto val = ofr.dyn_cast<Value>())
    return val;
  auto attr = ofr.dyn_cast<Attribute>();
  int64_t intVal = attr ? cast<IntegerAttr>(attr).getInt() : 0;
  return b.create<arith::ConstantIndexOp>(intVal);
}

// [Fixed] DRAM 切片函数 - 修复了 Use-After-Free 问题
// 必须传入 oldIv，如果发现 OriginalOffset == oldIv，则直接替换，而不是相加。
Value createDramSliceOffset(ImplicitLocOpBuilder &b,
    memref::SubViewOp originalOp, Value baseIv, Value innerIv, Value oldIv,
    int64_t targetDim) {
  auto source = originalOp.getSource();
  auto mixedOffsets = originalOp.getMixedOffsets();
  auto mixedSizes = originalOp.getMixedSizes();
  auto mixedStrides = originalOp.getMixedStrides();

  if (targetDim >= 0 && targetDim < (int64_t)mixedOffsets.size()) {
    // 1. 获取该维度原有的偏移量 (可能是 Value 或 Constant)
    OpFoldResult origOfr = mixedOffsets[targetDim];
    Value originalOffset = getVal(b, origOfr);

    // 2. 安全处理：如果原偏移就是旧的循环变量 (oldIv)，
    // 则将其映射为新的外层循环变量 (baseIv)
    if (originalOffset == oldIv) {
      originalOffset = baseIv;
    }

    // 3. 核心逻辑：新的偏移 = 原偏移 + 内层循环增量
    Value finalOffset = b.create<arith::AddIOp>(originalOffset, innerIv);

    // 4. 更新偏移和尺寸 (尺寸强制设为 1)
    mixedOffsets[targetDim] = finalOffset;
    mixedSizes[targetDim] = b.getIndexAttr(1);
  }

  return b.create<memref::SubViewOp>(
      source, mixedOffsets, mixedSizes, mixedStrides);
}

Value createSramSlice(
    ImplicitLocOpBuilder &b, Value buffer, Value innerIv, int64_t splitDim) {
  auto memType = cast<MemRefType>(buffer.getType());
  int64_t rank = memType.getRank();
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));

  for (int64_t i = 0; i < rank; ++i) {
    if (i == splitDim) {
      offsets[i] = innerIv;
      sizes.push_back(b.getIndexAttr(1));
    } else {
      sizes.push_back(b.getIndexAttr(memType.getDimSize(i)));
    }
  }
  return b.create<memref::SubViewOp>(buffer, offsets, sizes, strides);
}

// Weight 切片处理 (保持 oldIv 替换逻辑)
Value createDramSliceFull(ImplicitLocOpBuilder &b, memref::SubViewOp originalOp,
    Value baseIv, Value oldIv) {
  auto source = originalOp.getSource();
  auto mixedOffsets = originalOp.getMixedOffsets();
  auto mixedSizes = originalOp.getMixedSizes();
  auto mixedStrides = originalOp.getMixedStrides();

  for (size_t i = 0; i < mixedOffsets.size(); ++i) {
    if (auto val = mixedOffsets[i].dyn_cast<Value>()) {
      if (val == oldIv) {
        mixedOffsets[i] = baseIv;
      }
    }
  }
  return b.create<memref::SubViewOp>(
      source, mixedOffsets, mixedSizes, mixedStrides);
}

// ============================================================================
// 3. Pattern Logic
// ============================================================================

struct SplitConvIcPattern : public OpRewritePattern<scf::ForOp> {
    using OpRewritePattern<scf::ForOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::ForOp op, PatternRewriter &rewriter) const override {
        if (!op->hasAttr("npu.loop_part")) return failure();
        if ((!op->hasAttr("npu.computeop"))||(op->getAttrOfType<StringAttr>("npu.computeop")!="conv")) return failure();
        
        StringRef part = "body";
        if (auto attr = op->getAttrOfType<StringAttr>("npu.loop_part")) part = attr.getValue();

        ConvLoopAnalysis analysis = analyzeLoop(op);
        if (!analysis.isValid()) return failure();
        
        if (!analysis.inputChain.mvinOp) return rewriter.notifyMatchFailure(op, "Input mvin not found");
        
        auto newOuterLoop = rewriter.create<scf::ForOp>(
            op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(), ValueRange{},
            [&](OpBuilder &b, Location loc, Value outerIv, ValueRange) {
                b.create<scf::YieldOp>(loc);
            }
        );
        for (auto &attr : op->getAttrs()) newOuterLoop->setAttr(attr.getName(), attr.getValue());

        rewriter.setInsertionPointToStart(newOuterLoop.getBody());
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);
        
        // ====================================================================
        // [FIX START] Map old buffers to new buffers inside the new loop
        // ====================================================================
        DenseMap<Value, Value> bufferMap;
        auto mapBuffer = [&](Value oldBuffer) {
            if (!oldBuffer || bufferMap.count(oldBuffer)) return;
            // Check if defined inside the loop (AllocOp)
            if (Operation* defOp = oldBuffer.getDefiningOp()) {
                if (op->isAncestor(defOp)) {
                    // Clone the alloc op to the new scope
                    Operation* newAlloc = b.clone(*defOp);
                    bufferMap[oldBuffer] = newAlloc->getResult(0);
                    return;
                }
            }
            // If defined outside, use as is
            bufferMap[oldBuffer] = oldBuffer;
        };

        // Register all needed buffers
        mapBuffer(analysis.inputChain.sramBuffer);
        mapBuffer(analysis.weightChain.sramBuffer);
        if (analysis.biasChain.mvinOp) mapBuffer(analysis.biasChain.sramBuffer);
        mapBuffer(analysis.computeOp.getOutput());
        if (analysis.accToSpmOp) mapBuffer(analysis.accToSpmOp.getSpmDst());

        // Helper to get new buffer
        auto getNewBuf = [&](Value v) { return bufferMap.count(v) ? bufferMap[v] : v; };
        // ====================================================================
        // [FIX END]
        // ====================================================================

        Value c0 = b.create<arith::ConstantIndexOp>(0);
        Value c1 = b.create<arith::ConstantIndexOp>(1);
        Value loopSize = op.getStep(); 
        Value oldIv = op.getInductionVar(); 
        Value outerIv = newOuterLoop.getInductionVar();

        const int64_t DIM_WEGHIT_OC = 0;
        const int64_t DIM_INPUT_C = 1; 
        const int64_t DIM_BIAS_C = 0;

        // -----------------------------------------------------------
        // 1. Loop 1: Input Mvin
        // -----------------------------------------------------------
        rewriter.create<scf::ForOp>(op.getLoc(), c0, loopSize, c1, ValueRange{}, 
            [&](OpBuilder &ib, Location loc, Value innerIv, ValueRange) {
                ImplicitLocOpBuilder nb(loc, ib);
                
                Value hostSlice;
                if (analysis.inputChain.dramSubview) {
                    hostSlice = createDramSliceOffset(nb, analysis.inputChain.dramSubview, outerIv, innerIv, oldIv, DIM_INPUT_C);
                } else {
                    hostSlice = createSramSlice(nb, analysis.inputChain.hostPtr, innerIv, DIM_INPUT_C);
                }

                // [Fix] Use new buffer
                Value sramSlice = createSramSlice(nb, getNewBuf(analysis.inputChain.sramBuffer), innerIv, DIM_INPUT_C);
                
                auto newMvin = cast<DmaMvinOp>(ib.clone(*analysis.inputChain.mvinOp));
                newMvin.getHostPtrMutable().assign(hostSlice);
                newMvin.getDstMemrefMutable().assign(sramSlice);
                ib.create<scf::YieldOp>(loc);
            });

        // -----------------------------------------------------------
        // 2. Block 2: Weight Mvin
        // -----------------------------------------------------------
        Value weightBuffer = getNewBuf(analysis.computeOp.getInputB()); // [Fix] Use new buffer
        if (analysis.weightChain.mvinOp) {
             Value hostWeight;
             if (analysis.weightChain.dramSubview) {
                 hostWeight = createDramSliceFull(b, analysis.weightChain.dramSubview, outerIv, oldIv);
             } else {
                 hostWeight = analysis.weightChain.hostPtr;
             }
             
             // [Fix] Use new buffer
             Value sramWeightFull = getNewBuf(analysis.weightChain.sramBuffer);
             auto mvin = cast<DmaMvinOp>(b.clone(*analysis.weightChain.mvinOp));
             mvin.getHostPtrMutable().assign(hostWeight);
             mvin.getDstMemrefMutable().assign(sramWeightFull);
        }

        // -----------------------------------------------------------
        // 3. Loop 3: Compute Loop
        // -----------------------------------------------------------
        rewriter.create<scf::ForOp>(op.getLoc(), c0, loopSize, c1, ValueRange{}, 
            [&](OpBuilder &ib, Location loc, Value innerIv, ValueRange) {
                ImplicitLocOpBuilder nb(loc, ib);

                bool isHead = (part == "head" || part == "single");
                bool isTail = (part == "tail" || part == "single");

                Value cTrue = nb.create<arith::ConstantIntOp>(1, 1);
                Value cFalse = nb.create<arith::ConstantIntOp>(0, 1);

                Value weightSramSlice = createSramSlice(nb, weightBuffer, innerIv, DIM_WEGHIT_OC);
                // [Fix] Use new buffer
                Value outputAccSlice = createSramSlice(nb, getNewBuf(analysis.computeOp.getOutput()), innerIv, DIM_INPUT_C);
                // [Fix] Use new buffer
                Value inputFull = getNewBuf(analysis.computeOp.getInputA()); 

                Value biasPsumOperand = Value();
                
                if (isHead) {
                    if (analysis.computeOp.getBiaspsumMemref()) {
                        Value biasBuffer = getNewBuf(analysis.computeOp.getBiaspsumMemref()); // [Fix]
                        Value biasSramSlice = createSramSlice(nb, biasBuffer, innerIv, DIM_BIAS_C);
                        biasPsumOperand = biasSramSlice;

                        if (analysis.biasChain.mvinOp) {
                             Value hostBiasSlice;
                             if (analysis.biasChain.dramSubview) {
                                 hostBiasSlice = createDramSliceOffset(nb, analysis.biasChain.dramSubview, 
                                                                       outerIv, innerIv, oldIv, DIM_BIAS_C);
                             } else {
                                 hostBiasSlice = createSramSlice(nb, analysis.biasChain.hostPtr, innerIv, DIM_BIAS_C);
                             }
                             
                             auto mvin = cast<DmaMvinOp>(ib.clone(*analysis.biasChain.mvinOp));
                             mvin.getHostPtrMutable().assign(hostBiasSlice);
                             mvin.getDstMemrefMutable().assign(biasSramSlice);
                             mvin.getIsBiasMutable().assign(cTrue);
                        }
                    }
                } else {
                    biasPsumOperand = outputAccSlice;
                }

                auto newComp = cast<ComputeRunOp>(ib.clone(*analysis.computeOp));
                newComp.getInputAMutable().assign(inputFull);
                newComp.getInputBMutable().assign(weightSramSlice);
                newComp.getOutputMutable().assign(outputAccSlice);
                
                if (biasPsumOperand) {
                    newComp.getBiaspsumMemrefMutable().assign(biasPsumOperand);
                } else {
                    newComp.getBiaspsumMemrefMutable().assign(ValueRange{});
                }

                if (isHead) {
                    newComp.getIsAccumulateMutable().assign(cFalse);
                    bool hasBias = (analysis.computeOp.getBiaspsumMemref() != nullptr);
                    newComp.getAccBiasMutable().assign(hasBias ? cTrue : cFalse);
                } else {
                    newComp.getIsAccumulateMutable().assign(cTrue); 
                    newComp.getAccBiasMutable().assign(cFalse);      
                }

                if (isTail && analysis.accToSpmOp) {
                    // [Fix] Use new buffer
                    Value spmDstSlice = createSramSlice(nb, getNewBuf(analysis.accToSpmOp.getSpmDst()), innerIv, DIM_INPUT_C);
                    auto newAcc = cast<MvAccToSpmOp>(ib.clone(*analysis.accToSpmOp));
                    newAcc.getAccSrcMutable().assign(outputAccSlice);
                    newAcc.getSpmDstMutable().assign(spmDstSlice);
                }

                ib.create<scf::YieldOp>(loc);
            });

        // -----------------------------------------------------------
        // 4. Loop 4: Output Loop
        // -----------------------------------------------------------
        if (part == "tail" || part == "single") {
             rewriter.create<scf::ForOp>(op.getLoc(), c0, loopSize, c1, ValueRange{}, 
                [&](OpBuilder &ib, Location loc, Value innerIv, ValueRange) {
                    ImplicitLocOpBuilder nb(loc, ib);
                    
                    // [Fix] Use new buffer
                    Value spmSlice = createSramSlice(nb, getNewBuf(analysis.accToSpmOp.getSpmDst()), innerIv, DIM_INPUT_C);
                    Value hostOut;
                    
                    if (analysis.outputDramSubview) {
                        hostOut = createDramSliceOffset(nb, analysis.outputDramSubview, 
                                                        outerIv, innerIv, oldIv, DIM_INPUT_C);
                    } else if (analysis.mvoutOp) {
                        hostOut = createSramSlice(nb, analysis.mvoutOp.getHostPtr(), innerIv, DIM_INPUT_C);
                    }
                    
                    if (analysis.mvoutOp) {
                        auto newMvout = cast<DmaMvoutOp>(ib.clone(*analysis.mvoutOp));
                        newMvout.getHostPtrMutable().assign(hostOut);
                        newMvout.getSramMemrefMutable().assign(spmSlice);
                    }

                    ib.create<scf::YieldOp>(loc);
                });
        }
        
        // [IMPORTANT] Clone Free Ops at the end of the new loop to prevent leaks/verifier errors
        op.getBody()->walk([&](Operation *childOp) {
             if (isa<SramFreeOp, AccFreeOp>(childOp)) {
                 Value oldMem = childOp->getOperand(0);
                 if (bufferMap.count(oldMem)) {
                     auto newFree = b.clone(*childOp);
                     newFree->setOperand(0, bufferMap[oldMem]);
                 }
             }
        });

        rewriter.eraseOp(op);
        return success();
    }
};

// ============================================================================
// 新的 Pattern：直接匹配 ComputeRunOp 进行拆分
// ============================================================================
struct SplitIndependentConvPattern : public OpRewritePattern<ComputeRunOp> {
  using OpRewritePattern<ComputeRunOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ComputeRunOp op, PatternRewriter &rewriter) const override {
    // 1. 防御性检查：必须不在循环里
    if (op->getParentOfType<scf::ForOp>())
      return failure();
    if (op.getOpType() != ::npux::ComputeOpType::conv)
        return failure();
    // 2. 查找关联 Op
    Operation *scope = op->getParentOp();
    OpChain biasChain;
    MvAccToSpmOp accToSpmOp = nullptr;
    DmaMvoutOp mvoutOp = nullptr; // [Added] 用于寻找输出的 Mvout

    // 找 Bias Mvin
    if (op.getBiaspsumMemref()) {
      biasChain = traceMvinChain(op.getBiaspsumMemref(), scope);
    }

    // 找 AccToSpm
    for (Operation *user : op.getOutput().getUsers()) {
      if (auto acc = dyn_cast<MvAccToSpmOp>(user)) {
        accToSpmOp = acc;
        break;
      }
    }
    if (!accToSpmOp)
      return failure();

    // [Added] 找 Output Mvout (通过 AccToSpm 的输出继续向下找)
    // 逻辑：Compute -> Acc -> AccToSpm -> SpmBuffer -> Mvout
    for (Operation *user : accToSpmOp.getSpmDst().getUsers()) {
      if (auto mv = dyn_cast<DmaMvoutOp>(user)) {
        mvoutOp = mv;
        break;
      }
    }

    // 3. [New Logic] 修改 Input Mvin 和 Output Mvout 的 Row/Col/Stride
    // 不需要分块，但需要修改维度属性：
    // row = dim[1] - 1
    // col = (dim[2] * dim[3] * dim[4]) - 1
    // stride = col + 1
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto modifyMvFlatten = [&](Operation *mv, Value memref) {
      if (!mv)
        return;
      auto memType = cast<MemRefType>(memref.getType());
      auto shape = memType.getShape();

      // 确保维度足够
      if (shape.size() < 5)
        return;

      int64_t dim1 = shape[1];
      int64_t dim234 = shape[2] * shape[3] * shape[4];

      int64_t newRow = dim1 - 1;
      int64_t newCol = dim234 - 1;
      int64_t newStride = newCol + 1;

      // 1. 创建常量 Value (根据 Npux.td 定义，这些字段是 I16)
      Value valRow = b.create<arith::ConstantIntOp>(newRow, 16);
      Value valCol = b.create<arith::ConstantIntOp>(newCol, 16);
      Value valStride = b.create<arith::ConstantIntOp>(newStride, 16);

      // 2. 使用 ODS 生成的强类型接口修改 Operand
      if (auto mvin = dyn_cast<DmaMvinOp>(mv)) {
        mvin.getColNumMutable().assign(valCol);
        mvin.getRowNumMutable().assign(valRow);
        mvin.getSramStrideMutable().assign(valStride);
        mvin.getDramStrideMutable().assign(valStride);
      } else if (auto mvout = dyn_cast<DmaMvoutOp>(mv)) {
        mvout.getColNumMutable().assign(valCol);
        mvout.getRowNumMutable().assign(valRow);
        mvout.getSramStrideMutable().assign(valStride);
        mvout.getDramStrideMutable().assign(valStride);
      }
    };

    // 3.1 修改 Input A 对应的 Mvin
    OpChain inputChain = traceMvinChain(op.getInputA(), scope);
    if (inputChain.mvinOp) {
      modifyMvFlatten(inputChain.mvinOp, op.getInputA());
    }

    // 3.2 修改 Output 对应的 Mvout
    if (mvoutOp) {
      modifyMvFlatten(mvoutOp, accToSpmOp.getSpmDst());
    }

    // =========================================================
    // 下面开始原有的 Split 逻辑 (处理 Bias, Weight, Compute 循环)
    // =========================================================

    // 设定拆分参数 (假设拆分 Dim 0: Output Channel)
    const int64_t SPLIT_DIM = 0;

    Value outputBuffer = op.getOutput();
    auto memType = cast<MemRefType>(outputBuffer.getType());
    int64_t totalSize =
        memType.getDimSize(1); // 注意：这里取的是 output channel 维度
    int64_t stepSize = 1;

    if (totalSize <= stepSize)
      return failure();

    Location loc = op.getLoc();

    Value c0 = b.create<arith::ConstantIndexOp>(0);
    Value cTotal = b.create<arith::ConstantIndexOp>(totalSize);
    Value cStep = b.create<arith::ConstantIndexOp>(stepSize);
    Value cTrue = b.create<arith::ConstantIntOp>(1, 1);
    Value cFalse = b.create<arith::ConstantIntOp>(0, 1);

    // 4. 创建循环
    auto loop = rewriter.create<scf::ForOp>(loc, c0, cTotal, cStep,
        ValueRange{},
        [&](OpBuilder &ib, Location loc, Value innerIv, ValueRange) {
          ImplicitLocOpBuilder nb(loc, ib);

          // ---------------------------------------------------------
          // Step A: Bias Mvin (在循环内，每次切片搬运)
          // ---------------------------------------------------------
          Value currentBias = Value();
          if (biasChain.mvinOp) {
            const int64_t BIAS_SPLIT_DIM = 0;
            Value biasSramSlice = createSramSlice(
                nb, biasChain.sramBuffer, innerIv, BIAS_SPLIT_DIM);

            Value biasDramSlice;
            if (biasChain.dramSubview) {
              biasDramSlice = createDramSliceOffset(
                  nb, biasChain.dramSubview, c0, innerIv, c0, BIAS_SPLIT_DIM);
            } else {
              Value rawHostPtr = biasChain.mvinOp.getHostPtr();
              biasDramSlice =
                  createSramSlice(nb, rawHostPtr, innerIv, BIAS_SPLIT_DIM);
            }

            

            auto newBiasMvin = cast<DmaMvinOp>(ib.clone(*biasChain.mvinOp));
            newBiasMvin.getHostPtrMutable().assign(biasDramSlice);
            newBiasMvin.getDstMemrefMutable().assign(biasSramSlice);
            newBiasMvin.getIsBiasMutable().assign(cTrue);

            currentBias = biasSramSlice;
          }

          // ---------------------------------------------------------
          // Step B: Compute Run (Input A 保持全量)
          // ---------------------------------------------------------
          // Input: 通常全量广播，不需要切 (Input Mvin
          // 已经在循环外被修改过属性了)
          Value inputFull = op.getInputA();

          // Weight: 需要切对应 OC 的部分
          Value weightSlice =
              createSramSlice(nb, op.getInputB(), innerIv, SPLIT_DIM);

          // Output: 切对应 OC 的 Acc Buffer
          Value outputSlice = createSramSlice(nb, op.getOutput(), innerIv, 1);

          // Clone Compute
          auto newComp = cast<ComputeRunOp>(ib.clone(*op));
          newComp.getInputAMutable().assign(inputFull);
          newComp.getInputBMutable().assign(weightSlice);
          newComp.getOutputMutable().assign(outputSlice);

          if (currentBias) {
            newComp.getBiaspsumMemrefMutable().assign(currentBias);
          }

          // 设置 Single 模式逻辑
          newComp.getIsAccumulateMutable().assign(cFalse);
          newComp.getAccBiasMutable().assign(currentBias ? cTrue : cFalse);

          // ---------------------------------------------------------
          // Step C: Acc To Spm (在循环内，算完立刻搬)
          // ---------------------------------------------------------
          Value spmDstSlice =
              createSramSlice(nb, accToSpmOp.getSpmDst(), innerIv, 1);

          auto newAcc = cast<MvAccToSpmOp>(ib.clone(*accToSpmOp));
          newAcc.getAccSrcMutable().assign(outputSlice);
          newAcc.getSpmDstMutable().assign(spmDstSlice);

          ib.create<scf::YieldOp>(loc);
        });

    // 5. 清理旧 Op
    if (accToSpmOp)
      rewriter.eraseOp(accToSpmOp);
    if (biasChain.mvinOp)
      rewriter.eraseOp(biasChain.mvinOp);
    rewriter.eraseOp(op);

    return success();
  }
};

struct SplitConvIcPass
    : public PassWrapper<SplitConvIcPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitConvIcPass)
  llvm::StringRef getArgument() const override { return "split-conv-ic"; }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<SplitConvIcPattern, SplitIndependentConvPattern>(context);
    if (failed(applyPatternsGreedily(
            getOperation().getBody(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> npux::createSplitConvIcPass() {
  return std::make_unique<SplitConvIcPass>();
}