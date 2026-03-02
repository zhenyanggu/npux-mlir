//=============================================================
// src/Conversion/NpuTiling/ConvInnerTiling.cpp
//=============================================================

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Conversion/NpuTiling/NpuTilingHelper.hpp"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::utils;

namespace {

enum class LoopStage { Head, Body, Tail, Single };

StringRef getStageAttrName(LoopStage stage) {
  switch (stage) {
  case LoopStage::Head:
    return "head";
  case LoopStage::Body:
    return "body";
  case LoopStage::Tail:
    return "tail";
  case LoopStage::Single:
    return "single";
  }
  return "";
}

SmallVector<AffineMap> getAffineMapsFromArrayAttr(ArrayAttr mapAttr) {
  SmallVector<AffineMap> maps;
  for (auto attr : mapAttr) {
    maps.push_back(cast<AffineMapAttr>(attr).getValue());
  }
  return maps;
}

void copyAttributes(Operation *src, Operation *dst) {
  for (const NamedAttribute &attr : src->getAttrs()) {
    StringRef name = attr.getName();
    if (name == "indexing_maps" || name == "iterator_types" ||
        name == "operand_segment_sizes" || name == "operandSegmentSizes") {
      continue;
    }
    dst->setAttr(name, attr.getValue());
  }
}

// 这个函数保持你提供的逻辑不变
Value createPeeledLoopBody(OpBuilder &b, scf::ForOp originalLoop, Value iv,
    ValueRange iterArgs, LoopStage stage, IRMapping &mapping) {

  Block *originalBody = originalLoop.getBody();
  Value stageResult = nullptr;

  mapping.map(originalLoop.getInductionVar(), iv);

  for (auto &op : originalBody->without_terminator()) {

    // --- 1. ExtractSlice ---
    if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      if (extractOp.getSource() == originalLoop.getRegionIterArgs()[0]) {
        if (stage == LoopStage::Head || stage == LoopStage::Single) {
          auto resultType = cast<RankedTensorType>(extractOp.getType());
          auto i32TensorType = resultType.clone(b.getI32Type());

          auto allocTensorOp =
              b.create<bufferization::AllocTensorOp>(op.getLoc(), i32TensorType,
                  ValueRange{}, Value{}, b.getI64IntegerAttr(3));
          Value emptyI32 = allocTensorOp.getResult();

          mapping.map(extractOp.getResult(), emptyI32);
        } else {
          mapping.map(extractOp.getResult(), iterArgs[0]);
        }
        continue;
      }
    }

    // --- 2. Conv (LinalgGeneric) ---
    if (isa<GenericOp>(op)) {
      auto genericOp = cast<GenericOp>(op);

      SmallVector<Value> newInputs;
      for (Value in : genericOp.getInputs()) {
        newInputs.push_back(mapping.lookup(in));
      }

      SmallVector<Value> newOutputs;
      SmallVector<Type> newResultTypes;
      SmallVector<AffineMap> newMaps;

      auto originalMaps =
          getAffineMapsFromArrayAttr(genericOp.getIndexingMapsAttr());

      // Branch 1: Head / Single
      if (stage == LoopStage::Head || stage == LoopStage::Single) {
        Value outTensor = mapping.lookup(genericOp.getOutputs()[0]);
        newOutputs.push_back(outTensor);
        newResultTypes.push_back(outTensor.getType());
        newMaps = originalMaps;
      }
      // Branch 2: Body / Tail
      else {
        if (newInputs.size() > 2) {
          newInputs.pop_back(); // Remove Bias
        }

        Value accTensor = iterArgs[0];
        newOutputs.push_back(accTensor);
        newResultTypes.push_back(accTensor.getType());

        if (originalMaps.size() > 2) {
          newMaps = originalMaps;
          newMaps.erase(newMaps.begin() + 2); // Remove Bias map
        }
      }

      auto newGenericOp = b.create<GenericOp>(op.getLoc(), newResultTypes,
          newInputs, newOutputs, newMaps, genericOp.getIteratorTypesArray());

      copyAttributes(genericOp, newGenericOp);
      newGenericOp->setAttr(
          "npu.loop_stage", b.getStringAttr(getStageAttrName(stage)));

      newGenericOp->setAttr("npu.inner_tiled", b.getUnitAttr());

      Region &region = newGenericOp.getRegion();
      Block *block = new Block();
      region.push_back(block);

      Type i32 = b.getI32Type();
      Type i8 = b.getI8Type();

      if (stage == LoopStage::Head || stage == LoopStage::Single) {
        block->addArgument(i8, op.getLoc());
        block->addArgument(i8, op.getLoc());
        block->addArgument(i32, op.getLoc());
        block->addArgument(i32, op.getLoc());
      } else {
        block->addArgument(i8, op.getLoc());
        block->addArgument(i8, op.getLoc());
        block->addArgument(i32, op.getLoc());
      }

      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(block);

        Value argIn = block->getArgument(0);
        Value argWt = block->getArgument(1);
        Value argThird = block->getArgument(2);

        Value extIn = b.create<arith::ExtSIOp>(op.getLoc(), i32, argIn);
        Value extWt = b.create<arith::ExtSIOp>(op.getLoc(), i32, argWt);
        Value prod = b.create<arith::MulIOp>(op.getLoc(), extIn, extWt);
        Value resultSum = b.create<arith::AddIOp>(op.getLoc(), prod, argThird);

        b.create<linalg::YieldOp>(op.getLoc(), resultSum);
      }

      mapping.map(op.getResult(0), newGenericOp.getResult(0));
      stageResult = newGenericOp.getResult(0);
      continue;
    }

    // --- 3. InsertSlice ---
    if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
      if (stage == LoopStage::Tail || stage == LoopStage::Single) {
        Value srcI32 = mapping.lookup(insertOp.getSource());
        auto srcType = cast<RankedTensorType>(srcI32.getType());
        auto targetType = srcType.clone(b.getI8Type());

        auto memSpace2Attr = b.getI64IntegerAttr(2);
        Value alloc = b.create<bufferization::AllocTensorOp>(
            op.getLoc(), targetType, ValueRange{}, Value{}, memSpace2Attr);

        SmallVector<utils::IteratorType> iteratorTypes(
            targetType.getRank(), utils::IteratorType::parallel);

        AffineMap identityMap = AffineMap::getMultiDimIdentityMap(
            targetType.getRank(), b.getContext());
        SmallVector<AffineMap> indexingMaps(2, identityMap);

        auto mvOp = b.create<linalg::GenericOp>(op.getLoc(), targetType,
            /*inputs=*/ValueRange{srcI32},
            /*outputs=*/ValueRange{alloc},
            /*indexingMaps=*/indexingMaps,
            /*iteratorTypes=*/iteratorTypes,
            /*bodyBuilder=*/
            [&](OpBuilder &builder, Location nestedLoc, ValueRange args) {
              Value trunc = builder.create<arith::TruncIOp>(
                  nestedLoc, b.getI8Type(), args[0]);
              builder.create<linalg::YieldOp>(nestedLoc, trunc);
            });

        mvOp->setAttr("library_call", b.getStringAttr("mv_acc_to_spm"));
        mvOp->setAttr("npu.target", b.getStringAttr("npu"));

        Value destTensor = originalLoop.getInitArgs()[0];
        if (stage == LoopStage::Single)
          destTensor = iterArgs[0];

        auto newInsert = b.create<tensor::InsertSliceOp>(op.getLoc(),
            mvOp.getResult(0), destTensor, insertOp.getMixedOffsets(),
            insertOp.getMixedSizes(), insertOp.getMixedStrides());

        mapping.map(op.getResult(0), newInsert.getResult());
        stageResult = newInsert.getResult();
      } else {
        Value srcI32 = mapping.lookup(insertOp.getSource());
        stageResult = srcI32;
      }
      continue;
    }

    Operation *clonedOp = b.clone(op, mapping);
  }
  return stageResult;
}

// --- 修改部分：将匹配目标从 scf::ForOp 改为 linalg::GenericOp ---
struct NpuConvInnerTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp convOp, PatternRewriter &rewriter) const override {


    if (convOp->hasAttr("npu.inner_tiled")) {
      return failure();
    }

    // 1. 检查 library_call 是否为 "npu_conv"
    auto libraryCall = convOp->getAttrOfType<StringAttr>("library_call");
    if (!libraryCall || libraryCall.getValue() != "npu_conv") {
      return failure();
    }

    // 2. 向上查找最近的 scf::ForOp
    auto loopOp = convOp->getParentOfType<scf::ForOp>();
    if (!loopOp) {
      return failure();
    }

    // 3. 检查循环维度是否为 "IC"
    // 注意：我们检查的是包含这个 conv 的循环
    if (auto loopDimAttr = loopOp->getAttrOfType<StringAttr>("npu.loop_dim")) {
      if (loopDimAttr.getValue() != "IC")
        return failure();
    } else {
      return failure();
    }

    // 接下来的逻辑是对 loopOp 进行操作，变量名 op 在下方指代 loopOp
    scf::ForOp op = loopOp;

    Location loc = op.getLoc();
    Value lb = op.getLowerBound();
    Value ub = op.getUpperBound();
    Value step = op.getStep();
    Value initOutput = op.getInitArgs()[0];

    std::optional<int64_t> staticLb = getConstantIntValue(lb);
    std::optional<int64_t> staticUb = getConstantIntValue(ub);
    std::optional<int64_t> staticStep = getConstantIntValue(step);
    bool isStatic =
        staticLb.has_value() && staticUb.has_value() && staticStep.has_value();

    int64_t constNumIters = -1;
    if (isStatic) {
      int64_t diff = *staticUb - *staticLb;
      constNumIters = (diff + *staticStep - 1) / *staticStep;
    }

    // Case A: Single
    if (isStatic && constNumIters == 1) {
      IRMapping mapping;
      Value res = createPeeledLoopBody(
          rewriter, op, lb, {initOutput}, LoopStage::Single, mapping);
      rewriter.replaceOp(op, res); // 替换的是 Loop，不是 Conv
      return success();
    }

    // Case B: Static Multi (Head -> Body -> Tail)
    if (isStatic && constNumIters > 1) {
      // 1. Head
      IRMapping mapHead;
      Value headAccI32 =
          createPeeledLoopBody(rewriter, op, lb, {}, LoopStage::Head, mapHead);

      Value currentAcc = headAccI32;

      // 2. Body
      if (constNumIters > 2) {
        int64_t lastIterIdxVal = *staticLb + (constNumIters - 1) * *staticStep;
        Value bodyLb = rewriter.create<arith::ConstantIndexOp>(
            loc, *staticLb + *staticStep);
        Value lastIdx =
            rewriter.create<arith::ConstantIndexOp>(loc, lastIterIdxVal);

        auto bodyLoop = rewriter.create<scf::ForOp>(loc, bodyLb, lastIdx, step,
            ValueRange{currentAcc},
            [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
              IRMapping mapBody;
              Value res = createPeeledLoopBody(
                  b, op, iv, args, LoopStage::Body, mapBody);
              b.create<scf::YieldOp>(loc, res);
            });

        bodyLoop->setAttrs(op->getAttrs());
        bodyLoop->removeAttr("npu.loop_dim");

        currentAcc = bodyLoop.getResult(0);
      }

      // 3. Tail
      int64_t lastIterIdxVal = *staticLb + (constNumIters - 1) * *staticStep;
      Value lastIdx =
          rewriter.create<arith::ConstantIndexOp>(loc, lastIterIdxVal);
      IRMapping mapTail;
      Value finalRes = createPeeledLoopBody(
          rewriter, op, lastIdx, {currentAcc}, LoopStage::Tail, mapTail);

      rewriter.replaceOp(op, finalRes);
      return success();
    }

    // Case C: Dynamic (Runtime Check)
    {
      Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value diff = rewriter.create<arith::SubIOp>(loc, ub, lb);
      Value stepMinus1 = rewriter.create<arith::SubIOp>(loc, step, c1);
      Value numPlus = rewriter.create<arith::AddIOp>(loc, diff, stepMinus1);
      Value numIters = rewriter.create<arith::DivUIOp>(loc, numPlus, step);

      Value isSingle = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, numIters, c1);
      auto ifOp =
          rewriter.create<scf::IfOp>(loc, op.getResultTypes(), isSingle, true);

      // Branch 1: Single
      OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
      {
        IRMapping mapSingle;
        Value res = createPeeledLoopBody(
            thenBuilder, op, lb, {initOutput}, LoopStage::Single, mapSingle);
        thenBuilder.create<scf::YieldOp>(loc, res);
      }

      // Branch 2: Multi
      OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
      {
        IRMapping mapHead;
        Value headAccI32 = createPeeledLoopBody(
            elseBuilder, op, lb, {}, LoopStage::Head, mapHead);

        // Body Loop
        Value nMinus1 = elseBuilder.create<arith::SubIOp>(loc, numIters, c1);
        Value totalStepOffset =
            elseBuilder.create<arith::MulIOp>(loc, nMinus1, step);
        Value lastIdx =
            elseBuilder.create<arith::AddIOp>(loc, lb, totalStepOffset);
        Value bodyLb = elseBuilder.create<arith::AddIOp>(loc, lb, step);

        auto bodyLoop = elseBuilder.create<scf::ForOp>(loc, bodyLb, lastIdx,
            step, ValueRange{headAccI32},
            [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
              IRMapping mapBody;
              Value res = createPeeledLoopBody(
                  b, op, iv, args, LoopStage::Body, mapBody);
              b.create<scf::YieldOp>(loc, res);
            });
        bodyLoop->setAttrs(op->getAttrs());
        bodyLoop->removeAttr("npu.loop_dim");

        Value bodyResultAcc = bodyLoop.getResult(0);

        IRMapping mapTail;
        Value tailRes = createPeeledLoopBody(elseBuilder, op, lastIdx,
            {bodyResultAcc}, LoopStage::Tail, mapTail);
        elseBuilder.create<scf::YieldOp>(loc, tailRes);
      }

      rewriter.replaceOp(op, ifOp.getResult(0));
      return success();
    }
  }
};

} // namespace

void npux::populateConvInnerTilingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<NpuConvInnerTilingPattern>(context);
}