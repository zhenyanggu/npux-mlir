//==============================================================
// src/Conversion/NpuPartition/Linalg/Gemm.cpp
// this file implements the conversion of ONNX Gemm operation(including
// QLinearMatMul&Gemm) to linalg operations for NPU partitioning.
//==============================================================

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/NpuPartition/LinalgConversionHelper.hpp"
#include "llvm/ADT/SmallPtrSet.h" 

using namespace mlir;
using namespace npux;

namespace {

// =============================================================================
// Helper: Extract quantization parameters
// =============================================================================
QuantizationParam getQuantParamsFromOperands(
    Operation *op, int scaleIdx, int zpIdx) {
  QuantizationParam params;
  params.scale = 1.0;
  params.zeroPoint = 0;

  if (auto scaleAttr = getConstAttrFromOperand(op, scaleIdx)) {
    if (scaleAttr.isSplat()) {
      params.scale = scaleAttr.getSplatValue<float>();
    }
  }

  if (auto zpAttr = getConstAttrFromOperand(op, zpIdx)) {
    if (zpAttr.isSplat()) {
      Type zpType = zpAttr.getElementType();
      if (zpType.isUnsignedInteger(8)) {
        params.zeroPoint = (int64_t)zpAttr.getSplatValue<uint8_t>();
      } else if (zpType.isInteger(8)) {
        params.zeroPoint = (int64_t)zpAttr.getSplatValue<int8_t>();
      } else if (zpType.isInteger(32)) {
        params.zeroPoint = (int64_t)zpAttr.getSplatValue<int32_t>();
      }
    }
  }
  return params;
}

// Helper: Check if the consumer can absorb quantization
bool isAbsorbableConsumer(Operation *op) {
  return llvm::isa<ONNXConvOp, ONNXLayerNormalizationOp, ONNXSoftmaxOp,
      ONNXGeluOp, ONNXQLinearMatMulOp, ONNXGemmOp>(op);
}

// =============================================================================
// Pattern 1: GemmToLinalg
// =============================================================================
struct GemmToLinalg : public OpConversionPattern<ONNXGemmOp> {
  using OpConversionPattern<ONNXGemmOp>::OpConversionPattern;

  struct InputConfig {
    Value externalValue;            
    ONNXConstantOp constOpToCopy;   
    ONNXQuantizeLinearOp quantOpToMove; 
    
    bool needsTranspose;
    SmallVector<int64_t> perm;
    
    ONNXDequantizeLinearOp absorbedDq;
  };

  LogicalResult matchAndRewrite(ONNXGemmOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    // 1. 分析输入 A
    InputConfig configA;
    configA.constOpToCopy = nullptr;
    configA.quantOpToMove = nullptr;
    configA.absorbedDq = nullptr;
    configA.needsTranspose = (op.getTransA() != 0);
    configA.perm = {1, 0};

    Value rawA = op.getA();
    double aScale = 1.0;
    int64_t aZp = 0;

    if (auto dqOp = rawA.getDefiningOp<ONNXDequantizeLinearOp>()) {
      configA.absorbedDq = dqOp;
      auto params = getScalarQuantParams(dqOp);
      aScale = params.scale;
      aZp = params.zeroPoint;
      rawA = dqOp.getX(); 
    }

    Operation *producerA = rawA.getDefiningOp();
    if (producerA && llvm::isa<ONNXConstantOp>(producerA)) {
      configA.constOpToCopy = llvm::cast<ONNXConstantOp>(producerA);
    } else if (producerA && llvm::isa<ONNXQuantizeLinearOp>(producerA)) {
      configA.quantOpToMove = llvm::cast<ONNXQuantizeLinearOp>(producerA);
    } else {
      configA.externalValue = rawA;
    }


    // 2. 分析输入 B
    InputConfig configB;
    configB.constOpToCopy = nullptr;
    configB.quantOpToMove = nullptr;
    configB.absorbedDq = nullptr;
    configB.needsTranspose = (op.getTransB() != 0);
    configB.perm = {1, 0};

    Value rawB = op.getB();
    double bScale = 1.0;
    int64_t bZp = 0;

    if (auto dqOp = rawB.getDefiningOp<ONNXDequantizeLinearOp>()) {
      configB.absorbedDq = dqOp;
      auto params = getScalarQuantParams(dqOp);
      bScale = params.scale;
      bZp = params.zeroPoint;
      rawB = dqOp.getX();
    }

    Operation *producerB = rawB.getDefiningOp();
    if (producerB && llvm::isa<ONNXConstantOp>(producerB)) {
      configB.constOpToCopy = llvm::cast<ONNXConstantOp>(producerB);
    } else if (producerB && llvm::isa<ONNXQuantizeLinearOp>(producerB)) {
      configB.quantOpToMove = llvm::cast<ONNXQuantizeLinearOp>(producerB);
    } else {
      configB.externalValue = rawB;
    }


    // 3. 分析输出
    if (!op.getResult().hasOneUse())
      return failure();
    Operation *userOp = *op.getResult().getUsers().begin();
    auto outQuantOp = mlir::dyn_cast<ONNXQuantizeLinearOp>(userOp);
    if (!outQuantOp)
      return failure();

    auto qParams = getScalarQuantParams(outQuantOp);
    double outScale = qParams.scale;
    int64_t outZp = qParams.zeroPoint;

    ONNXDequantizeLinearOp outDqOp = nullptr;
    bool shouldPullDqIntoRegion = false;

    if (outQuantOp.getResult().hasOneUse()) {
      Operation *nextUser = *outQuantOp.getResult().getUsers().begin();
      if (auto dq = mlir::dyn_cast<ONNXDequantizeLinearOp>(nextUser)) {
        outDqOp = dq;
        bool downstreamAbsorbsDQ = false;
        for (auto *dqUser : dq.getResult().getUsers()) {
            if (isAbsorbableConsumer(dqUser)) {
                downstreamAbsorbsDQ = true;
                break;
            }
        }
        if (!downstreamAbsorbsDQ) {
            shouldPullDqIntoRegion = true;
        }
      }
    }

    Type regionResultType;
    if (shouldPullDqIntoRegion) {
      regionResultType = outDqOp.getResult().getType(); 
    } else {
      regionResultType = outQuantOp.getResult().getType(); 
    }

    // ============================================================
    // 创建 Region
    // ============================================================
    auto executeRegion = rewriter.create<scf::ExecuteRegionOp>(
        loc, regionResultType);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&executeRegion.getRegion());

      // --------------------------------------------------------
      // 处理 Value A
      // --------------------------------------------------------
      Value valA;
      if (configA.constOpToCopy) {
        Value upstreamA = configA.constOpToCopy.getResult(); 
        auto tensorType = mlir::cast<RankedTensorType>(upstreamA.getType());
        SmallVector<Value> dynSizes = 
            getDynamicSizes(rewriter, loc, upstreamA, tensorType.getShape());
        Value initA = rewriter.create<tensor::EmptyOp>(
            loc, tensorType.getShape(), tensorType.getElementType(), dynSizes);
        auto copyOp = rewriter.create<linalg::CopyOp>(loc, upstreamA, initA);
        valA = copyOp.getResult(0);

      } else if (configA.quantOpToMove) {
        Operation *cloned = rewriter.clone(*configA.quantOpToMove);
        valA = cloned->getResult(0);

      } else {
        valA = configA.externalValue;
      }

      // --------------------------------------------------------
      // 处理 Value B
      // --------------------------------------------------------
      Value valB;
      if (configB.constOpToCopy) {
        Value upstreamB = configB.constOpToCopy.getResult();
        auto tensorType = mlir::cast<RankedTensorType>(upstreamB.getType());
        SmallVector<Value> dynSizes = 
            getDynamicSizes(rewriter, loc, upstreamB, tensorType.getShape());
        Value initB = rewriter.create<tensor::EmptyOp>(
            loc, tensorType.getShape(), tensorType.getElementType(), dynSizes);
        auto copyOp = rewriter.create<linalg::CopyOp>(loc, upstreamB, initB);
        valB = copyOp.getResult(0);

      } else if (configB.quantOpToMove) {
        Operation *cloned = rewriter.clone(*configB.quantOpToMove);
        valB = cloned->getResult(0);

      } else {
        valB = configB.externalValue;
      }

      // --------------------------------------------------------
      // Transpose
      // --------------------------------------------------------
      if (configA.needsTranspose) {
        auto typeA = mlir::cast<RankedTensorType>(valA.getType());
        SmallVector<int64_t> transShape = {
            typeA.getShape()[1], typeA.getShape()[0]};
        auto transType =
            RankedTensorType::get(transShape, typeA.getElementType());
        valA = rewriter.create<ONNXTransposeOp>(
            loc, transType, valA, rewriter.getI64ArrayAttr(configA.perm));
      }

      if (configB.needsTranspose) {
        auto typeB = mlir::cast<RankedTensorType>(valB.getType());
        SmallVector<int64_t> transShape = {
            typeB.getShape()[1], typeB.getShape()[0]};
        auto transType =
            RankedTensorType::get(transShape, typeB.getElementType());
        valB = rewriter.create<ONNXTransposeOp>(
            loc, transType, valB, rewriter.getI64ArrayAttr(configB.perm));
      }

      // --------------------------------------------------------
      // Linalg Matmul
      // --------------------------------------------------------
      auto typeA = mlir::cast<RankedTensorType>(valA.getType());
      auto typeB = mlir::cast<RankedTensorType>(valB.getType());
      auto outQType =
          mlir::cast<RankedTensorType>(outQuantOp.getResult().getType());

      SmallVector<int64_t> outShape = {
          typeA.getShape()[0], typeB.getShape()[1]};
      auto linalgOutType =
          RankedTensorType::get(outShape, outQType.getElementType());

      SmallVector<Value> dynSizes =
          getDynamicSizes(rewriter, loc, valA, outShape);
      Value outputInit =
          rewriter.create<tensor::EmptyOp>(loc, linalgOutType, dynSizes);

      auto linalgOp = rewriter.create<linalg::MatmulOp>(
          loc, 
          linalgOutType,         
          ValueRange{valA, valB}, 
          outputInit              
      );

      // Attributes
      linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_matmul"));
      linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
      linalgOp->setAttr("a_scale", rewriter.getF32FloatAttr(aScale));
      linalgOp->setAttr("a_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), aZp));
      linalgOp->setAttr("b_scale", rewriter.getF32FloatAttr(bScale));
      linalgOp->setAttr("b_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), bZp));
      linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outScale));
      linalgOp->setAttr("out_zp", rewriter.getIntegerAttr(rewriter.getI16Type(), outZp));

      Value regionResult = linalgOp.getResults()[0];

      // ========================================================
      // Bias Add using linalg.generic (Broadcasting)
      // ========================================================
      if (Value bias = op.getC()) {
          // 1. 准备 Bias Input (通过 implicit capture)
          // 通常 Bias 是 [N]，需要广播到 [M, N]
          
          // 2. 准备 Output Buffer for Add (Reuse same shape/type as Matmul result)
          Value addInit = rewriter.create<tensor::EmptyOp>(
              loc, linalgOutType, dynSizes);

          // 3. 构建 Indexing Maps
          // Matmul Result: (m, n) -> (m, n)
          // Bias:          (m, n) -> (n)  <-- Broadcast happens here
          // Output:        (m, n) -> (m, n)
          SmallVector<AffineMap> maps;
          auto m = rewriter.getAffineDimExpr(0);
          auto n = rewriter.getAffineDimExpr(1);
          
          maps.push_back(AffineMap::get(2, 0, {m, n}, rewriter.getContext())); // Input 1
          maps.push_back(AffineMap::get(2, 0, {n}, rewriter.getContext()));    // Input 2 (Bias)
          maps.push_back(AffineMap::get(2, 0, {m, n}, rewriter.getContext())); // Output

          // 4. Iterator Types
          SmallVector<utils::IteratorType> iterators = {
              utils::IteratorType::parallel, utils::IteratorType::parallel
          };

          // 5. Create Generic Op
          auto addOp = rewriter.create<linalg::GenericOp>(
              loc, 
              linalgOutType, // Result Type
              ValueRange{regionResult, bias}, // Inputs
              addInit, // Outputs
              maps,
              iterators,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value in1 = args[0];
                  Value in2 = args[1];
                  // 假设是 float，如果是 int 用 arith::AddIOp
                  Value res = b.create<arith::AddIOp>(loc, in1, in2);
                  b.create<linalg::YieldOp>(loc, res);
              }
          );
          
          regionResult = addOp.getResult(0);
      }

      if (shouldPullDqIntoRegion) {
        Operation *clonedDq = rewriter.clone(*outDqOp);
        clonedDq->setOperand(0, regionResult);
        regionResult = clonedDq->getResult(0);
      }

      rewriter.create<scf::YieldOp>(loc, regionResult);
    }

    // Cleanup / Replacements
    if (shouldPullDqIntoRegion) {
      rewriter.replaceOp(outDqOp, executeRegion.getResults());
      rewriter.eraseOp(outQuantOp);
    } else {
      rewriter.replaceOp(outQuantOp, executeRegion.getResults());
    }

    rewriter.eraseOp(op);

    llvm::SmallPtrSet<Operation*, 4> potentialDeadOps;
    if (configA.absorbedDq) potentialDeadOps.insert(configA.absorbedDq);
    if (configB.absorbedDq) potentialDeadOps.insert(configB.absorbedDq);
    if (configA.constOpToCopy) potentialDeadOps.insert(configA.constOpToCopy);
    if (configB.constOpToCopy) potentialDeadOps.insert(configB.constOpToCopy);
    if (configA.quantOpToMove) potentialDeadOps.insert(configA.quantOpToMove);
    if (configB.quantOpToMove) potentialDeadOps.insert(configB.quantOpToMove);

    for (Operation* deadOp : potentialDeadOps) {
        if (deadOp->use_empty()) {
            rewriter.eraseOp(deadOp);
        }
    }

    return success();
  }
};

// =============================================================================
// Pattern 2: QLinearMatMulToLinalg
// =============================================================================
struct QLinearMatMulToLinalg : public OpConversionPattern<ONNXQLinearMatMulOp> {
  using OpConversionPattern<ONNXQLinearMatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXQLinearMatMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value inputA = op.getA();
    Value inputB = op.getB();

    QuantizationParam aParams = getQuantParamsFromOperands(op, 1, 2);
    QuantizationParam bParams = getQuantParamsFromOperands(op, 4, 5);
    QuantizationParam outParams = getQuantParamsFromOperands(op, 6, 7);

    Operation *producerA = inputA.getDefiningOp();
    bool copyA = producerA && llvm::isa<ONNXConstantOp>(producerA);
    bool moveA = producerA && llvm::isa<ONNXQuantizeLinearOp>(producerA);

    Operation *producerB = inputB.getDefiningOp();
    bool copyB = producerB && llvm::isa<ONNXConstantOp>(producerB);
    bool moveB = producerB && llvm::isa<ONNXQuantizeLinearOp>(producerB);

    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());

    auto executeRegion =
        rewriter.create<scf::ExecuteRegionOp>(loc, outputType); 
    
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&executeRegion.getRegion());

      Value valA;
      if (copyA) {
        Value upstreamA = producerA->getResult(0);
        auto tensorType = mlir::cast<RankedTensorType>(upstreamA.getType());
        SmallVector<Value> dynSizes = 
            getDynamicSizes(rewriter, loc, upstreamA, tensorType.getShape());
        Value initA = rewriter.create<tensor::EmptyOp>(
            loc, tensorType.getShape(), tensorType.getElementType(), dynSizes);
        valA = rewriter.create<linalg::CopyOp>(loc, upstreamA, initA).getResult(0);
      } else if (moveA) {
        valA = rewriter.clone(*producerA)->getResult(0);
      } else {
        valA = inputA; 
      }

      Value valB;
      if (copyB) {
        Value upstreamB = producerB->getResult(0);
        auto tensorType = mlir::cast<RankedTensorType>(upstreamB.getType());
        SmallVector<Value> dynSizes = 
            getDynamicSizes(rewriter, loc, upstreamB, tensorType.getShape());
        Value initB = rewriter.create<tensor::EmptyOp>(
            loc, tensorType.getShape(), tensorType.getElementType(), dynSizes);
        valB = rewriter.create<linalg::CopyOp>(loc, upstreamB, initB).getResult(0);
      } else if (moveB) {
        valB = rewriter.clone(*producerB)->getResult(0);
      } else {
        valB = inputB;
      }

      SmallVector<Value> dynSizes =
          getDynamicSizes(rewriter, loc, valA, outputType.getShape());
      Value outputInit =
          rewriter.create<tensor::EmptyOp>(loc, outputType, dynSizes);

      auto linalgOp = rewriter.create<linalg::MatmulOp>(
          loc, 
          outputType,            
          ValueRange{valA, valB}, 
          outputInit              
      );

      linalgOp->setAttr("library_call", rewriter.getStringAttr("npu_matmul"));
      linalgOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
      linalgOp->setAttr("a_scale", rewriter.getF32FloatAttr(aParams.scale));
      linalgOp->setAttr("a_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), aParams.zeroPoint));
      linalgOp->setAttr("b_scale", rewriter.getF32FloatAttr(bParams.scale));
      linalgOp->setAttr("b_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), bParams.zeroPoint));
      linalgOp->setAttr("out_scale", rewriter.getF32FloatAttr(outParams.scale));
      linalgOp->setAttr("out_zp", rewriter.getIntegerAttr(rewriter.getI16Type(), outParams.zeroPoint));

      rewriter.create<scf::YieldOp>(loc, linalgOp.getResults());
    }

    rewriter.replaceOp(op, executeRegion.getResults());

    llvm::SmallPtrSet<Operation*, 4> qOpsToClean;
    if (moveA || copyA) qOpsToClean.insert(producerA);
    if (moveB || copyB) qOpsToClean.insert(producerB);

    for (Operation* deadOp : qOpsToClean) {
        if (deadOp->use_empty()) rewriter.eraseOp(deadOp);
    }

    return success();
  }
};

} // namespace

void npux::populateLinalgGemmPattern(RewritePatternSet &patterns) {
  patterns.add<GemmToLinalg>(patterns.getContext());
  patterns.add<QLinearMatMulToLinalg>(patterns.getContext());
}