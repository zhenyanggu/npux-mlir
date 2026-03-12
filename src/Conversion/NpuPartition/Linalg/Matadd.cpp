//==============================================================
// src/Conversion/NpuPartition/Linalg/Matadd.cpp
// This file implements ONNX Add -> linalg.generic(npu_matadd)
//==============================================================

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/NpuPartition/LinalgConversionHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include <cmath>

using namespace mlir;
using namespace npux;

namespace {

static RankedTensorType addEncoding1(RankedTensorType type, OpBuilder &b) {
  if (type.getEncoding()) {
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(type.getEncoding())) {
      if (intAttr.getInt() == 1)
        return type;
    }
  }
  return RankedTensorType::get(
      type.getShape(), type.getElementType(), b.getI64IntegerAttr(1));
}

static bool hasSameStaticShape(
    RankedTensorType a, RankedTensorType b, RankedTensorType c) {
  if (!a || !b || !c)
    return false;
  if (!a.hasStaticShape() || !b.hasStaticShape() || !c.hasStaticShape())
    return false;
  if (a.getRank() != b.getRank() || a.getRank() != c.getRank())
    return false;
  if (a.getRank() < 1 || a.getRank() > 2)
    return false;

  for (int64_t i = 0; i < a.getRank(); ++i) {
    if (a.getDimSize(i) != b.getDimSize(i) || a.getDimSize(i) != c.getDimSize(i))
      return false;
  }
  return true;
}

static void createMataddBody(OpBuilder &b, Location loc, ValueRange args) {
  Value lhs = args[0];
  Value rhs = args[1];
  Value outInit = args[2];
  Type outElemType = outInit.getType();

  auto toI32 = [&](Value v) -> Value {
    Type t = v.getType();
    if (t.isInteger(32))
      return v;
    if (t.isInteger(8) || t.isInteger(16) || t.isInteger(1))
      return b.create<arith::ExtSIOp>(loc, b.getI32Type(), v);
    return v;
  };

  Value sum = b.create<arith::AddIOp>(loc, toI32(lhs), toI32(rhs));
  Value res = sum;
  if (outElemType.isInteger(8))
    res = b.create<arith::TruncIOp>(loc, outElemType, sum);
  else if (outElemType.isInteger(32))
    res = sum;

  b.create<linalg::YieldOp>(loc, res);
}

struct AddToLinalgMatadd : public OpConversionPattern<ONNXAddOp> {
  using OpConversionPattern<ONNXAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto lhsDq = op.getA().getDefiningOp<ONNXDequantizeLinearOp>();
    auto rhsDq = op.getB().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!lhsDq || !rhsDq)
      return failure();

    if (!op.getResult().hasOneUse())
      return failure();
    auto outQ = dyn_cast<ONNXQuantizeLinearOp>(*op.getResult().getUsers().begin());
    if (!outQ)
      return failure();

    // Keep this first version strict/safe: symmetric int8 path only.
    auto lhsParams = getScalarQuantParams(lhsDq);
    auto rhsParams = getScalarQuantParams(rhsDq);
    auto outParams = getScalarQuantParams(outQ);
    if (std::abs(lhsParams.scale - rhsParams.scale) > 1e-6)
      return failure();
    if (lhsParams.zeroPoint != 0 || rhsParams.zeroPoint != 0)
      return failure();

    Value lhsQ = lhsDq.getX();
    Value rhsQ = rhsDq.getX();

    auto lhsType = dyn_cast<RankedTensorType>(lhsQ.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhsQ.getType());
    auto outType = dyn_cast<RankedTensorType>(outQ.getResult().getType());
    if (!hasSameStaticShape(lhsType, rhsType, outType))
      return failure();
    if (!lhsType.getElementType().isInteger(8) ||
        !rhsType.getElementType().isInteger(8) ||
        !outType.getElementType().isInteger(8))
      return failure();

    RankedTensorType lhsEncType = addEncoding1(lhsType, rewriter);
    RankedTensorType rhsEncType = addEncoding1(rhsType, rewriter);
    RankedTensorType outEncType = addEncoding1(outType, rewriter);

    lhsQ.setType(lhsEncType);
    rhsQ.setType(rhsEncType);

    SmallVector<Value> dynSizes =
        getDynamicSizes(rewriter, op.getLoc(), lhsQ, outEncType.getShape());
    Value outAlloc =
        rewriter.create<bufferization::AllocTensorOp>(op.getLoc(), outEncType, dynSizes);

    int64_t rank = outType.getRank();
    SmallVector<AffineMap, 3> maps = {
        rewriter.getMultiDimIdentityMap(rank),
        rewriter.getMultiDimIdentityMap(rank),
        rewriter.getMultiDimIdentityMap(rank)};
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);

    auto mataddOp = rewriter.create<linalg::GenericOp>(op.getLoc(),
        outEncType, ValueRange{lhsQ, rhsQ}, ValueRange{outAlloc}, maps, iters,
        createMataddBody);

    mataddOp->setAttr("library_call", rewriter.getStringAttr("npu_matadd"));
    mataddOp->setAttr("npu.target", rewriter.getStringAttr("npu"));
    mataddOp->setAttr("lhs_scale", rewriter.getF32FloatAttr(lhsParams.scale));
    mataddOp->setAttr(
        "lhs_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), lhsParams.zeroPoint));
    mataddOp->setAttr("rhs_scale", rewriter.getF32FloatAttr(rhsParams.scale));
    mataddOp->setAttr(
        "rhs_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), rhsParams.zeroPoint));
    mataddOp->setAttr("out_scale", rewriter.getF32FloatAttr(outParams.scale));
    mataddOp->setAttr(
        "out_zp", rewriter.getIntegerAttr(rewriter.getI32Type(), outParams.zeroPoint));

    outQ.getResult().setType(mataddOp.getResult(0).getType());
    rewriter.replaceOp(outQ, mataddOp.getResult(0));
    rewriter.eraseOp(op);
    if (lhsDq->use_empty())
      rewriter.eraseOp(lhsDq);
    if (rhsDq->use_empty())
      rewriter.eraseOp(rhsDq);
    return success();
  }
};

} // namespace

void npux::populateLinalgMataddPattern(RewritePatternSet &patterns) {
  patterns.add<AddToLinalgMatadd>(patterns.getContext());
}
