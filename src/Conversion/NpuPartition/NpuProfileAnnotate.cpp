/*
 * SPDX-License-Identifier: Apache-2.0
 */

//=============================================================================
// /src/Conversion/NpuPartition/NpuProfileAnnotate.cpp
// Annotate ONNX logical layers with profiling boundaries and emit a manifest.
//=============================================================================

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "src/Conversion/NpuPartition/LinalgConversionHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kLayerNameAttr = "npu.layer_name";
constexpr llvm::StringLiteral kLayerKindAttr = "npu.layer_kind";
constexpr llvm::StringLiteral kFusedOpsAttr = "npu.fused_ops";
constexpr llvm::StringLiteral kOriginOpTypeAttr = "npu.origin_op_type";
constexpr llvm::StringLiteral kNodeNameAttr = "onnx_node_name";

struct LayerSlice {
  int64_t layerId = -1;
  Operation *beginOp = nullptr;
  Operation *endOp = nullptr;
  SmallVector<Operation *> ops;
  SmallVector<Operation *> taggedOps;
  std::string layerName;
  std::string originOpType;
  std::string device;
  SmallVector<std::string> fusedOps;
  std::string fallbackReason;
};

static bool isRankedShapedType(Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  return shapedType && shapedType.hasRank();
}

static std::string stringifyType(Type type) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  type.print(os);
  return os.str();
}

static std::optional<int64_t> multiplyChecked(
    std::optional<int64_t> lhs, int64_t rhs) {
  if (!lhs)
    return std::nullopt;
  return *lhs * rhs;
}

static std::optional<int64_t> getNumElements(Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType || !shapedType.hasStaticShape())
    return std::nullopt;
  int64_t numElements = 1;
  for (int64_t dim : shapedType.getShape())
    numElements *= dim;
  return numElements;
}

static std::optional<int64_t> getTypeByteSize(Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType || !shapedType.hasStaticShape())
    return std::nullopt;
  auto numElements = getNumElements(type);
  if (!numElements)
    return std::nullopt;

  Type elementType = shapedType.getElementType();
  unsigned bitWidth = 0;
  if (auto intType = dyn_cast<IntegerType>(elementType))
    bitWidth = intType.getWidth();
  else if (auto floatType = dyn_cast<FloatType>(elementType))
    bitWidth = floatType.getWidth();
  else if (elementType.isIndex())
    bitWidth = 64;
  else
    return std::nullopt;

  return *numElements * static_cast<int64_t>((bitWidth + 7) / 8);
}

static llvm::json::Value getShapeJson(Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType || !shapedType.hasRank())
    return llvm::json::Value(nullptr);

  llvm::json::Array dims;
  for (int64_t dim : shapedType.getShape()) {
    if (ShapedType::isDynamic(dim))
      dims.push_back("?");
    else
      dims.push_back(dim);
  }
  return llvm::json::Value(std::move(dims));
}

static void appendUniqueString(
    SmallVectorImpl<std::string> &values, llvm::StringRef candidate) {
  if (candidate.empty())
    return;
  for (const std::string &value : values) {
    if (value == candidate)
      return;
  }
  values.push_back(candidate.str());
}

static bool isCpuHelperOp(Operation *op) {
  return isa<ONNXConstantOp, ONNXQuantizeLinearOp, ONNXDequantizeLinearOp>(op);
}

static bool isConstantLikeOp(Operation *op) {
  return isa<ONNXConstantOp, arith::ConstantOp>(op);
}

static bool isPotentialNpuOp(Operation *op) {
  return isa<ONNXConvOp, ONNXAddOp, ONNXQLinearMatMulOp, ONNXGemmOp,
      ONNXLayerNormalizationOp, ONNXSoftmaxOp, ONNXGeluOp, ONNXTransposeOp,
      ONNXMaxPoolSingleOutOp, ONNXResizeOp>(op);
}

static bool isCpuProfileCandidate(Operation *op) {
  if (op->getName().getDialectNamespace() != "onnx")
    return false;
  if (isCpuHelperOp(op))
    return false;
  return op->getNumResults() > 0;
}

static bool isCpuBeginBoundaryHelper(Operation *op) {
  auto dqOp = dyn_cast<ONNXDequantizeLinearOp>(op);
  if (!dqOp || op->getNumOperands() == 0)
    return false;

  Operation *dataDefOp = op->getOperand(0).getDefiningOp();
  return !dataDefOp || !isConstantLikeOp(dataDefOp);
}

static bool isCpuEndBoundaryHelper(Operation *op) {
  return isa<ONNXQuantizeLinearOp>(op);
}

static bool isNpuAuxiliaryOp(Operation *op) {
  return isa<bufferization::AllocTensorOp, linalg::CopyOp, tensor::DimOp,
      tensor::PadOp, ONNXConstantOp, arith::ConstantOp>(op);
}

static std::string getFallbackReason(Operation *op) {
  if (op->hasAttr("npu_qdq_warning_emitted"))
    return "missing_strict_qdq_context";
  if (isPotentialNpuOp(op))
    return "not_lowered_to_npu";
  return "unsupported_op";
}

static Operation *getRepresentativeTaggedOp(const LayerSlice &slice) {
  if (!slice.taggedOps.empty())
    return slice.taggedOps.front();
  return slice.beginOp;
}

static Operation *findCpuBoundaryBeginOp(Operation *op) {
  Operation *boundary = op;
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp || defOp->getBlock() != op->getBlock())
      continue;
    if (!isCpuBeginBoundaryHelper(defOp))
      continue;
    if (defOp->isBeforeInBlock(boundary))
      boundary = defOp;
  }
  return boundary;
}

static Operation *findCpuBoundaryEndOp(Operation *op) {
  Operation *boundary = op;
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (user->getBlock() != op->getBlock())
        continue;
      if (!isCpuEndBoundaryHelper(user))
        continue;
      if (boundary->isBeforeInBlock(user))
        boundary = user;
    }
  }
  return boundary;
}

static void normalizeBlockLayerBoundaries(MutableArrayRef<LayerSlice> blockLayers) {
  if (blockLayers.empty())
    return;

  for (size_t i = 1; i < blockLayers.size(); ++i) {
    LayerSlice &prev = blockLayers[i - 1];
    LayerSlice &curr = blockLayers[i];
    if (!prev.endOp || !curr.beginOp)
      continue;
    if (prev.endOp->getBlock() != curr.beginOp->getBlock())
      continue;

    bool overlapsPrev = curr.beginOp == prev.endOp ||
                        curr.beginOp->isBeforeInBlock(prev.endOp);
    if (!overlapsPrev)
      continue;

    if (Operation *nextOp = prev.endOp->getNextNode())
      curr.beginOp = nextOp;
    else if (!curr.ops.empty())
      curr.beginOp = curr.ops.front();
  }
}

static Operation *findTaggedOpByKind(
    const LayerSlice &slice, llvm::StringRef kind) {
  for (Operation *op : slice.taggedOps) {
    if (auto attr = op->getAttrOfType<StringAttr>(kLayerKindAttr)) {
      if (attr.getValue() == kind)
        return op;
    }
  }
  return nullptr;
}

static void collectBoundaryValues(const LayerSlice &slice,
    SmallVectorImpl<Value> &externalInputs,
    SmallVectorImpl<Value> &externalOutputs) {
  llvm::SmallPtrSet<Operation *, 32> opSet;
  for (Operation *op : slice.ops)
    opSet.insert(op);

  SmallVector<Value> seenInputs;
  SmallVector<Value> seenOutputs;

  auto wasSeen = [](Value value, ArrayRef<Value> values) {
    return llvm::is_contained(values, value);
  };

  for (Operation *op : slice.ops) {
    for (Value operand : op->getOperands()) {
      if (!isRankedShapedType(operand.getType()))
        continue;
      Operation *defOp = operand.getDefiningOp();
      if (defOp && opSet.contains(defOp))
        continue;
      if (!wasSeen(operand, seenInputs)) {
        externalInputs.push_back(operand);
        seenInputs.push_back(operand);
      }
    }
  }

  for (Operation *op : slice.ops) {
    for (Value result : op->getResults()) {
      if (!isRankedShapedType(result.getType()))
        continue;
      bool escapes = false;
      for (Operation *user : result.getUsers()) {
        if (!opSet.contains(user)) {
          escapes = true;
          break;
        }
      }
      if (escapes && !wasSeen(result, seenOutputs)) {
        externalOutputs.push_back(result);
        seenOutputs.push_back(result);
      }
    }
  }

  if (!externalOutputs.empty())
    return;

  for (Value result : slice.endOp->getResults()) {
    if (!isRankedShapedType(result.getType()))
      continue;
    if (!wasSeen(result, seenOutputs)) {
      externalOutputs.push_back(result);
      seenOutputs.push_back(result);
    }
  }
}

static llvm::json::Array buildShapesJson(ArrayRef<Value> values) {
  llvm::json::Array shapes;
  for (Value value : values)
    shapes.push_back(getShapeJson(value.getType()));
  return shapes;
}

static llvm::json::Array buildDtypesJson(ArrayRef<Value> values) {
  llvm::json::Array dtypes;
  for (Value value : values) {
    auto shapedType = dyn_cast<ShapedType>(value.getType());
    dtypes.push_back(
        shapedType ? stringifyType(shapedType.getElementType()) : "unknown");
  }
  return dtypes;
}

static std::optional<int64_t> sumTypeBytes(ArrayRef<Value> values) {
  int64_t total = 0;
  for (Value value : values) {
    auto bytes = getTypeByteSize(value.getType());
    if (!bytes)
      return std::nullopt;
    total += *bytes;
  }
  return total;
}

static std::optional<int64_t> getLayerMacs(
    const LayerSlice &slice, ArrayRef<Value> externalInputs,
    ArrayRef<Value> externalOutputs) {
  llvm::StringRef origin = slice.originOpType;

  if ((origin == "Gemm" || origin == "QLinearMatMul" || origin == "MatMul") &&
      !slice.taggedOps.empty()) {
    Operation *computeOp = findTaggedOpByKind(slice, "compute");
    if (!computeOp)
      computeOp = slice.taggedOps.front();
    if (computeOp->getNumOperands() < 2)
      return std::nullopt;
    auto lhsType = dyn_cast<ShapedType>(computeOp->getOperand(0).getType());
    auto outType = externalOutputs.empty()
                       ? dyn_cast<ShapedType>(computeOp->getResult(0).getType())
                       : dyn_cast<ShapedType>(externalOutputs.front().getType());
    if (!lhsType || !lhsType.hasStaticShape() || lhsType.getRank() < 2 ||
        !outType || !outType.hasStaticShape())
      return std::nullopt;
    int64_t k = lhsType.getShape().back();
    auto outElements = getNumElements(outType);
    return multiplyChecked(outElements, k);
  }

  if (origin == "Conv" && !externalInputs.empty() && !externalOutputs.empty()) {
    Operation *computeOp = findTaggedOpByKind(slice, "compute");
    if (!computeOp || computeOp->getNumOperands() < 2)
      return std::nullopt;
    auto inputType = dyn_cast<ShapedType>(externalInputs.front().getType());
    auto outputType = dyn_cast<ShapedType>(externalOutputs.front().getType());
    auto weightType = dyn_cast<ShapedType>(computeOp->getOperand(1).getType());
    if (!inputType || !outputType || !weightType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape() || !weightType.hasStaticShape() ||
        inputType.getRank() < 4 || outputType.getRank() < 4 ||
        weightType.getRank() < 4) {
      return std::nullopt;
    }
    int64_t n = inputType.getShape()[0];
    int64_t ic = inputType.getShape()[1];
    int64_t oc = outputType.getShape()[1];
    int64_t oh = outputType.getShape()[2];
    int64_t ow = outputType.getShape()[3];
    int64_t kh = weightType.getShape()[2];
    int64_t kw = weightType.getShape()[3];

    auto macs = std::optional<int64_t>(1);
    macs = multiplyChecked(macs, n);
    macs = multiplyChecked(macs, oc);
    macs = multiplyChecked(macs, oh);
    macs = multiplyChecked(macs, ow);
    macs = multiplyChecked(macs, ic);
    macs = multiplyChecked(macs, kh);
    macs = multiplyChecked(macs, kw);
    return macs;
  }

  if (origin == "Add" && !externalOutputs.empty())
    return getNumElements(externalOutputs.front().getType());

  return std::nullopt;
}

static llvm::json::Object buildLayerManifestObject(const LayerSlice &slice) {
  SmallVector<Value> externalInputs;
  SmallVector<Value> externalOutputs;
  collectBoundaryValues(slice, externalInputs, externalOutputs);

  std::optional<int64_t> inputBytes = std::nullopt;
  std::optional<int64_t> weightBytes = std::nullopt;
  if (slice.originOpType == "Conv" || slice.originOpType == "Gemm" ||
      slice.originOpType == "QLinearMatMul" || slice.originOpType == "MatMul") {
    if (!externalInputs.empty()) {
      auto dataBytes = getTypeByteSize(externalInputs.front().getType());
      if (dataBytes)
        inputBytes = dataBytes;
      if (externalInputs.size() > 1) {
        SmallVector<Value> weightInputs(
            externalInputs.begin() + 1, externalInputs.end());
        weightBytes = sumTypeBytes(weightInputs);
      }
    }
  } else {
    inputBytes = sumTypeBytes(externalInputs);
  }
  std::optional<int64_t> outputBytes = sumTypeBytes(externalOutputs);
  std::optional<int64_t> macs = getLayerMacs(slice, externalInputs, externalOutputs);

  llvm::json::Array fusedOps;
  for (const std::string &fusedOp : slice.fusedOps)
    fusedOps.push_back(fusedOp);

  llvm::json::Object layer;
  layer["layer_id"] = slice.layerId;
  layer["onnx_node_name"] = slice.layerName;
  layer["origin_op_type"] = slice.originOpType;
  layer["device"] = slice.device;
  layer["fused_ops"] = std::move(fusedOps);
  layer["input_shapes"] = buildShapesJson(externalInputs);
  layer["input_dtypes"] = buildDtypesJson(externalInputs);
  layer["output_shapes"] = buildShapesJson(externalOutputs);
  layer["output_dtypes"] = buildDtypesJson(externalOutputs);
  layer["input_bytes"] =
      inputBytes ? llvm::json::Value(*inputBytes) : llvm::json::Value(nullptr);
  layer["output_bytes"] =
      outputBytes ? llvm::json::Value(*outputBytes) : llvm::json::Value(nullptr);
  layer["weight_bytes"] =
      weightBytes ? llvm::json::Value(*weightBytes) : llvm::json::Value(nullptr);
  layer["macs"] =
      macs ? llvm::json::Value(*macs) : llvm::json::Value(nullptr);
  layer["fallback_reason"] = slice.fallbackReason.empty()
                                 ? llvm::json::Value(nullptr)
                                 : llvm::json::Value(slice.fallbackReason);
  return layer;
}

static func::FuncOp ensureProfileDecl(
    ModuleOp module, StringRef name, MLIRContext *context) {
  if (func::FuncOp func = module.lookupSymbol<func::FuncOp>(name))
    return func;

  OpBuilder builder(context);
  builder.setInsertionPointToEnd(module.getBody());
  auto funcType =
      builder.getFunctionType(TypeRange{builder.getI64Type()}, TypeRange{});
  func::FuncOp func = builder.create<func::FuncOp>(module.getLoc(), name, funcType);
  func.setVisibility(SymbolTable::Visibility::Private);
  return func;
}

static void appendTaggedOpMetadata(
    Operation *op, LayerSlice &slice, bool updateLayerName) {
  if (auto fusedAttr = op->getAttrOfType<ArrayAttr>(kFusedOpsAttr)) {
    for (Attribute attr : fusedAttr.getValue()) {
      if (auto strAttr = dyn_cast<StringAttr>(attr))
        appendUniqueString(slice.fusedOps, strAttr.getValue());
    }
  }

  if (auto originAttr = op->getAttrOfType<StringAttr>(kOriginOpTypeAttr))
    slice.originOpType = originAttr.getValue().str();

  if (updateLayerName) {
    if (auto nodeAttr = op->getAttrOfType<StringAttr>(kNodeNameAttr))
      slice.layerName = nodeAttr.getValue().str();
  }
}

struct NpuProfileAnnotatePass
    : public PassWrapper<NpuProfileAnnotatePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NpuProfileAnnotatePass)

  NpuProfileAnnotatePass() = default;
  NpuProfileAnnotatePass(const NpuProfileAnnotatePass &pass)
      : PassWrapper<NpuProfileAnnotatePass, OperationPass<ModuleOp>>() {}

  Option<std::string> profileManifest{*this, "npu-profile-manifest",
      llvm::cl::desc("Path to emitted profiling manifest JSON"),
      llvm::cl::init("profile_manifest.json")};

  StringRef getArgument() const override { return "npu-profile-annotate"; }
  StringRef getDescription() const override {
    return "Annotate ONNX logical layers with profiling begin/end calls.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    const std::string manifestPath = profileManifest;

    func::FuncOp beginDecl = ensureProfileDecl(module, "npu_profile_begin", context);
    func::FuncOp endDecl = ensureProfileDecl(module, "npu_profile_end", context);
    (void)beginDecl;
    (void)endDecl;

    SmallVector<LayerSlice, 8> layers;
    int64_t nextLayerId = 0;

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isDeclaration())
        continue;
      if (func.getName() == "npu_profile_begin" || func.getName() == "npu_profile_end" ||
          func.getName() == "npu_profile_dump")
        continue;
      if (func->hasAttr("npu.target"))
        continue;

      for (Block &block : func.getBody().getBlocks()) {
        size_t blockLayerBegin = layers.size();
        for (Operation *op = block.empty() ? nullptr : &block.front(); op != nullptr;) {
          Operation *next = op->getNextNode();
          auto layerNameAttr = op->getAttrOfType<StringAttr>(kLayerNameAttr);

          if (layerNameAttr) {
            LayerSlice slice;
            slice.layerId = nextLayerId++;
            slice.beginOp = op;
            slice.endOp = op;
            slice.device = "npu";
            if (auto nodeAttr = op->getAttrOfType<StringAttr>(kNodeNameAttr))
              slice.layerName = nodeAttr.getValue().str();
            else
              slice.layerName = layerNameAttr.getValue().str();

            llvm::StringRef activeLayerName = layerNameAttr.getValue();
            for (Operation *cursor = op; cursor != nullptr;
                 cursor = cursor->getNextNode()) {
              auto cursorLayerName = cursor->getAttrOfType<StringAttr>(kLayerNameAttr);
              bool sameLayer =
                  cursorLayerName && cursorLayerName.getValue() == activeLayerName;
              if (cursor != op && !sameLayer && !isNpuAuxiliaryOp(cursor))
                break;

              slice.ops.push_back(cursor);
              slice.endOp = cursor;
              if (sameLayer) {
                slice.taggedOps.push_back(cursor);
                appendTaggedOpMetadata(
                    cursor, slice, cursor == op && slice.layerName.empty());
              }
              next = cursor->getNextNode();
            }

            if (slice.originOpType.empty()) {
              Operation *repOp = getRepresentativeTaggedOp(slice);
              slice.originOpType = repOp
                                       ? npux::getNpuProfileOpTypeName(repOp)
                                       : std::string("unknown");
            }
            if (slice.fusedOps.empty())
              appendUniqueString(slice.fusedOps, slice.originOpType);
            layers.push_back(std::move(slice));
            op = next;
            continue;
          }

          if (isCpuProfileCandidate(op)) {
            LayerSlice slice;
            slice.layerId = nextLayerId++;
            slice.beginOp = findCpuBoundaryBeginOp(op);
            slice.endOp = findCpuBoundaryEndOp(op);
            slice.device = "cpu";
            slice.ops.push_back(op);
            slice.layerName = npux::getNpuProfileLayerName(op);
            slice.originOpType = npux::getNpuProfileOpTypeName(op);
            appendUniqueString(slice.fusedOps, slice.originOpType);
            slice.fallbackReason = getFallbackReason(op);
            layers.push_back(std::move(slice));
          }

          op = next;
        }

        if (layers.size() != blockLayerBegin) {
          MutableArrayRef<LayerSlice> blockLayers(
              layers.data() + blockLayerBegin, layers.size() - blockLayerBegin);
          normalizeBlockLayerBoundaries(blockLayers);
        }
      }
    }

    for (LayerSlice &slice : layers) {
      OpBuilder beginBuilder(slice.beginOp);
      Value beginId = beginBuilder.create<arith::ConstantIntOp>(
          slice.beginOp->getLoc(), slice.layerId, 64);
      beginBuilder.create<func::CallOp>(
          slice.beginOp->getLoc(), "npu_profile_begin", TypeRange{},
          ValueRange{beginId});

      OpBuilder endBuilder(slice.endOp->getContext());
      endBuilder.setInsertionPointAfter(slice.endOp);
      Value endId = endBuilder.create<arith::ConstantIntOp>(
          slice.endOp->getLoc(), slice.layerId, 64);
      endBuilder.create<func::CallOp>(
          slice.endOp->getLoc(), "npu_profile_end", TypeRange{},
          ValueRange{endId});
    }

    llvm::json::Array layerArray;
    for (const LayerSlice &slice : layers)
      layerArray.push_back(buildLayerManifestObject(slice));

    llvm::json::Object root;
    root["layers"] = std::move(layerArray);

    std::error_code ec;
    llvm::raw_fd_ostream manifestOS(manifestPath, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      module.emitError() << "failed to write profiling manifest to '"
                         << manifestPath << "': " << ec.message();
      signalPassFailure();
      return;
    }
    manifestOS << llvm::formatv("{0:2}", llvm::json::Value(std::move(root)));
    manifestOS << "\n";
  }
};

} // namespace

std::unique_ptr<Pass> npux::createNpuProfileAnnotatePass() {
  return std::make_unique<NpuProfileAnnotatePass>();
}
