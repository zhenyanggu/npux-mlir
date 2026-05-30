//=============================================
// src/Dialect/Npucore/Npucore.td
// this file defines the npucore dialect.
//=============================================

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "src/Interface/NpucoreStructuredOpInterface.td"

def Npucore_Dialect : Dialect {
  let name = "npucore";
  let cppNamespace = "::npucore";
  let summary = "High-level NPU core semantics dialect";
  let description = [{
    The npucore dialect is the semantic layer between ONNX and npux. The
    current experiment focuses on a native npucore Gelu pipeline.
  }];
}

class Npucore_Op<string mnemonic, list<Trait> traits = []>
    : Op<Npucore_Dialect, mnemonic, traits>;

def Npucore_GeluOp : Npucore_Op<"gelu",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<NpucoreStructuredOpInterface>]> {
  let summary = "NPU Gelu op for tensor or memref semantics";

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyShaped>:$outputs,
    F32Attr:$in_scale,
    I32Attr:$in_zp,
    F32Attr:$out_scale,
    I16Attr:$out_zp
  );

  let results = (outs Variadic<AnyTensor>:$resultTensors);

  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    `quant` `(` $in_scale `,` $in_zp `,` $out_scale `,` $out_zp `)`
    attr-dict (`->` type($resultTensors)^)?
  }];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Return the destination operands used by DPS and bufferization helpers.
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    /// Return true when this op carries tensor semantics.
    bool hasTensorSemantics() { return !getResultTensors().empty(); }

    /// Return true when this op carries memref semantics.
    bool hasBufferSemantics() { return getResultTensors().empty(); }

    /// Verify the restricted Gelu experiment contracts.
    static ::llvm::LogicalResult verifyInvariantsImpl(::mlir::Operation *op);

  }];
}

def Npucore_SoftmaxOp : Npucore_Op<"softmax",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<NpucoreStructuredOpInterface>]> {
  let summary = "NPU Softmax op for tensor or memref semantics";

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyShaped>:$outputs,
    F32Attr:$in_scale,
    I32Attr:$in_zp,
    F32Attr:$out_scale,
    I16Attr:$out_zp,
    I64Attr:$axis
  );

  let results = (outs Variadic<AnyTensor>:$resultTensors);

  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    `quant` `(` $in_scale `,` $in_zp `,` $out_scale `,` $out_zp `)`
    `axis` `(` $axis `)`
    attr-dict (`->` type($resultTensors)^)?
  }];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Return the destination operands used by DPS and bufferization helpers.
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    /// Return true when this op carries tensor semantics.
    bool hasTensorSemantics() { return !getResultTensors().empty(); }

    /// Return true when this op carries memref semantics.
    bool hasBufferSemantics() { return getResultTensors().empty(); }

    /// Verify the restricted Softmax experiment contracts.
    static ::llvm::LogicalResult verifyInvariantsImpl(::mlir::Operation *op);
  }];
}

def Npucore_LayerNormOp : Npucore_Op<"layernorm",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<NpucoreStructuredOpInterface>]> {
  let summary = "NPU LayerNorm op for tensor or memref semantics";

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyShaped>:$outputs,
    F32Attr:$in_scale,
    I32Attr:$in_zp,
    F32Attr:$out_scale,
    I16Attr:$out_zp,
    I64Attr:$axis,
    F32Attr:$epsilon
  );

  let results = (outs Variadic<AnyTensor>:$resultTensors);

  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    `quant` `(` $in_scale `,` $in_zp `,` $out_scale `,` $out_zp `)`
    `axis` `(` $axis `)`
    `epsilon` `(` $epsilon `)`
    attr-dict (`->` type($resultTensors)^)?
  }];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Return the destination operands used by DPS and bufferization helpers.
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    /// Return true when this op carries tensor semantics.
    bool hasTensorSemantics() { return !getResultTensors().empty(); }

    /// Return true when this op carries memref semantics.
    bool hasBufferSemantics() { return getResultTensors().empty(); }

    /// Verify the restricted LayerNorm experiment contracts.
    static ::llvm::LogicalResult verifyInvariantsImpl(::mlir::Operation *op);
  }];
}

def Npucore_MatAddOp : Npucore_Op<"matadd",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<NpucoreStructuredOpInterface>]> {
  let summary = "NPU MatAdd op for tensor or memref semantics";

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyShaped>:$outputs,
    F32Attr:$in1_scale,
    I32Attr:$in1_zp,
    F32Attr:$in2_scale,
    I32Attr:$in2_zp,
    F32Attr:$out_scale,
    I16Attr:$out_zp
  );

  let results = (outs Variadic<AnyTensor>:$resultTensors);

  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    `quant` `(` $in1_scale `,` $in1_zp `,` $in2_scale `,` $in2_zp `,`
                $out_scale `,` $out_zp `)`
    attr-dict (`->` type($resultTensors)^)?
  }];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Return the destination operands used by DPS and bufferization helpers.
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    /// Return true when this op carries tensor semantics.
    bool hasTensorSemantics() { return !getResultTensors().empty(); }

    /// Return true when this op carries memref semantics.
    bool hasBufferSemantics() { return getResultTensors().empty(); }

    /// Verify the restricted MatAdd experiment contracts.
    static ::llvm::LogicalResult verifyInvariantsImpl(::mlir::Operation *op);
  }];
}

def Npucore_MatMulOp : Npucore_Op<"matmul",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<NpucoreStructuredOpInterface>]> {
  let summary = "NPU MatMul op for tensor or memref semantics";

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyShaped>:$outputs,
    F32Attr:$lhs_scale,
    I32Attr:$lhs_zp,
    F32Attr:$rhs_scale,
    I32Attr:$rhs_zp,
    F32Attr:$out_scale,
    I32Attr:$out_zp,
    I32Attr:$with_bias,
    I32Attr:$do_relu,
    I32Attr:$relu_type
  );

  let results = (outs Variadic<AnyTensor>:$resultTensors);

  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    `quant` `(` $lhs_scale `,` $lhs_zp `,` $rhs_scale `,` $rhs_zp `,`
                $out_scale `,` $out_zp `)`
    `bias` `(` $with_bias `)`
    `relu` `(` $do_relu `,` $relu_type `)`
    attr-dict (`->` type($resultTensors)^)?
  }];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Return the destination operands used by DPS and bufferization helpers.
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    /// Return true when this op carries tensor semantics.
    bool hasTensorSemantics() { return !getResultTensors().empty(); }

    /// Return true when this op carries memref semantics.
    bool hasBufferSemantics() { return getResultTensors().empty(); }

    /// Verify the restricted MatMul experiment contracts.
    static ::llvm::LogicalResult verifyInvariantsImpl(::mlir::Operation *op);
  }];
}

def Npucore_MaxPoolOp : Npucore_Op<"maxpool",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<NpucoreStructuredOpInterface>]> {
  let summary = "NPU MaxPool op for tensor or memref semantics";

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyShaped>:$outputs,
    F32Attr:$in_scale,
    I32Attr:$in_zp,
    F32Attr:$out_scale,
    I16Attr:$out_zp,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$dilations,
    I64ArrayAttr:$pads
  );

  let results = (outs Variadic<AnyTensor>:$resultTensors);

  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    `quant` `(` $in_scale `,` $in_zp `,` $out_scale `,` $out_zp `)`
    `kernel` `(` $kernel_shape `)`
    `strides` `(` $strides `)`
    `dilations` `(` $dilations `)`
    `pads` `(` $pads `)`
    attr-dict (`->` type($resultTensors)^)?
  }];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Return the destination operands used by DPS and bufferization helpers.
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    /// Return true when this op carries tensor semantics.
    bool hasTensorSemantics() { return !getResultTensors().empty(); }

    /// Return true when this op carries memref semantics.
    bool hasBufferSemantics() { return getResultTensors().empty(); }

    /// Verify the restricted MaxPool experiment contracts.
    static ::llvm::LogicalResult verifyInvariantsImpl(::mlir::Operation *op);
  }];
}

def Npucore_TransposeOp : Npucore_Op<"transpose",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<NpucoreStructuredOpInterface>]> {
  let summary = "NPU transpose op for tensor or memref semantics";

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyShaped>:$outputs
  );

  let results = (outs Variadic<AnyTensor>:$resultTensors);

  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    attr-dict (`->` type($resultTensors)^)?
  }];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Return the destination operands used by DPS and bufferization helpers.
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    /// Return true when this op carries tensor semantics.
    bool hasTensorSemantics() { return !getResultTensors().empty(); }

    /// Return true when this op carries memref semantics.
    bool hasBufferSemantics() { return getResultTensors().empty(); }

    /// Verify the restricted transpose experiment contracts.
    static ::llvm::LogicalResult verifyInvariantsImpl(::mlir::Operation *op);
  }];
}

def Npucore_LayoutNchwToNchwc32Op : Npucore_Op<"layout_nchw_to_nchwc32",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<NpucoreStructuredOpInterface>]> {
  let summary = "Pack NCHW tensors or memrefs into NCHWc32 layout";

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyShaped>:$outputs,
    I64Attr:$tile_factor
  );

  let results = (outs Variadic<AnyTensor>:$resultTensors);

  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    `tile` `(` $tile_factor `)`
    attr-dict (`->` type($resultTensors)^)?
  }];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Return the destination operands used by DPS and bufferization helpers.
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    /// Return true when this op carries tensor semantics.
    bool hasTensorSemantics() { return !getResultTensors().empty(); }

    /// Return true when this op carries memref semantics.
    bool hasBufferSemantics() { return getResultTensors().empty(); }

    /// Verify the restricted layout pack experiment contracts.
    static ::llvm::LogicalResult verifyInvariantsImpl(::mlir::Operation *op);
  }];
}

def Npucore_LayoutNchwc32ToNchwOp : Npucore_Op<"layout_nchwc32_to_nchw",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<NpucoreStructuredOpInterface>]> {
  let summary = "Unpack NCHWc32 tensors or memrefs into NCHW layout";

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyShaped>:$outputs,
    I64Attr:$tile_factor
  );

  let results = (outs Variadic<AnyTensor>:$resultTensors);

  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    `tile` `(` $tile_factor `)`
    attr-dict (`->` type($resultTensors)^)?
  }];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Return the destination operands used by DPS and bufferization helpers.
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    /// Return true when this op carries tensor semantics.
    bool hasTensorSemantics() { return !getResultTensors().empty(); }

    /// Return true when this op carries memref semantics.
    bool hasBufferSemantics() { return getResultTensors().empty(); }

    /// Verify the restricted layout unpack experiment contracts.
    static ::llvm::LogicalResult verifyInvariantsImpl(::mlir::Operation *op);
  }];
}

def Npucore_MvAccToSpmOp : Npucore_Op<"mv_acc_to_spm",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<NpucoreStructuredOpInterface>]> {
  let summary = "Move accumulator tensors or memrefs into SPM layout";

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyShaped>:$outputs
  );

  let results = (outs Variadic<AnyTensor>:$resultTensors);

  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    attr-dict (`->` type($resultTensors)^)?
  }];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Return the destination operands used by DPS and bufferization helpers.
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    /// Return true when this op carries tensor semantics.
    bool hasTensorSemantics() { return !getResultTensors().empty(); }

    /// Return true when this op carries memref semantics.
    bool hasBufferSemantics() { return getResultTensors().empty(); }

    /// Verify the restricted ACC to SPM move contracts.
    static ::llvm::LogicalResult verifyInvariantsImpl(::mlir::Operation *op);
  }];
}

def Npucore_DmaMvinOp : Npucore_Op<"dma_mvin",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<NpucoreStructuredOpInterface>]> {
  let summary = "Move host tensor or memref into SPM or ACC";

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyShaped>:$outputs,
    StrAttr:$dma_type,
    I32Attr:$dma_col_dim,
    BoolAttr:$is_quant,
    I32Attr:$quant_zero,
    I64Attr:$quant_scale,
    I64Attr:$quant_shift
  );

  let results = (outs Variadic<AnyTensor>:$resultTensors);

  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    `dma` `(` $dma_type `,` $dma_col_dim `)`
    `quant` `(` $is_quant `,` $quant_zero `,` $quant_scale `,` $quant_shift `)`
    attr-dict (`->` type($resultTensors)^)?
  }];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Return the destination operands used by DPS and bufferization helpers.
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    /// Return true when this op carries tensor semantics.
    bool hasTensorSemantics() { return !getResultTensors().empty(); }

    /// Return true when this op carries memref semantics.
    bool hasBufferSemantics() { return getResultTensors().empty(); }

    /// Verify the restricted DMA mvin contracts.
    static ::llvm::LogicalResult verifyInvariantsImpl(::mlir::Operation *op);
  }];
}

def Npucore_DmaMvoutOp : Npucore_Op<"dma_mvout",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<NpucoreStructuredOpInterface>]> {
  let summary = "Move SPM tensor or memref back to host";

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyShaped>:$outputs,
    StrAttr:$dma_type,
    I32Attr:$dma_col_dim,
    BoolAttr:$is_quant,
    I32Attr:$quant_zero,
    I64Attr:$quant_scale,
    I64Attr:$quant_shift
  );

  let results = (outs Variadic<AnyTensor>:$resultTensors);

  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    `dma` `(` $dma_type `,` $dma_col_dim `)`
    `quant` `(` $is_quant `,` $quant_zero `,` $quant_scale `,` $quant_shift `)`
    attr-dict (`->` type($resultTensors)^)?
  }];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Return the destination operands used by DPS and bufferization helpers.
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    /// Return true when this op carries tensor semantics.
    bool hasTensorSemantics() { return !getResultTensors().empty(); }

    /// Return true when this op carries memref semantics.
    bool hasBufferSemantics() { return getResultTensors().empty(); }

    /// Verify the restricted DMA mvout contracts.
    static ::llvm::LogicalResult verifyInvariantsImpl(::mlir::Operation *op);
  }];
}

def Npucore_ConvOp : Npucore_Op<"conv",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<NpucoreStructuredOpInterface>]> {
  let summary = "NPU Conv op for tensor or memref semantics";

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyShaped>:$outputs,
    F32Attr:$in_scale,
    I32Attr:$in_zp,
    F32Attr:$w_scale,
    I32Attr:$w_zp,
    F32Attr:$out_scale,
    I16Attr:$out_zp,
    I64ArrayAttr:$pads,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$dilations,
    I64Attr:$group,
    I32Attr:$do_relu,
    I32Attr:$relu_type
  );

  let results = (outs Variadic<AnyTensor>:$resultTensors);

  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    `quant` `(` $in_scale `,` $in_zp `,` $w_scale `,` $w_zp `,`
                $out_scale `,` $out_zp `)`
    `pads` `(` $pads `)`
    `strides` `(` $strides `)`
    `dilations` `(` $dilations `)`
    `group` `(` $group `)`
    `relu` `(` $do_relu `,` $relu_type `)`
    attr-dict (`->` type($resultTensors)^)?
  }];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Return the destination operands used by DPS and bufferization helpers.
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    /// Return true when this op carries tensor semantics.
    bool hasTensorSemantics() { return !getResultTensors().empty(); }

    /// Return true when this op carries memref semantics.
    bool hasBufferSemantics() { return getResultTensors().empty(); }

    /// Verify the restricted Conv experiment contracts.
    static ::llvm::LogicalResult verifyInvariantsImpl(::mlir::Operation *op);
  }];
}
