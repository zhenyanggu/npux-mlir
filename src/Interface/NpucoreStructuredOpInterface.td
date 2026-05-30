// SPDX-License-Identifier: Apache-2.0

//===----------- NpucoreStructuredOpInterface.td -------------------------===//
//===--------- Npucore structured op interface definition ----------------===//
//
// Copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file defines the shared semantic interface for npucore structured ops.
//
//===----------------------------------------------------------------------===//

#ifdef NPUCORE_STRUCTURED_OP_INTERFACE
#else
#define NPUCORE_STRUCTURED_OP_INTERFACE

include "mlir/IR/OpBase.td"

def NpucoreStructuredOpInterface
    : OpInterface<"NpucoreStructuredOpInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    Operation interface that exposes the common semantic properties shared by
    npucore structured operations. This interface is intended for NPU-specific
    passes and helpers, while standard MLIR infrastructure continues to use
    interfaces such as TilingInterface, DestinationStyleOpInterface, and
    BufferizableOpInterface.
  }];

  let methods = [
    InterfaceMethod<
      "Return the npucore semantic category of this op.",
      "::llvm::StringRef",
      "getNpucoreOpKind",
      (ins)
    >,
    InterfaceMethod<
      "Return the logical loop rank used by NPU-specific analyses.",
      "unsigned",
      "getLogicalLoopRank",
      (ins)
    >,
    InterfaceMethod<
      "Return whether the given logical loop dimension is a reduction loop.",
      "bool",
      "isReductionLoop",
      (ins "unsigned":$dim)
    >,
    InterfaceMethod<
      "Return whether this op expects a following acc-to-spm materialization.",
      "bool",
      "requiresAccToSpmMove",
      (ins)
    >
  ];
}

#endif // NPUCORE_STRUCTURED_OP_INTERFACE
