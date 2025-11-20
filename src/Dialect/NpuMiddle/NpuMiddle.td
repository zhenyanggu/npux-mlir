include "mlir/IR/OpBase.td"

def NpuMiddle_Dialect : Dialect {
  let name = "npu_middle";
  let cppNamespace = "npu_middle";
  let usePropertiesForAttributes = 0;
  let dependentDialects = [

  ];
}

class AnyStrAttrOf<list<string> cases>
    : StringBasedAttr<CPred<!foldl("mlir::cast<mlir::StringAttr>($_self)."
                                   "getValue() == \""#!head(cases)#"\"",
                          !foreach(case, !tail(cases),
                              "mlir::cast<mlir::StringAttr>($_self).getValue() "
                              "== \""#case#"\""),
                          prev, cur, prev#" || "#cur)>,

          "string attribute whose value is "#!foldl(/*init*/ !head(cases),
              /*list*/ !tail(cases), prev, cur, prev#", or "#cur)>;

def AddressSpaceAttr : AnyStrAttrOf<["DDR", "NPU_SPM", "NPU_ACC"]>;

def NpuMiddleGeluOp : Op<NpuMiddle_Dialect, "gelu", []> {
  let summary = "gelu operation";
  let description = [{
        This operation applies the GELU activation function.
        output = 0.5 * input * ( 1 + erf(input / sqrt(2)) )
    }];
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyMemRef:$output);

}

def NpuMiddleMvinOp : Op<NpuMiddle_Dialect, "mvin", []> {
  let summary = "memory to npu transfer operation";
  let description = [{
        This operation transfers data from main memory to NPU memory.
    }];
  let arguments = (ins AnyMemRef:$input, AddressSpaceAttr:$memspace);
  let results = (outs AnyMemRef:$output);

}

def NpuMiddleMvoutOp : Op<NpuMiddle_Dialect, "mvout", []> {
  let summary = "npu to memory transfer operation";
  let description = [{
        This operation transfers data from NPU memory to main memory.
    }];
  let arguments = (ins AnyMemRef:$input, AddressSpaceAttr:$memspace);
  let results = (outs AnyMemRef:$output);

}