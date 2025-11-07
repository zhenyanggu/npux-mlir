#ifndef NPUX_DIALECT_TD
#define NPUX_DIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def npuxDialect : Dialect {
    let name = "npux";
    let summary = "NPUX dialect";
    let cppNamespace = "::npux";
    let description = [{
        The NPUX dialect contains operations specific to the NPUX architecture.
    }];
}


// #TODO: 后面可能要定义自己 attribute traits 和 types

// npux attributes
// A string attribute whose value are one of the values in `cases`.
class AnyStrAttrOf<list<string> cases> : StringBasedAttr<
  CPred<!foldl(
    // 将 .cast<>() 修改为 mlir::cast<>()
    "mlir::cast<mlir::StringAttr>($_self).getValue() == \"" # !head(cases) # "\"",
    !foreach(case, !tail(cases),
             // 同样修改这里
             "mlir::cast<mlir::StringAttr>($_self).getValue() == \"" # case # "\""),
    prev, cur, prev # " || " # cur)>,

  // 下面这部分是生成描述字符串的，不需要修改
  "string attribute whose value is " #
    !foldl(/*init*/!head(cases), /*list*/!tail(cases),
           prev, cur, prev # ", or " # cur)>;


def AddressSpaceAttr : AnyStrAttrOf<["DDR", "NPU_SPM", "NPU_ACC"]>;


#endif // NPUX_DIALECT_TD