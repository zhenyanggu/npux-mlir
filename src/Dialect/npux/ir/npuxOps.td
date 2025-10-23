// Defines npux operations.

#ifndef NPU_MLIR_NPUX_OPS
#define NPU_MLIR_NPUX_OPS
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
// include "dialect/npu_common/npu_attrs.td"

include "npuxDialect.td"


// 定义一个基础操作类，其他操作可以继承它
class npux_BaseOp<string name, list<Trait> traits = []> :  
    Op<npuxDialect, name, traits> ;

// #TODO: 后面可能再封装一遍op，把自定义的attribute分类封装进去，类似
// class Top_Op<string mnemonic, list<Trait> traits = []> :
//     Top_BaseOp<mnemonic, !listconcat(traits,
//        [DeclareOpInterfaceMethods<InferenceInterface>,
//         DeclareOpInterfaceMethods<FlopsInterface>,
//         DeclareOpInterfaceMethods<ShapeInterface>])>;

// #TODO: 后面可能像tpu-mlir一样实现一些比较抽象的op

// #TODO: 这里先实现一些MVP需要的OP，暂时之想到了GELU、DMA，还需要什么吗？

// GELU Op
def npux_geluOp : npux_BaseOp<"gelu"> {
    let summary = "GELU activation operation";

    let description = [{
        This operation applies the GELU activation function.
        output = 0.5 * input * ( 1 + erf(input / sqrt(2)) )
    }];

    let arguments = (ins
        AnyTensor:$input
    );


    let results = (outs
        AnyTensor:$output
    );
}

// mvin op
def npux_mvinOp : npux_BaseOp<"mvin"> {
    let summary = "Memory to NPU transfer operation";

    let description = [{
        This operation transfers data from main memory to NPU memory.
    }];

    let arguments = (ins
        AnyTensor:$input,
        AddressSpaceAttr:$memspace  // 这里用上面定义的枚举类型
    );

    let results = (outs
        AnyTensor:$output
    );
}

def npux_mvoutOp : npux_BaseOp<"mvout"> {
    let summary = "NPU to Memory transfer operation";

    let description = [{
        This operation transfers data from NPU memory to main memory.
    }];

    let arguments = (ins
        AnyTensor:$input,
        AddressSpaceAttr:$memspace  // 这里用上面定义的枚举类型
    );

    let results = (outs
        AnyTensor:$output
    );


}

#endif

