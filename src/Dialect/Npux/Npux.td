//=============================================
// src/Dialect/Npux/Npux.td
// this file defines npux Dialect
//=============================================

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/EnumAttr.td"

def Npux_Dialect : Dialect {
  let name = "npux";
  let cppNamespace = "::npux";
  let summary = "A dialect for Custom NPU-CPU Co-design System";
  let description = [{
    The NPU dialect maps directly to the C-API runtime of the NPU.
    It manages memory allocation in NPU-addressable DRAM, data movement
    between DRAM and SRAM, and compute execution.
  }];
}

class Npux_Op<string mnemonic, list<Trait> traits = []>
    : Op<Npux_Dialect, mnemonic, traits>;

def Npux_AllocOp : Npux_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate memory in NPU-managed DRAM region";

  let arguments = (ins Variadic<Index>:$dynamic_sizes);

  let results = (outs Res<AnyMemRef, "allocated buffer", [MemAlloc]>:$memref);

  let assemblyFormat = "(`(` $dynamic_sizes^ `)`)? attr-dict `:` type($memref)";
}

def Npux_FreeOp : Npux_Op<"free", [MemoryEffects<[MemFree]>]> {
  let summary = "Free memory in NPU-managed DRAM region";

  let arguments = (ins AnyMemRef:$memref);
  let results = (outs);

  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

def Npux_InitOp : Npux_Op<"init"> {
  let summary = "Initialize NPU Runtime";
  let results = (outs I32:$status);
}

def Npux_DestroyOp : Npux_Op<"destroy"> {
  let summary = "Destroy NPU Runtime context";
  let results = (outs);
}

def IsNpuSram : CPred<"llvm::cast<::mlir::MemRefType>($_self).getMemorySpaceAsInt() == 2">;
def NpuSramMemRef : Type<IsNpuSram, "memref in NPU SRAM", "::mlir::MemRefType">;

def Npux_DmaMvinOp : Npux_Op<"dma_mvin"> {
  let summary = "Host -> SRAM DMA";

  let arguments = (ins AnyMemRef:$host_ptr, NpuSramMemRef:$sram_memref,


      I16:$col_num, I16:$row_num, I16:$sram_stride, I16:$dram_stride,
      I8:$precision, I8:$input_type, I8:$dest,

      I1:$is_quant, I32:$quant_zero, I16:$quant_scale, I16:$quant_shift);

  let results = (outs);


  let assemblyFormat = [{
    $host_ptr `,` $sram_memref 
    `shape` `(` $col_num `x` $row_num `)` 
    `stride` `(` $sram_stride `,` $dram_stride `)`
    `cfg` `(` $precision `,` $input_type `,` $dest `)`
    `quant` `(` $is_quant `,` $quant_zero `,` $quant_scale `,` $quant_shift `)`
    attr-dict `:` type($host_ptr) `,` type($sram_memref)
  }];
}

def Npux_DmaMvoutOp : Npux_Op<"dma_mvout"> {
  let summary = "SRAM -> Host DMA";

  let arguments = (ins AnyMemRef:$host_ptr, NpuSramMemRef:$sram_memref, I16:$col_num,
      I16:$row_num, I16:$sram_stride, I16:$dram_stride, I8:$precision,
      I8:$output_type, I8:$source, I1:$is_quant, I32:$quant_zero,
      I16:$quant_scale, I16:$quant_shift);
  let results = (outs);
  let assemblyFormat = [{
    $host_ptr `,` $sram_memref 
    `shape` `(` $col_num `x` $row_num `)` 
    `stride` `(` $sram_stride `,` $dram_stride `)`
    `cfg` `(` $precision `,` $output_type `,` $source `)`
    `quant` `(` $is_quant `,` $quant_zero `,` $quant_scale `,` $quant_shift `)`
    attr-dict `:` type($host_ptr) `,` type($sram_memref)
  }];
}

def SFU_Op_SOFTMAX : I32EnumAttrCase<"softmax", 0>;
def SFU_Op_GELU : I32EnumAttrCase<"gelu", 1>;
def SFU_Op_LAYERNORM : I32EnumAttrCase<"layernorm", 2>;
def SFU_Op_TRANSPOSE : I32EnumAttrCase<"transpose", 8>;
def SFU_OpTypeAttr
    : I32EnumAttr<"SFUOpType", "SFU Operation Type",
          [SFU_Op_SOFTMAX, SFU_Op_GELU, SFU_Op_LAYERNORM, SFU_Op_TRANSPOSE]> {
  let cppNamespace = "::npux";
}

def Npux_SfuRunOp : Npux_Op<"sfu_run"> {
  let summary = "Run SFU Operation";

  let arguments = (ins

      SFU_OpTypeAttr:$op_type,


      I8:$int_type, I1:$is_quant,

      NpuSramMemRef:$input_sram_memref, I16:$input_col_num, I16:$input_row_num,

      NpuSramMemRef:$output_sram_memref, 

      I32:$input_zeropoint, I16:$output_zeropoint, I16:$input_scale,
      I16:$input_scale_shift, I16:$output_scale, I16:$output_scale_shift);

  let results = (outs);


  let assemblyFormat = [{
    $op_type 
    $int_type `,` $is_quant
    `in` `(` $input_sram_memref `)` 
    `shape` `(` $input_col_num `x` $input_row_num `)`
    `out` `(` $output_sram_memref `)`
    `quant` `(` 
        $input_zeropoint `,` $output_zeropoint `,` 
        $input_scale `,` $input_scale_shift `,` 
        $output_scale `,` $output_scale_shift 
    `)`
    attr-dict `:` type($input_sram_memref) `,` type($output_sram_memref)
  }];
}