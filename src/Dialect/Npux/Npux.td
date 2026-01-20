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

def IsNpuSram : CPred<"llvm::cast<::mlir::MemRefType>($_self).getMemorySpaceAsInt() == 2">;
def NpuSramMemRef : Type<IsNpuSram, "memref in NPU SRAM", "::mlir::MemRefType">;

def Npux_SramAllocOp : Npux_Op<"sram_alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate memory in NPU Internal SRAM";
  let description = [{
    Allocates a buffer specifically in the NPU SRAM (Memory Space 2).
    Unlike DRAM allocation, this represents on-chip memory which is 
    limited in size and requires manual management via DMA.
  }];

  let arguments = (ins Variadic<Index>:$dynamic_sizes);

  let results = (outs Res<NpuSramMemRef, "allocated sram buffer", [MemAlloc]>:$memref);

  let assemblyFormat = "(`(` $dynamic_sizes^ `)`)? attr-dict `:` type($memref)";
}


def Npux_SramFreeOp : Npux_Op<"sram_free", [MemoryEffects<[MemFree]>]> {
  let summary = "Free memory in NPU Internal SRAM";
  let description = [{
    Deallocates a buffer explicitly in the NPU SRAM (memory space 2).
    This is used to mark the end of life for an SRAM buffer to reuse internal memory.
  }];
  let arguments = (ins NpuSramMemRef:$sram_memref);
  let results = (outs);

  let assemblyFormat = "$sram_memref attr-dict `:` type($sram_memref)";
}


def Npux_InitOp : Npux_Op<"init"> {
  let summary = "Initialize NPU Runtime";
  let results = (outs I32:$status);
}

def Npux_DestroyOp : Npux_Op<"destroy"> {
  let summary = "Destroy NPU Runtime context";
  let results = (outs);
}


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


def COMPUTE_Op_GEMM : I32EnumAttrCase<"gemm", 0>;
def COMPUTE_Op_CONV : I32EnumAttrCase<"conv", 1>;
def COMPUTE_Op_GEMV : I32EnumAttrCase<"gemv", 2>;

def COMPUTE_OpTypeAttr
    : I32EnumAttr<"ComputeOpType", "Compute Operation Type",
          [COMPUTE_Op_GEMM, COMPUTE_Op_CONV, COMPUTE_Op_GEMV]> {
  let cppNamespace = "::npux";
}

def ACT_Op_RELU        : I32EnumAttrCase<"relu", 0>;
def ACT_Op_RELU6       : I32EnumAttrCase<"relu6", 1>;
def ACT_Op_LEAKY_0_1   : I32EnumAttrCase<"leaky_0_1", 2>;
def ACT_Op_LEAKY_0_2   : I32EnumAttrCase<"leaky_0_2", 3>;
def ACT_Op_LEAKY_0_01  : I32EnumAttrCase<"leaky_0_01", 4>;

def ACT_TypeAttr
    : I32EnumAttr<"ActivationType", "Activation Function Type",
          [ACT_Op_RELU, ACT_Op_RELU6, ACT_Op_LEAKY_0_1, ACT_Op_LEAKY_0_2, ACT_Op_LEAKY_0_01]> {
  let cppNamespace = "::npux";
}


def Npux_ComputeRunOp : Npux_Op<"compute_run"> {
  let summary = "Unified Compute Operation (GEMM, CONV, GEMV)";
  let description = [{
    Execute a compute operation on the NPU. This op maps to the unified
    'npu_compute_run' C-API and handles Matrix Multiplication, Convolution,
    and Matrix-Vector Multiplication.
  }];

  let arguments = (ins
      // --- 1. Operation Control ---
      COMPUTE_OpTypeAttr:$op_type,
      I32:$precision,
      I1:$is_quant,

      // --- 2. Buffers (SRAM Addresses) ---
      NpuSramMemRef:$input_a,
      NpuSramMemRef:$input_b,
      NpuSramMemRef:$bias,
      NpuSramMemRef:$output,

      // --- 3. Dimensions (GEMM/GEMV/CONV sizes) ---
      I32:$dim_h,
      I32:$dim_w,
      I32:$dim_c_in,
      I32:$dim_c_out,

      // --- 4. Strides ---
      I32:$stride_in_a,
      I32:$stride_in_b,
      I32:$stride_bias,
      I32:$stride_out,

      // --- 5. Convolution Specifics ---
      I32:$kernel_size,
      I32:$stride,
      I32:$dilation,
      I32:$pad_top, I32:$pad_bottom, I32:$pad_left, I32:$pad_right,
      I1:$is_group_conv,

      // --- 6. Post-Process (Accumulation & Activation) ---
      I32:$out_width,  // Output Feature Map Width
      I32:$out_height, // Output Feature Map Height
      I1:$do_accumulate,
      I1:$do_relu,
      ACT_TypeAttr:$relu_type,

      // --- 7. Quantization Parameters ---
      I32:$input_a_zp,
      I32:$input_b_zp,
      I32:$output_zp,
      I32:$quant_scale,
      I32:$quant_shift
  );

  let results = (outs);

  let assemblyFormat = [{
    $op_type 
    `cfg` `(` $precision `,` $is_quant `)`
    `buffers` `(` $input_a `,` $input_b `,` $bias `)` `->` $output
    `shape` `(` $dim_h `,` $dim_w `,` $dim_c_in `,` $dim_c_out `)`
    `strides` `(` $stride_in_a `,` $stride_in_b `,` $stride_bias `,` $stride_out `)`
    `conv` `(` 
        $kernel_size `,` $stride `,` $dilation `,` 
        `pad` `[` $pad_top `,` $pad_bottom `,` $pad_left `,` $pad_right `]` `,`
        $is_group_conv 
    `)`
    `pp` `(` 
        `size` `[` $out_width `,` $out_height `]` `,`
        $do_accumulate `,` $do_relu `,` $relu_type 
    `)`
    `quant` `(` 
        $input_a_zp `,` $input_b_zp `,` $output_zp `,` 
        $quant_scale `,` $quant_shift 
    `)`
    attr-dict 
    `:` type($input_a) `,` type($input_b) `,` type($bias) `,` type($output)
  }];
}