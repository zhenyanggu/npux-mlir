//=============================================
// src/Dialect/Npux/Npux.td
// this file defines npux Dialect
//=============================================

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/EnumAttr.td"


def IsNpuSram : CPred<"llvm::cast<::mlir::MemRefType>($_self).getMemorySpaceAsInt() == 2">;
def NpuSramMemRef : Type<IsNpuSram, "memref in NPU SRAM", "::mlir::MemRefType">;

def IsNpuAcc : CPred<"llvm::cast<::mlir::MemRefType>($_self).getMemorySpaceAsInt() == 3">;
def NpuAccMemRef : Type<IsNpuAcc, "memref in NPU Accumulator", "::mlir::MemRefType">;

def IsNpuSramOrAcc : CPred<"llvm::cast<::mlir::MemRefType>($_self).getMemorySpaceAsInt() == 2 || llvm::cast<::mlir::MemRefType>($_self).getMemorySpaceAsInt() == 3">;
def NpuSramOrAccMemRef : Type<IsNpuSramOrAcc, "memref in NPU SRAM or ACC", "::mlir::MemRefType">;

def DATAFLOW_WS : I32EnumAttrCase<"ws", 0>; 
def DATAFLOW_OS : I32EnumAttrCase<"os", 1>; 
def DataflowModeAttr
    : I32EnumAttr<"DataflowMode", "Dataflow Mode",
          [DATAFLOW_WS, DATAFLOW_OS]> {
  let cppNamespace = "::npux";
}


def DEST_SPM : I32EnumAttrCase<"spm", 0>;
def DEST_ACC : I32EnumAttrCase<"acc", 1>;

def AccoutDestAttr
    : I32EnumAttr<"AccoutDest", "Compute Output Destination",
          [DEST_SPM, DEST_ACC]> {
  let cppNamespace = "::npux";
}

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


def Npux_AccAllocOp : Npux_Op<"acc_alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate memory in NPU Accumulator (ACC)";
  let description = [{
    Allocates a buffer specifically in the NPU Accumulator (Memory Space 3).
    Used for storing intermediate high-precision results before moving to SRAM.
  }];

  let arguments = (ins Variadic<Index>:$dynamic_sizes);
  let results = (outs Res<NpuAccMemRef, "allocated acc buffer", [MemAlloc]>:$memref);

  let assemblyFormat = "(`(` $dynamic_sizes^ `)`)? attr-dict `:` type($memref)";
}

def Npux_AccFreeOp : Npux_Op<"acc_free", [MemoryEffects<[MemFree]>]> {
  let summary = "Free memory in NPU Accumulator (ACC)";
  let arguments = (ins NpuAccMemRef:$acc_memref);
  let results = (outs);

  let assemblyFormat = "$acc_memref attr-dict `:` type($acc_memref)";
}

def Npux_MvAccToSpmOp : Npux_Op<"mv_acc_to_spm"> {
  let summary = "Logical move from ACC to SPM";
  let description = [{
    Represents a logical data transfer from Accumulator (Space 3) to SRAM (Space 2).
    
    Hardware Note: 
    This op might not correspond to a standalone hardware instruction. 
    It is typically fused into the compute operation (e.g., GEMM/CONV) 
    during lowering, where the quantization parameters will be applied.
  }];

  let arguments = (ins 
      NpuAccMemRef:$acc_src, 
      NpuSramMemRef:$spm_dst,

      // 仅保留数据搬运所需的维度和步长信息
      I16:$col_num, I16:$row_num, 
      I16:$acc_stride, I16:$spm_stride
  );

  let results = (outs);

  let assemblyFormat = [{
    $acc_src `->` $spm_dst
    `shape` `(` $col_num `x` $row_num `)`
    `stride` `(` $acc_stride `,` $spm_stride `)`
    attr-dict `:` type($acc_src) `,` type($spm_dst)
  }];
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
  let summary = "Host -> SRAM/ACC DMA";

  let arguments = (ins 
      AnyMemRef:$host_ptr, 
      NpuSramOrAccMemRef:$dst_memref, 

      I16:$col_num, I16:$row_num, I16:$sram_stride, I16:$dram_stride,
      I8:$precision, I8:$input_type, 
      
      I8:$dest, 
      I1:$is_bias, 

      I1:$is_quant, I32:$quant_zero, I16:$quant_scale, I16:$quant_shift);

  let results = (outs);

  let assemblyFormat = [{
    $host_ptr `,` $dst_memref 
    `shape` `(` $col_num `x` $row_num `)` 
    `stride` `(` $sram_stride `,` $dram_stride `)`
    `cfg` `(` $precision `,` $input_type `,` $dest `,` $is_bias `)` 
    `quant` `(` $is_quant `,` $quant_zero `,` $quant_scale `,` $quant_shift `)`
    attr-dict `:` type($host_ptr) `,` type($dst_memref)
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
  let summary = "Low-level mapped compute operation";
  let description = [{
    Directly maps to npu_conv_run. 
    Reuses biaspsum_* fields for output geometry control to match hardware behavior.
  }];

  let arguments = (ins
      // --- 1. Operation Control ---
      COMPUTE_OpTypeAttr:$op_type,
      DataflowModeAttr:$dataflow_mode,
      AccoutDestAttr:$accout_dest,
      I8:$int_type, 
      
      // --- 2. Buffers (提供地址) ---
      NpuSramMemRef:$input_a, 
      NpuSramMemRef:$input_b, 
      
      // [你的疑问在这里]
      // 这个参数是为了提供 API 里的 `biaspsum_addr`。
      // 如果没有它，Lowering 时不知道该填什么地址作为 bias/psum 的读基地址。
      Optional<NpuAccMemRef>:$biaspsum_memref, 

      // 输出 Buffer (提供 output_addr) - 支持 Sram 或 Acc
      NpuSramOrAccMemRef:$output,

      // --- 3. Padding ---
      I32:$pad_top, I32:$pad_bottom, I32:$pad_left, I32:$pad_right,
      I32:$pad_mode,

      // --- 4. Kernel / Weight ---
      I32:$weight_shape_m1,    
      I32:$weight_stride_m1,
      I32:$weight_dilation_m1,
      I1:$is_group_conv,

      // --- 5. Input Geometry ---
      I32:$input_a_col_num_m1, I32:$input_a_row_num_m1, I32:$input_a_stride,
      I32:$input_b_col_num_m1, I32:$input_b_row_num_m1, I32:$input_b_stride,

      // --- 6. Bias/Psum & Output Loop Control (复用!) ---
      // 这里既是 Bias/Psum 的读取形状，也是硬件计算的循环计数器 (Output Shape)
      I32:$biaspsum_width,  
      I32:$biaspsum_height,
      I32:$biaspsum_stride, 

      // --- 7. Output Stride ---
      I32:$output_stride, 

      // --- 8. Post-Processing & Quantization ---
      I1:$is_accumulate,
      I1:$relu_enable,
      ACT_TypeAttr:$relu_type,
      I1:$acc_bias,
      
      I32:$output_zeropoint, 
      I32:$quant_scale, 
      I32:$quant_scaleshift, 
      I32:$input_a_zeropoint, 
      I32:$input_b_zeropoint
  );

  let results = (outs);

  let assemblyFormat = [{
    $op_type 
    $dataflow_mode 
    `dest` `(` $accout_dest `)`
    `prec` `(` $int_type `)`
    
    `buffers` `(` 
        `a` `:` $input_a `,` 
        `b` `:` $input_b 
        (`,` `bias` `:` $biaspsum_memref^)? 
    `)` `->` $output

    `shape_a` `(` $input_a_col_num_m1 `x` $input_a_row_num_m1 `)` `stride` $input_a_stride
    `shape_b` `(` $input_b_col_num_m1 `x` $input_b_row_num_m1 `)` `stride` $input_b_stride
    
    `loop_shape` `(` $biaspsum_width `x` $biaspsum_height `)` 
    `strides` `(` `bias` $biaspsum_stride `,` `out` $output_stride `)`
    
    `conv` `(` 
        `k` $weight_shape_m1 `,` `s` $weight_stride_m1 `,` `d` $weight_dilation_m1 `,` 
        `pad` `[` $pad_top `,` $pad_bottom `,` $pad_left `,` $pad_right `]` `,` `mode` $pad_mode `,`
        `group` $is_group_conv 
    `)`

    `pp` `(` 
        `accum` $is_accumulate `,` `bias_sel` $acc_bias `,` 
        `relu` $relu_enable $relu_type 
    `)`

    `quant` `(` 
        `zp` `[` $input_a_zeropoint `,` $input_b_zeropoint `,` $output_zeropoint `]` `,`
        `scale` $quant_scale `,` `shift` $quant_scaleshift 
    `)`

    attr-dict 
    `:` type($input_a) `,` type($input_b) `,` type($output) (`,` type($biaspsum_memref)^)?
  }];
}
def Npux_SubviewOp : Npux_Op<"subview", [Pure]> {
  let summary = "Simple NPU SRAM/ACC slicing";
  let description = [{
    A simplified subview operation for on-chip memory.
    Supports both SRAM (Space 2) and Accumulator (Space 3).
  }];

  let arguments = (ins
    NpuSramOrAccMemRef:$source,
    Variadic<Index>:$offsets 
  );

  let results = (outs NpuSramOrAccMemRef:$result);
  
  let assemblyFormat = [{
    $source `[` $offsets `]` attr-dict `:` type($source) `to` type($result)
  }];
}