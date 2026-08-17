# Versa-P Scheduler Roadmap

This document records the current Versa-P compiler scheduling scope and the
remaining work required to lower complete NPUX graphs to the current RTL ABI.
The source of truth for descriptor fields and bank ownership is
`C:\set\VersaEdge\rtl`.

## Implemented

- DMA-A, DMA-W, metadata DMA, residual DMA, SA GEMM, and MVOUT descriptors.
- SA GEMM bank scheduling for A/W/O, metadata, ResAdd, and ACC partial
  accumulation.
- Static GEMM output selection for raw INT32 and BF16, including the matching
  O-bank capacity and MVOUT row-byte validation.
- Static INT8 GEMM output with RTL per-tensor scale encoding when the computed
  Q8.24 multiplier is exactly representable as `scale_addr << 4`. The compiler
  records that multiplier on `npux.compute_run` and verifies the descriptor
  scale field before static selection.
- Static INT8 GEMM output with per-channel Q8.24 scale metadata. The explicit
  `npux.mvin_scale` operation loads exactly one 32-byte scale word per eight
  output channels into the metadata bank and selects the RTL per-channel mode.
- Static INT8 GEMM output with Bias plus per-channel Q8.24 scale metadata.
  The explicit `npux.mvin_metadata` operation transfers one packed buffer by
  AUX_MVIN: `ceil(N/8)` bias words followed by `ceil(N/8)` scale words.
  Descriptor scheduling sets `bias_addr=0` and `scale_addr=ceil(N/8)`, so both
  consumers use the same physical metadata bank correctly.
- Source-level packing for independent INT32 Bias and Q8.24 scale arrays. A
  static GEMM with `npu.versa_p_pack_metadata=1` and inputs `(A, W, bias,
  scale)` materializes a temporary host metadata buffer through
  `npux.pack_metadata`, submits its `mvin_metadata`, then releases it. The
  runtime helper zero-pads the final incomplete eight-channel group.
- Static multi-slice GEMM ACC scheduling. Frontend lowering preserves
  `first`/`body`/`final` ACC stages; RegionMark maps them to
  write-partial, accumulate-and-write-partial, and accumulate-with-final-MVOUT
  descriptors. Partial commands retain only ACC-bank liveness, not O-bank
  liveness.
- SA Conv descriptor encoding and static DMA--Conv--DMA region marking.
- Explicitly paired QK/PV attention scheduling with QK logP MVOUT ordering.
- GEMV descriptor scheduling with A/W/metadata/O dependencies. The current
  RTL top-level GEMV format is BF16 and writes 16-bit elements to an O bank.
- Static `npu_gemv` frontend conversion to `npux.gemv_run`, with explicit A,
  W, metadata, and O operands. RegionMark now emits the complete
  DMA-A/DMA-W/DMA-meta/GEMV/MVOUT descriptor sequence.
- VPU SPECIAL descriptor scheduling using only O0/O1.
  - BF16 input: RMSNorm, LayerNorm, Softmax, GELU, and Sigmoid.
  - INT8 input/output: Transpose and PoolMax.
  - BF16 SPECIAL output may be BF16, FP16, or INT8. INT8 output requires a
    Q8.24 inverse scale.
- O-bank liveness tracking. A VPU source must contain a live GEMM/GEMV/VPU
  result, and a VPU destination must be the other empty O bank. A VPU command
  consumes and releases its source O bank while retaining its destination.
- Runtime command API slots for GEMV and VPU.
- `npux.vpu_run` and an O-bank-aware static GEMV -> VPU -> MVOUT region. The
  GEMV result remains live in its selected O bank; VPU writes the opposite
  bank; the final MVOUT drains only that destination bank.
- An O-bank-aware static Conv(INT8) -> VPU(Transpose/PoolMax) -> MVOUT
  region. The Conv producer omits its own MVOUT and the final VPU MVOUT drains
  the opposite O bank. Transpose swaps rows/columns; PoolMax is the RTL's
  fixed INT8 2x2, stride-2 operation.
- Automatic source-level `npu_maxpool` selection after an INT8 Conv when both
  values are static 2D O-bank views, the input dimensions are even, and the
  output dimensions are exactly half. Unsupported pool geometry remains on
  the legacy resample path.
- An O-bank-aware static GEMM(BF16) -> VPU(BF16 SPECIAL) -> MVOUT region.
  The GEMM producer selects BF16 output, retains it in an O bank, and lets the
  VPU drain its opposite O bank through the sole final MVOUT. Softmax,
  LayerNorm, GELU, and Sigmoid immediately following a BF16 GEMM are selected
  automatically.
- Automatic QK/PV pairing when QK MVOUT and PV A-MVIN use the same host
  buffer, followed by the existing static shape and QK/PV ABI validation.
- QK gamma metadata scheduling. QK emits
  `MVIN_A -> MVIN_W -> AUX_MVIN(gamma) -> SA_QK -> MVOUT(logP)`. The first
  little-endian i32 in the metadata buffer is a positive Q8.24 gamma and SA
  programs `scale_addr=0`. Source `linalg.generic` supplies the gamma buffer
  as its third input and carries `npu.versa_p_qk_gamma_q8_24`.

## QK Gamma And Quantization Granularity

The current RTL reads one `gamma16_fix` value per QK descriptor. For
per-tensor Q/K quantization, descriptor gamma is:

```text
gamma_q8_24 = round(Sq * Sk / Sscore / sqrt(headDim) * 2^24)
```

`Sq` and `Sk` are the Q/K dequantization scales and `Sscore` is the scale
expected by QK postprocess. Gamma must be supplied explicitly because a
compiler cannot infer `Sq`, `Sk`, or the intended score domain from integer
memref shapes.

Per-channel needs to be classified by its axis, rather than by its name:

- A scale varying along the QK reduction dimension `D` is inside the dot
  product and cannot be recovered by a post-dot gamma.
- Per-token Q/K scales are outside `D`, but yield
  `gamma[m,n] = Sq[m] * Sk[n] / sqrt(D)`. A 32x32 tile therefore has 1024
  elementwise gamma values, although they factor into 32 row and 32 column
  scales.
- The current one-gamma descriptor is correct only when gamma is constant for
  the whole tile. Splitting a tile at every token scale boundary is correct
  but can reduce it to 1x1 work and is not an efficient implementation.

SmoothQuant's offline per-input-channel smoothing scale is normally absorbed
into the Q/K projection weights; it does not by itself require QK gamma to be
per-channel. AWQ and GPTQ are commonly weight-only, group-wise formats, so
they likewise do not imply a per-channel QK postprocess gamma.

To efficiently support per-token Q/K quantization, the RTL needs a compatible
extension: metadata vectors for Q row scales and K column scales, and QK
postprocess must multiply `dot[m,n] * Sq[m] * Sk[n] / sqrt(D)`. The scheduler
can then load 32 row plus 32 column scales for a 32x32 tile instead of 1024
independent gamma values. Until that ABI exists, Versa-P QK remains restricted
to a per-tensor gamma for each descriptor tile.

## Not Yet Implemented

### VPU Frontend Coverage

`npux.vpu_run` is deliberately restricted to an O-bank producer and final
MVOUT. The existing generic `npux.sfu_run` remains an SRAM path. The explicit
`npu_versa_p_vpu` library call now creates `npux.vpu_run` from
`vpu_function`, precision, and Q8.24 inverse-scale attributes. Softmax,
LayerNorm, GELU, and Sigmoid immediately following an `npu_gemv` writer are
also selected automatically. INT8 Transpose/PoolMax can now be selected with
the explicit `npu_versa_p_vpu` call after a static Conv producer. The
two-dimensional INT8 `npu_transpose` directly after a static Conv is also
selected automatically. Softmax, LayerNorm, GELU, and Sigmoid directly after
a static BF16 GEMM are selected automatically. PoolMax automatic selection is
currently limited to static 2D O-bank views; higher-rank layouts require an
explicit O-bank layout view before they can use the VPU path.

### Complex SA Regions

Static RegionMark now converts a preceding `npux.mvin_bias` into the normal
metadata-bank AUX_MVIN required by the descriptor. It supports static GEMM and
Conv bias regions when the legacy lowering's `acc_bias=1,is_accumulate=1`
combination denotes a first bias slice; the descriptor uses the RTL-legal
`bias=1,accumulate=0` combination.

For a source `npu.resadd=1` operation with three inputs, the lowering now
emits `npux.resadd_load` for the third input. Static GEMM and Conv regions
validate its exact output size and schedule it as the descriptor's residual
AUX_MVIN before SA execution. GEMM residuals use the FP32 output layout and
Conv residuals use the INT8 output layout.

The following still require complete region extraction and tests:
- Group convolution: the current SA descriptor has no group field and the RTL
  `sa_compute_scheduler` drives `conv_is_group` low, so this requires an RTL
  ABI extension before compiler scheduling can enable it.
- Multi-tile output handling and metadata ping-pong across a full graph.

### Global Tiling And Scheduling

The existing NPU tiling passes choose the loop tiles. RegionMark now keeps one
scheduler per MLIR block for ordinary static GEMM/Conv regions in IR order, so
bank rotation, metadata ping-pong, and command dependencies span adjacent
tiles without crossing control-flow boundaries.
The remaining graph-level work is to derive descriptor-safe tile sizes from
large dynamic shapes and carry O-bank lifetimes across mixed SA/GEMV/VPU
regions, rather than only across adjacent static SA tiles.

## Recommended Implementation Order

1. Extend full-graph tiling: derive legal descriptor tile sizes and preserve
   A/W/metadata/O-bank lifetimes across mixed GEMM, Conv, GEMV, and VPU
   regions.
2. Address group convolution only together with an RTL descriptor/issue-path
   extension. Compiler-side scheduling must remain disabled until
   `conv_is_group` is representable and propagated by the RTL.

## Validation Plan

- Descriptor unit tests: exact descriptor words for GEMV and every VPU
  SPECIAL precision combination.
- Scheduler unit tests: O-bank liveness, bank conflicts, and command
  dependencies for GEMV/Conv/GEMM -> VPU -> MVOUT, ACC partial chains, and
  quantized output mode selection.
- MLIR tests: region marking and LLVM lowering for static GEMM, GEMV, Conv,
  bias/ResAdd, and VPU paths.
- RTL simulation: run the corresponding VersaEdge GEMV, VPU, Conv, and
  attention testbenches with compiler-emitted descriptors.
