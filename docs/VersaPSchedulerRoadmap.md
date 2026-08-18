# Versa-P Scheduler Roadmap

This document records the current Versa-P compiler scheduling scope and the
remaining work required to lower complete NPUX graphs to the current RTL ABI.
The source of truth for descriptor fields and bank ownership is
`C:\set\VersaEdge\rtl`.

O-bank/VPU parallelism, the deferred bank-count decision, and the ACC-only SA
opportunity are recorded in `VersaPOBankParallelism_CN.md`.

## 2026-08-18 RTL 同步记录

今天完成的编译器与最新 RTL 同步包括：

- 编译器级 GEMM 分块不再把 RTL 内部 32x32 微分块当作 descriptor tile。
  `planGemmTiles` 根据 A/W/O/metadata bank 容量、实际搬运量、数据复用、
  descriptor 数量和边界浪费搜索合法 tile；自动 GEMM tiling 使用相同的容量
  与流量模型。
- Bias 和 per-channel scale 按连续 metadata 布局合并，由一次 AUX_MVIN 尽量
  搬运当前 descriptor tile 所需的全部 scale，而不是逐 scale 单独发起 DMA。
- QK 同时支持 scalar Q8.24 gamma 和 row-column gamma。后者通过 QK 专用
  descriptor 传递 Q/K scale metadata 的 base/stride，并由一次 metadata DMA
  搬运当前 compiler tile 的 Q 行 scale 和 K 列 scale。
- QK block-max 改为独立双 slot 资源；scheduler 将 gamma metadata bank 与
  block-max slot 配对分配，并保证 QK SA 到 logP MVOUT 的生产消费依赖。
- dispatcher 的每 opcode “一条执行、一条 pending”窗口已映射为调度器的
  两条 in-flight 命令窗口，第三条同 opcode 命令等待最早命令完成。
- 普通 GEMM `write_partial` 已与新 ACC scoreboard 对齐：descriptor 的
  `acc_bank` 是 source，partial 写入相反 bank；first/body/final 的 ACC
  ping-pong 和依赖均由单测固定。
- ACC-only partial SA 不声明 O-bank 资源，可以与占用 O0/O1 的 VPU 并行；
  final SA 仍通过 O-bank ownership 依赖等待 VPU。
- VPU SPECIAL function `0x6` 按当前 RTL 语义从 Sigmoid 更正为 SiLU；
  `npu_sigmoid` 不再错误下发为 function `0x6`。

仍有一个 RTL 顶层问题未解决：`npu_top.sv` 已用
`sa_compute_writes_o` 门控 O-bank `load_req_valid_i`，但 `load_done_i` 仍无条件
接收 `sa_compute_done`。partial SA 与写 O 的 VPU 并行时，partial 完成可能提前
结束 VPU 的 O-bank load ownership。RTL 应锁存启动命令的 writes-O 属性，并只
把 `sa_compute_done && active_sa_writes_o` 送入 O bank；完成修正和并行仿真前，
不能认为端到端 SA/VPU 并行已经安全。

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
  liveness. The descriptor ACC selector is the source bank and each partial
  writes its opposite bank, matching the current dispatcher and ACC-bank RTL.
- ACC-only partial SA scheduling can overlap a VPU that owns O0/O1; the final
  SA command waits for O-bank ownership before publishing its result. RTL
  end-to-end use still requires `npu_top` to gate O-bank `load_done_i` with the
  active SA writes-O state.
- SA Conv descriptor encoding and static DMA--Conv--DMA region marking.
- Explicitly paired QK/PV attention scheduling with QK logP MVOUT ordering.
- GEMV descriptor scheduling with A/W/metadata/O dependencies. The current
  RTL top-level GEMV format is BF16 and writes 16-bit elements to an O bank.
- Static `npu_gemv` frontend conversion to `npux.gemv_run`, with explicit A,
  W, metadata, and O operands. RegionMark now emits the complete
  DMA-A/DMA-W/DMA-meta/GEMV/MVOUT descriptor sequence.
- VPU SPECIAL descriptor scheduling using only O0/O1.
  - BF16 input: RMSNorm, LayerNorm, Softmax, GELU, and SiLU.
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
  LayerNorm, GELU, and SiLU immediately following a BF16 GEMM are selected
  automatically.
- Automatic QK/PV pairing when QK MVOUT and PV A-MVIN use the same host
  buffer, followed by the existing static shape and QK/PV ABI validation.
- QK gamma metadata scheduling. QK emits
  `MVIN_A -> MVIN_W -> AUX_MVIN(gamma) -> SA_QK -> MVOUT(logP)`. The first
  little-endian i32 in the metadata buffer is a positive Q8.24 gamma and SA
  programs `scale_addr=0`. Source `linalg.generic` supplies the gamma buffer
  as its third input and carries `npu.versa_p_qk_gamma_q8_24`.

## QK Gamma And Quantization Granularity

### 简要结论

当前 RTL 和调度器同时支持 scalar gamma 与 row-column gamma。scalar 模式
适用于整个 QK descriptor tile 共用一个 Q8.24 gamma 的 per-tensor 量化；
row-column 模式适用于常见的 INT8 per-token Q/K 量化，其中 Q 的每个 token
有一个 `Sq[m]`，K 的每个 token 有一个 `Sk[n]`，输出使用
`gamma[m,n] = Sq[m] * Sk[n] / sqrt(D) / Sscore`；一个 RTL 32x32 微分块虽有
1024 个有效 gamma，但它们可由 32 个行 scale 和 32 个列 scale 的外积得到，
不需要保存 1024 个独立数值。descriptor 给出 Q/K scale metadata 的起始地址
和步长，QK postprocess 为每个输出组合对应的行、列 scale。若 scale 沿归约
维 D 变化，则缩放在 dot-product 求和内部，仍不能靠计算后的 gamma 修正，
需要在 MAC 前实现逐 D 缩放；当前 ABI 不支持这种情况。

The current RTL reads one `gamma16_fix` value per QK descriptor. For
per-tensor Q/K quantization, descriptor gamma is:

```text
gamma_q8_24 = round(Sq * Sk / Sscore / sqrt(headDim) * 2^24)
```

The QK gamma remains in a normal metadata bank. New RTL stores QK block-max
values in a dedicated two-slot buffer: `SA_COMPUTE DESC2[60]` selects the
gamma metadata bank and the writer uses the opposite block-max slot; QK
`MVOUT DESC1[49]` consumes that slot. The scheduler therefore allocates these
two resources as a pair and does not treat MVOUT bit 49 as a metadata-bank
selector.

The latest QK postprocess also supports row-column gamma. For each score it
computes `round((Sq_q8_24 * Sk_q8_24) / 2^24)`, saturates the result to 25-bit
unsigned Q8.24, then applies the existing dot-product multiply and rounding.
`SA_QK_GAMMA_DESC` at MMIO `0xd8` selects this mode and provides Q/K metadata
word bases; scales are eight i32 Q8.24 values per word and the RTL requires a
four-word stride per 32-element microtile. The QK SA operation carries it as
`npux.versa_p_qk_gamma_desc`; lowering emits the QK-only runtime submission,
which writes `0xd8` after any same-SA dependency completes and immediately
before QK start. Scalar QK explicitly writes zero to clear a preceding
row-column configuration. Other engines and non-QK SA operations do not write
this register.

`Sq` and `Sk` are the Q/K dequantization scales and `Sscore` is the scale
expected by QK postprocess. Gamma must be supplied explicitly because a
compiler cannot infer `Sq`, `Sk`, or the intended score domain from integer
memref shapes.

Per-channel needs to be classified by its axis, rather than by its name:

- A scale varying along the QK reduction dimension `D` is inside the dot
  product and cannot be recovered by a post-dot gamma.
- Per-token Q/K scales are outside `D`, but yield
  `gamma[m,n] = Sq[m] * Sk[n] / sqrt(D)`. A 32x32 microtile therefore has 1024
  elementwise gamma values, although they factor into 32 row and 32 column
  scales.
- The current one-gamma descriptor is correct only when gamma is constant for
  the whole tile. Splitting a tile at every token scale boundary is correct
  but can reduce it to 1x1 work and is not an efficient implementation.

SmoothQuant's offline per-input-channel smoothing scale is normally absorbed
into the Q/K projection weights; it does not by itself require QK gamma to be
per-channel. AWQ and GPTQ are commonly weight-only, group-wise formats, so
they likewise do not imply a per-channel QK postprocess gamma.

The row-column ABI now carries metadata vectors for Q row scales and K column
scales. The scheduler loads all scales needed by one compiler descriptor tile
in one AUX_MVIN rather than reloading them for every RTL 32x32 microtile. The
remaining unsupported case is a scale varying along reduction dimension `D`,
which would require scaling inside the dot-product accumulation.

## Not Yet Implemented

### VPU Frontend Coverage

`npux.vpu_run` is deliberately restricted to an O-bank producer and final
MVOUT. The existing generic `npux.sfu_run` remains an SRAM path. The explicit
`npu_versa_p_vpu` library call now creates `npux.vpu_run` from
`vpu_function`, precision, and Q8.24 inverse-scale attributes. Softmax,
LayerNorm, GELU, and SiLU immediately following an `npu_gemv` writer are
also selected automatically. INT8 Transpose/PoolMax can now be selected with
the explicit `npu_versa_p_vpu` call after a static Conv producer. The
two-dimensional INT8 `npu_transpose` directly after a static Conv is also
selected automatically. Softmax, LayerNorm, GELU, and SiLU directly after
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

`VersaPBankScheduler::planGemmTiles` exhaustively selects compiler descriptor
tiles from local A/W/O/metadata bank capacities. Its objective counts outer
tile A/W/O/metadata traffic, descriptor count, and RTL 32x32 edge waste, so
it captures data reuse without equating an RTL microtile with a compiler tile.
The automatic NPU GEMM tiler applies the same A/W/O capacity and traffic model
when materializing its spatial and K loops. `planGemmTiles` exposes the
metadata-aware form of that decision for Versa-P lowering, where scale and
bias metadata are known. RegionMark now keeps one
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
