# Versa-P O Bank 与 VPU/SA 并行性记录

本文记录当前双 O bank 的并行能力、是否增加 O bank 的决策条件，以及
SA 仅写 ACC 时与 VPU 并行的可行性。最终是否扩展 O bank 数量应由周期
profiling 和综合结果决定，不能只根据数据流直觉决定。

## 1. 当前硬件结构

- O bank 有两片，每片 `2048 x 256 bit = 64 KiB`，合计 128 KiB。
- 两片 SRAM 可以在不同 bank 上同时进行一次 load/write 和一次 use/read。
- SA postprocess、GEMV 和 VPU 共享一个 O-bank 写仲裁入口。
- MVOUT DMA 和 VPU 共享一个 O-bank 读仲裁入口。
- VPU SPECIAL 通常同时读取 source O bank 并写入 destination O bank，因此会
  同时占用 O-bank use owner 和 load owner。

双 bank 很适合普通的计算/搬运 ping-pong：

```text
SA/GEMV(tile i+1) -> write O1
MVOUT(tile i)      <- read  O0
```

同时，下一 tile 的 A/W/metadata DMA 仍可使用各自的 bank 预取。当前结构的
主要限制是 VPU 同时占用 O bank 的读、写 ownership，而不是 O bank 总容量。

## 2. 是否增加 O bank

当前结论是暂时保留两个 O bank，完成 profiling 后再决定。

仅增加 SRAM bank 数量而不修改 owner、仲裁器和端口不会提高并行度。若要让
SA 与 VPU 并行，至少需要第三片 O bank，并同时提供可并行的生产者写通路：

```text
O0: VPU source
O1: VPU destination
O2: SA/GEMV destination
```

若还要求 MVOUT 同时读取上一结果，在结果完整落 bank 后再消费的模型下，
通常需要第四个 live bank。第三/第四个 bank 分别使当前 O-bank SRAM 容量
增加 50%/100%，并且需要扩展 bank selector、owner 状态、仲裁和验证环境。

测试时至少记录以下指标：

- SA、GEMV、VPU、MVOUT、A/W/metadata DMA 的 busy cycles。
- `o_load_ready` 和 `o_use_ready` 导致的等待周期。
- O-bank 读写仲裁等待周期，以及各请求方获得的 beat 数。
- VPU 与下一层 A/W/metadata 预取的重叠比例。
- VPU 不可隐藏周期占整层周期的比例。
- Prefill 和 Decode 分别统计，不能用一种模式代替另一种模式。
- 两 bank、三 bank 和流式旁路方案的周期、BRAM/URAM、LUT、时序结果。

建议只有在 VPU/O-bank 冲突形成稳定的主要瓶颈，并且新增端口后的性能收益
能够覆盖存储与控制面积时，才增加 bank。否则优先保留双 bank，并通过跨层
预取、减少 MVOUT/MVIN 往返或 SA/GEMV 到 VPU 的小 FIFO/流式旁路隐藏延迟。

## 3. SA 仅写 ACC 时的并行机会

多 K-slice GEMM 的 first/body 阶段使用 `write_partial=1`。此时 postprocess
将 INT32 partial sum 写入独立 ACC ping-pong bank，不产生 O-bank 写请求。
因此，从数据通路资源看，下面的并行是成立的：

```text
VPU: read O0 -> write O1
SA:  read A/W, optional read old ACC -> write new ACC
DMA: 可继续预取未冲突的 A/W/metadata bank
```

必要条件包括：

- SA 与 VPU 使用独立计算数据通路。
- SA 的 A/W bank 已经 valid，且未与其他 A/W 使用者冲突。
- write-partial 的源/目标 ACC bank 满足 ping-pong ownership。
- SA 不需要被 VPU 占用的 O load owner，也不能改变任何 O busy/valid 状态。
- final K-slice 会写 O bank，因此 final 阶段仍必须等待 VPU 释放 O load owner。

这意味着长 K 维 GEMM 可以把 first/body partial SA 与 VPU 重叠，最后一片再
等待 O bank。该方式可能比直接增加 O bank 更有面积效率。

## 4. 新 RTL 与编译器同步状态

新 dispatcher 已经按命令的实际输出资源区分普通 O 输出和 ACC partial 输出：

- 普通 GEMM `write_partial=1` 时不申请、不更新 O load owner 或 O busy/valid。
- `acc_bank` 表示 ACC source bank，partial destination 为 `~acc_bank`。
- dispatcher 会检查并占用 ACC destination，完成后将其标记为 valid。
- QK/PV 单独解码，不能把 QK 使用的 descriptor bit 60 当成普通 GEMM
  `write_partial`。

编译器已按同一约定同步：

- `writePartial` SA 只声明写 `BankResource::Accumulator`。
- partial command 不声明读写 `BankResource::O`。
- final command 才写 O bank 并在需要时追加 MVOUT。
- first descriptor 选择一个 ACC source selector，并把相反的物理 bank 记录为
  `partialOutputBank`；body 从该 bank 读取并写另一 bank；final 从最后一个
  partial bank 读取并写 O bank。
- 调度测试固定了 first/body/final 的 descriptor bank 编码，并验证 partial SA
  不依赖正在占用 O0/O1 的 VPU，而 final SA 必须等待相应 O-bank ownership。

## 5. 仍需修正的 RTL 顶层细节

`npu_top.sv` 的 O-bank `load_req_valid_i` 已用 `sa_compute_writes_o` 门控，但
`load_done_i` 仍直接包含所有 `sa_compute_done`。如果 partial SA 与写 O 的 VPU
并行，partial 的完成脉冲可能提前结束 VPU 的 O-bank load ownership。

这里还需要用启动时锁存的 SA writes-O 状态门控完成脉冲，例如只将
`sa_compute_done && active_sa_writes_o` 送入 O bank。完成这一点并通过 RTL
并行仿真前，不能仅凭 dispatcher 修复认定硬件端到端并行已经安全。

## 6. 后续工作

- 修正并验证上述 O-bank `load_done_i` 门控。
- 在同一个跨 mixed-region 调度上下文中同时观察 VPU 和 ACC partial SA。
- 建立周期模型，比较“partial SA 与 VPU 重叠”和“增加第三 O bank”的收益。
- 增加 RTL 仿真：VPU 占用 O0/O1 时，partial SA 正常完成且 O
  busy/valid 不变；ACC destination 在完成后变为 valid；下一 body slice 从
  同一 bank 读取；final SA 必须等待 VPU 完成。

## 7. 当前决策

在完成顶层完成信号门控和性能测试之前：

- 不增加 O bank 数量。
- 保留双 O-bank ping-pong。
- 优先验证 ACC-only SA 与 VPU 并行。
- 使用测得的不可隐藏 VPU 周期和综合面积作为扩 bank 的最终依据。
