// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { NPU_VERSA_P_API_MVIN_A = 0, NPU_VERSA_P_API_MVIN_W = 1,
  NPU_VERSA_P_API_AUX_MVIN = 2, NPU_VERSA_P_API_SA = 3,
  NPU_VERSA_P_API_MVOUT = 4, NPU_VERSA_P_API_GEMV = 5,
  NPU_VERSA_P_API_VPU = 6 };

// Converts a host virtual address to the 32-bit DMA/physical base consumed by
// DESC0[31:0]. Board runtimes must implement their DMA mapping here; compiler
// lowering must never truncate a host pointer directly.
int32_t npu_versa_p_translate_dram_address(
    const void *host_ptr, uint32_t *dram_base);

// Resolves host_ptr through the translation contract and submits one DMA descriptor.
int32_t npu_versa_p_submit_host(uint8_t api, uint64_t desc0, uint64_t desc1,
    uint64_t desc2, const void *host_ptr);

// Submits and waits atomically. On any failure it issues GLOBAL_CLEAR before
// returning the nonzero status; callers must stop the containing region.
int32_t npu_versa_p_submit_wait_host(uint8_t api, uint64_t desc0,
    uint64_t desc1, uint64_t desc2, const void *host_ptr);
int32_t npu_versa_p_submit_wait_static(
    uint8_t api, uint64_t desc0, uint64_t desc1, uint64_t desc2);

// Fail-closed submission entry points used by compiler lowering. They never
// return after a descriptor failure, so later commands in the same lowered
// region cannot run with stale local-bank state.
void npu_versa_p_submit_wait_host_or_abort(uint8_t api, uint64_t desc0,
    uint64_t desc1, uint64_t desc2, const void *host_ptr);
void npu_versa_p_submit_wait_static_or_abort(
    uint8_t api, uint64_t desc0, uint64_t desc1, uint64_t desc2);

// Dependency-driven asynchronous submission used by the descriptor lowering.
// A command waits only for direct predecessors using the same API. Cross-API
// producer dependencies are submitted immediately; the RTL dispatcher's
// per-opcode pending slots and bank scoreboard gate their launch.
// dependency_count is at most 5.
void npu_versa_p_submit_host_after_or_abort(uint8_t api, uint64_t desc0,
    uint64_t desc1, uint64_t desc2, const void *host_ptr, uint32_t command_id,
    uint8_t dependency_count, uint32_t dependency0, uint32_t dependency1,
    uint32_t dependency2, uint32_t dependency3, uint32_t dependency4);
void npu_versa_p_submit_static_after_or_abort(uint8_t api, uint64_t desc0,
    uint64_t desc1, uint64_t desc2, uint32_t command_id,
    uint8_t dependency_count, uint32_t dependency0, uint32_t dependency1,
    uint32_t dependency2, uint32_t dependency3, uint32_t dependency4);
// Region-final MVOUT uses this variant so DRAM output is visible to following
// IR before the next independent descriptor region starts.
void npu_versa_p_submit_wait_host_after_or_abort(uint8_t api, uint64_t desc0,
    uint64_t desc1, uint64_t desc2, const void *host_ptr, uint32_t command_id,
    uint8_t dependency_count, uint32_t dependency0, uint32_t dependency1,
    uint32_t dependency2, uint32_t dependency3, uint32_t dependency4);

// Patches DESC0[31:0] with dram_base, writes descriptor fields before start,
// and returns zero only when the command was accepted for execution.
int32_t npu_versa_p_submit(
    uint8_t api, uint64_t desc0, uint64_t desc1, uint64_t desc2, uint32_t dram_base);
// Returns zero for done, nonzero hardware/mock error code otherwise.
int32_t npu_versa_p_wait(uint8_t api);
void npu_versa_p_clear_status(uint8_t api);
void npu_versa_p_global_clear(void);

// Packs contiguous i32 Bias and Q8.24 per-channel scale arrays into the
// 256-bit metadata words consumed by SA_COMPUTE. The destination must hold
// 2 * ceil(channel_count / 8) * 8 i32 elements.
void npu_versa_p_pack_metadata_i32(const int32_t *bias, const int32_t *scales,
    int32_t *destination, uint32_t channel_count);

// Host-mock test hooks. Board runtime need not export these.
uint64_t npu_versa_p_mock_last_desc0(uint8_t api);
uint32_t npu_versa_p_mock_wait_count(uint8_t api);
void npu_versa_p_mock_inject_error(uint8_t api, uint8_t error_code);

#ifdef __cplusplus
}
#endif
