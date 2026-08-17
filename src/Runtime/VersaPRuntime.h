// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { NPU_VERSA_P_API_MVIN_A = 0, NPU_VERSA_P_API_MVIN_W = 1,
  NPU_VERSA_P_API_AUX_MVIN = 2, NPU_VERSA_P_API_SA = 3,
  NPU_VERSA_P_API_MVOUT = 4 };

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

// Patches DESC0[31:0] with dram_base, writes descriptor fields before start,
// and returns zero only when the command was accepted for execution.
int32_t npu_versa_p_submit(
    uint8_t api, uint64_t desc0, uint64_t desc1, uint64_t desc2, uint32_t dram_base);
// Returns zero for done, nonzero hardware/mock error code otherwise.
int32_t npu_versa_p_wait(uint8_t api);
void npu_versa_p_clear_status(uint8_t api);
void npu_versa_p_global_clear(void);

// Host-mock test hooks. Board runtime need not export these.
uint64_t npu_versa_p_mock_last_desc0(uint8_t api);
void npu_versa_p_mock_inject_error(uint8_t api, uint8_t error_code);

#ifdef __cplusplus
}
#endif