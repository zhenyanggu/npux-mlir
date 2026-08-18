// SPDX-License-Identifier: Apache-2.0

#include "src/Runtime/VersaPRuntime.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

typedef struct {
  uint64_t desc0;
  uint64_t desc1;
  uint64_t desc2;
  uint8_t done;
  uint8_t error;
} VersaPStatus;

static VersaPStatus statuses[7];
static uint32_t wait_counts[7];
static uint64_t qk_gamma_desc;

enum { kMaxScheduledCommandId = 64, kMaxDependencies = 5 };
static uint8_t command_apis[kMaxScheduledCommandId];

static int valid_api(uint8_t api) { return api < 7; }

static int32_t npu_versa_p_submit_raw(
    uint8_t api, uint64_t desc0, uint64_t desc1, uint64_t desc2) {
  if (!valid_api(api))
    return 11;
  VersaPStatus *status = &statuses[api];
  if (status->error)
    return status->error;
  status->desc0 = desc0;
  status->desc1 = desc1;
  status->desc2 = desc2;
  status->done = 1;
  return 0;
}

int32_t npu_versa_p_translate_dram_address(
    const void *host_ptr, uint32_t *dram_base) {
  if (!host_ptr || !dram_base)
    return 12;
  // Mock-only identity mapping. Board code must perform DMA/IOMMU mapping.
  *dram_base = (uint32_t)(uintptr_t)host_ptr;
  return 0;
}

int32_t npu_versa_p_submit_host(uint8_t api, uint64_t desc0, uint64_t desc1,
    uint64_t desc2, const void *host_ptr) {
  uint32_t dram_base = 0;
  int32_t status = npu_versa_p_translate_dram_address(host_ptr, &dram_base);
  if (status != 0)
    return status;
  return npu_versa_p_submit(api, desc0, desc1, desc2, dram_base);
}

int32_t npu_versa_p_submit_wait_host(uint8_t api, uint64_t desc0,
    uint64_t desc1, uint64_t desc2, const void *host_ptr) {
  int32_t status = npu_versa_p_submit_host(api, desc0, desc1, desc2, host_ptr);
  if (status == 0)
    status = npu_versa_p_wait(api);
  if (status != 0)
    npu_versa_p_global_clear();
  else
    npu_versa_p_clear_status(api);
  return status;
}

int32_t npu_versa_p_submit_wait_static(
    uint8_t api, uint64_t desc0, uint64_t desc1, uint64_t desc2) {
  // SA DESC0 encodes M/N/K in both halves; it is not a DMA base field.
  int32_t status = npu_versa_p_submit_raw(api, desc0, desc1, desc2);
  if (status == 0)
    status = npu_versa_p_wait(api);
  if (status != 0)
    npu_versa_p_global_clear();
  else
    npu_versa_p_clear_status(api);
  return status;
}

void npu_versa_p_submit_wait_host_or_abort(uint8_t api, uint64_t desc0,
    uint64_t desc1, uint64_t desc2, const void *host_ptr) {
  if (npu_versa_p_submit_wait_host(api, desc0, desc1, desc2, host_ptr) != 0)
    abort();
}

void npu_versa_p_submit_wait_static_or_abort(
    uint8_t api, uint64_t desc0, uint64_t desc1, uint64_t desc2) {
  if (npu_versa_p_submit_wait_static(api, desc0, desc1, desc2) != 0)
    abort();
}

static void abort_on_error(int32_t status) {
  if (status != 0) {
    npu_versa_p_global_clear();
    abort();
  }
}

static void wait_same_api_dependencies_or_abort(uint8_t target_api,
    uint8_t dependency_count,
    uint32_t dependency0, uint32_t dependency1, uint32_t dependency2,
    uint32_t dependency3, uint32_t dependency4) {
  const uint32_t dependencies[kMaxDependencies] = {
      dependency0, dependency1, dependency2, dependency3, dependency4};
  bool waited_same_api = false;
  if (dependency_count > kMaxDependencies)
    abort_on_error(13);
  for (uint8_t index = 0; index < dependency_count; ++index) {
    const uint32_t command_id = dependencies[index];
    if (command_id == 0 || command_id >= kMaxScheduledCommandId)
      abort_on_error(13);
    const uint8_t dependency_api = command_apis[command_id];
    if (!valid_api(dependency_api))
      abort_on_error(13);
    if (dependency_api != target_api || waited_same_api)
      continue;
    abort_on_error(npu_versa_p_wait(target_api));
    npu_versa_p_clear_status(target_api);
    waited_same_api = true;
  }
}

static void remember_command_or_abort(uint32_t command_id, uint8_t api) {
  if (command_id == 0 || command_id >= kMaxScheduledCommandId ||
      !valid_api(api))
    abort_on_error(13);
  command_apis[command_id] = api;
}

void npu_versa_p_submit_host_after_or_abort(uint8_t api, uint64_t desc0,
    uint64_t desc1, uint64_t desc2, const void *host_ptr, uint32_t command_id,
    uint8_t dependency_count, uint32_t dependency0, uint32_t dependency1,
    uint32_t dependency2, uint32_t dependency3, uint32_t dependency4) {
  wait_same_api_dependencies_or_abort(api, dependency_count, dependency0,
      dependency1, dependency2, dependency3, dependency4);
  abort_on_error(npu_versa_p_submit_host(api, desc0, desc1, desc2, host_ptr));
  remember_command_or_abort(command_id, api);
}

void npu_versa_p_submit_static_after_or_abort(uint8_t api, uint64_t desc0,
    uint64_t desc1, uint64_t desc2, uint32_t command_id,
    uint8_t dependency_count, uint32_t dependency0, uint32_t dependency1,
    uint32_t dependency2, uint32_t dependency3, uint32_t dependency4) {
  wait_same_api_dependencies_or_abort(api, dependency_count, dependency0,
      dependency1, dependency2, dependency3, dependency4);
  abort_on_error(npu_versa_p_submit_raw(api, desc0, desc1, desc2));
  remember_command_or_abort(command_id, api);
}

void npu_versa_p_submit_qk_static_after_or_abort(uint64_t desc0,
    uint64_t desc1, uint64_t desc2, uint64_t qk_gamma_desc_value,
    uint32_t command_id, uint8_t dependency_count, uint32_t dependency0,
    uint32_t dependency1, uint32_t dependency2, uint32_t dependency3,
    uint32_t dependency4) {
  wait_same_api_dependencies_or_abort(NPU_VERSA_P_API_SA, dependency_count,
      dependency0, dependency1, dependency2, dependency3, dependency4);
  // Board runtime writes this value to SA_QK_GAMMA_DESC (offset 0xd8).
  qk_gamma_desc = qk_gamma_desc_value;
  abort_on_error(npu_versa_p_submit_raw(
      NPU_VERSA_P_API_SA, desc0, desc1, desc2));
  remember_command_or_abort(command_id, NPU_VERSA_P_API_SA);
}

void npu_versa_p_submit_wait_host_after_or_abort(uint8_t api, uint64_t desc0,
    uint64_t desc1, uint64_t desc2, const void *host_ptr, uint32_t command_id,
    uint8_t dependency_count, uint32_t dependency0, uint32_t dependency1,
    uint32_t dependency2, uint32_t dependency3, uint32_t dependency4) {
  npu_versa_p_submit_host_after_or_abort(api, desc0, desc1, desc2, host_ptr,
      command_id, dependency_count, dependency0, dependency1, dependency2,
      dependency3, dependency4);
  abort_on_error(npu_versa_p_wait(api));
  npu_versa_p_clear_status(api);
}

int32_t npu_versa_p_submit(
    uint8_t api, uint64_t desc0, uint64_t desc1, uint64_t desc2, uint32_t dram_base) {
  if (!valid_api(api))
    return 11;
  VersaPStatus *status = &statuses[api];
  if (status->error)
    return status->error;
  status->desc0 = (desc0 & UINT64_C(0xffffffff00000000)) | dram_base;
  status->desc1 = desc1;
  status->desc2 = desc2;
  // The board implementation writes DESC0/1/2 then the descriptor word
  // containing W1S start. The mock completes synchronously after acceptance.
  status->done = 1;
  return 0;
}

int32_t npu_versa_p_wait(uint8_t api) {
  if (!valid_api(api))
    return 11;
  ++wait_counts[api];
  return statuses[api].error;
}

void npu_versa_p_clear_status(uint8_t api) {
  if (valid_api(api)) {
    statuses[api].done = 0;
    statuses[api].error = 0;
  }
}

void npu_versa_p_global_clear(void) {
  for (size_t i = 0; i < 7; ++i) {
    statuses[i].done = 0;
    statuses[i].error = 0;
    statuses[i].desc0 = 0;
    statuses[i].desc1 = 0;
    statuses[i].desc2 = 0;
    wait_counts[i] = 0;
  }
  qk_gamma_desc = 0;
  for (size_t i = 0; i < kMaxScheduledCommandId; ++i)
    command_apis[i] = UINT8_MAX;
}

uint64_t npu_versa_p_mock_last_desc0(uint8_t api) {
  return valid_api(api) ? statuses[api].desc0 : 0;
}

uint64_t npu_versa_p_mock_last_qk_gamma_desc(void) { return qk_gamma_desc; }

uint32_t npu_versa_p_mock_wait_count(uint8_t api) {
  return valid_api(api) ? wait_counts[api] : 0;
}

void npu_versa_p_mock_inject_error(uint8_t api, uint8_t error_code) {
  if (valid_api(api))
    statuses[api].error = error_code;
}

void npu_versa_p_pack_metadata_i32(const int32_t *bias, const int32_t *scales,
    int32_t *destination, uint32_t channel_count) {
  if (!bias || !scales || !destination)
    return;
  const uint32_t words = (channel_count + 7) / 8;
  const uint32_t scaleBase = words * 8;
  for (uint32_t group = 0; group < words; ++group) {
    for (uint32_t lane = 0; lane < 8; ++lane) {
      const uint32_t channel = group * 8 + lane;
      destination[group * 8 + lane] =
          channel < channel_count ? bias[channel] : 0;
      destination[scaleBase + group * 8 + lane] =
          channel < channel_count ? scales[channel] : 0;
    }
  }
}
