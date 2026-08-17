// SPDX-License-Identifier: Apache-2.0

#include "src/Runtime/VersaPRuntime.h"

#include <stddef.h>
#include <stdlib.h>

typedef struct {
  uint64_t desc0;
  uint64_t desc1;
  uint64_t desc2;
  uint8_t done;
  uint8_t error;
} VersaPStatus;

static VersaPStatus statuses[5];

static int valid_api(uint8_t api) { return api < 5; }

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
  return statuses[api].error;
}

void npu_versa_p_clear_status(uint8_t api) {
  if (valid_api(api)) {
    statuses[api].done = 0;
    statuses[api].error = 0;
  }
}

void npu_versa_p_global_clear(void) {
  for (size_t i = 0; i < 5; ++i) {
    statuses[i].done = 0;
    statuses[i].error = 0;
    statuses[i].desc0 = 0;
    statuses[i].desc1 = 0;
    statuses[i].desc2 = 0;
  }
}

uint64_t npu_versa_p_mock_last_desc0(uint8_t api) {
  return valid_api(api) ? statuses[api].desc0 : 0;
}

void npu_versa_p_mock_inject_error(uint8_t api, uint8_t error_code) {
  if (valid_api(api))
    statuses[api].error = error_code;
}