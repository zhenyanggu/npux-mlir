// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iostream>

#include "src/Runtime/VersaPRuntime.h"

int main() {
  uint8_t hostByte = 0;
  uint32_t hostBase = 0;
  if (npu_versa_p_translate_dram_address(&hostByte, &hostBase) != 0 ||
      npu_versa_p_submit_host(NPU_VERSA_P_API_MVIN_W,
          0x1234000000000000ULL, 0, 0, &hostByte) != 0 ||
      npu_versa_p_mock_last_desc0(NPU_VERSA_P_API_MVIN_W) !=
          (0x1234000000000000ULL | hostBase)) {
    std::cerr << "VersaPRuntimeTest: host submit/address patch failed\\n";
    return 1;
  }
 uint32_t dramBase = 0;
  if (npu_versa_p_translate_dram_address(&hostByte, &dramBase) != 0 ||
      dramBase !=
          static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&hostByte)) ||
      npu_versa_p_translate_dram_address(nullptr, &dramBase) != 12) {
    std::cerr << "VersaPRuntimeTest: address translation contract failed\n";
    return 1;
  }

  npu_versa_p_global_clear();
  if (npu_versa_p_submit(NPU_VERSA_P_API_MVIN_A,
          0x1234000000000000ULL, 0x55, 0x66, 0x2000) != 0 ||
      npu_versa_p_mock_last_desc0(NPU_VERSA_P_API_MVIN_A) !=
          0x1234000000002000ULL ||
      npu_versa_p_wait(NPU_VERSA_P_API_MVIN_A) != 0) {
    std::cerr << "VersaPRuntimeTest: submit/address patch failed\n";
    return 1;
  }
  if (npu_versa_p_submit_wait_static(NPU_VERSA_P_API_SA,
          UINT64_C(0x1122334455667788), 2, 3) != 0 ||
      npu_versa_p_mock_last_desc0(NPU_VERSA_P_API_SA) !=
          UINT64_C(0x1122334455667788)) {
    std::cerr << "VersaPRuntimeTest: static descriptor changed\\n";
    return 1;
  }

  npu_versa_p_mock_inject_error(NPU_VERSA_P_API_SA, 4);
  if (npu_versa_p_wait(NPU_VERSA_P_API_SA) != 4 ||
      npu_versa_p_submit(NPU_VERSA_P_API_SA, 0, 0, 0, 0) != 4) {
    std::cerr << "VersaPRuntimeTest: error stop failed\n";
    return 1;
  }
  npu_versa_p_clear_status(NPU_VERSA_P_API_SA);
  if (npu_versa_p_wait(NPU_VERSA_P_API_SA) != 0) {
    std::cerr << "VersaPRuntimeTest: W1C clear failed\n";
    return 1;
  }
  npu_versa_p_global_clear();
  if (npu_versa_p_mock_last_desc0(NPU_VERSA_P_API_MVIN_A) != 0) {
    std::cerr << "VersaPRuntimeTest: global clear failed\n";
    return 1;
  }
  return 0;
}