# SPDX-License-Identifier: Apache-2.0

if(OMP_SUPPORTED)
  set(OMP_TOPDIR ${CMAKE_CURRENT_BINARY_DIR})
  set_directory_properties(PROPERTIES EP_BASE ${OMP_TOPDIR})
  set(OPENMP_SOURCE_DIR ${LLVM_BUILD_MAIN_SRC_DIR}/../openmp)
  find_file(SYSTEM_GOMP_LIBRARY
    NAMES libgomp.so.1 libgomp.so.1.0.0
    PATHS
      /usr/lib/x86_64-linux-gnu
      /usr/lib64
      /usr/lib
      /lib/x86_64-linux-gnu
      /lib64
      /lib
  )

  include(ExternalProject)
  if(EXISTS "${OPENMP_SOURCE_DIR}/CMakeLists.txt")
    ExternalProject_Add(OMomp
      DOWNLOAD_DIR "."
      BINARY_DIR ${OMP_TOPDIR}/openmp-build
      INSTALL_DIR ${OMP_TOPDIR}/install
      STAMP_DIR ${OMP_TOPDIR}/stamp
      TMP_DIR ${OMP_TOPDIR}/tmp

      DOWNLOAD_COMMAND ""
      # Build LLVM's shared libomp so host-sim model.so can dlopen cleanly and
      # resolve the LLVM OpenMP ABI symbols (__kmpc_*).
      CONFIGURE_COMMAND sh -c "CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} \
                               cmake -G Ninja -S ${OPENMP_SOURCE_DIR} \
                                  -B ${OMP_TOPDIR}/openmp-build \
                                  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
                                  -DLIBOMP_ENABLE_SHARED=ON \
                                  -DCMAKE_POSITION_INDEPENDENT_CODE=ON"

      BUILD_COMMAND sh -c "cmake --build ${OMP_TOPDIR}/openmp-build --target omp"
      INSTALL_COMMAND ""
    )

    add_custom_target(ompruntime
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${OMP_TOPDIR}/openmp-build/runtime/src/libomp.so ${ONNX_MLIR_LIBRARY_PATH}/libompruntime.so
      DEPENDS OMomp
      BYPRODUCTS ${OMP_TOPDIR}/openmp-build/runtime/src/libomp.so
    )

    install(FILES ${ONNX_MLIR_LIBRARY_PATH}/libompruntime.so DESTINATION lib)
    message(STATUS "OpenMP support           : ON (llvm libomp shared)")
  elseif(SYSTEM_GOMP_LIBRARY)
    add_custom_target(ompruntime
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              ${SYSTEM_GOMP_LIBRARY} ${ONNX_MLIR_LIBRARY_PATH}/libompruntime.so
      BYPRODUCTS ${ONNX_MLIR_LIBRARY_PATH}/libompruntime.so
    )
    install(FILES ${SYSTEM_GOMP_LIBRARY} DESTINATION lib RENAME libompruntime.so)
    message(STATUS "OpenMP support           : ON (system libgomp fallback)")
  else()
    message(STATUS "OpenMP support           : OFF (no LLVM OpenMP source and no fallback runtime)")
  endif()

else()
  message(STATUS "OpenMP support           : OFF")
endif()
