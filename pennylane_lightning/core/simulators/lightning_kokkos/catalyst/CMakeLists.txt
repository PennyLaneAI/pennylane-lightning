cmake_minimum_required(VERSION 3.20)

project(lightning_kokkos_catalyst LANGUAGES CXX)

set(LK_CATALYST_FILES LightningKokkosSimulator.cpp CACHE INTERNAL "")
add_library(lightning_kokkos_catalyst SHARED ${LK_CATALYST_FILES})

include(FetchContent)

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_catalyst.cmake")
FindCatalyst(lightning_kokkos_catalyst)

if(Kokkos_ENABLE_CUDA)
    message(STATUS "Kokkos_ENABLE_CUDA is ON")

    # KOKKOS_CUDA_OPTIONS:
    # - disable_malloc_async: Disable asynchronous memory allocation to ensure
    #   that memory operations complete before proceeding.
    # - enable_lambda: Allow the use of lambda expressions in parallel
    #   execution policies, used in LK kernel functors, and this flag is
    #   required when calling those functors from an external library.
    #   It's supported starting from NVCC v7.0.
    # - enable_constexpr: Enable compile-time evaluations for constants.
    #   There are some undefined behaviour when executing end-to-end Catalyst
    #   programs related to the value of `constexpr`s in the definition of gate kernels.
    #   This flag should be enforced for NVCC v7.5+.
    if(NVCC_VERSION VERSION_LESS "7.0")
        message(WARNING "Building lightning_kokkos_catalyst without lambda and constexpr support.")
        target_compile_definitions(lightning_kokkos_catalyst PRIVATE KOKKOS_CUDA_OPTIONS="disable_malloc_async")
    elseif(NVCC_VERSION VERSION_LESS "7.5")
        message(WARNING "Building lightning_kokkos_catalyst without constexpr support.")
        target_compile_definitions(lightning_kokkos_catalyst PRIVATE KOKKOS_CUDA_OPTIONS="disable_malloc_async enable_lambda")
    else()
        target_compile_definitions(lightning_kokkos_catalyst PRIVATE
            KOKKOS_CUDA_OPTIONS="disable_malloc_async enable_lambda enable_constexpr"
        )
    endif()
endif()

target_include_directories(lightning_kokkos_catalyst PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(lightning_kokkos_catalyst PUBLIC  lightning_compile_options
                                                        lightning_kokkos_algorithms
                                                        lightning_kokkos_measurements
                                                        lightning_catalyst_core
)

set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set_target_properties(lightning_kokkos_catalyst PROPERTIES BUILD_RPATH "${SCIPY_OPENBLAS32_RUNTIME_LIB_PATH}")

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
