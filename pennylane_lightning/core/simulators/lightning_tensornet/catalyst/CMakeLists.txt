cmake_minimum_required(VERSION 3.26)

project(lightning_tensornet_catalyst LANGUAGES CXX)

set(LTENSORNET_CATALYST_FILES LightningTensornetSimulator.cpp CACHE INTERNAL "")
add_library(lightning_tensornet_catalyst SHARED ${LTENSORNET_CATALYST_FILES})

include(FetchContent)

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_catalyst.cmake")
FindCatalyst(lightning_tensornet_catalyst)

if(Kokkos_ENABLE_CUDA)
    # Same Kokkos CUDA options as lightning_kokkos_catalyst (see that file).
    if(NVCC_VERSION VERSION_LESS "7.0")
        target_compile_definitions(lightning_tensornet_catalyst PRIVATE KOKKOS_CUDA_OPTIONS="disable_malloc_async")
    elseif(NVCC_VERSION VERSION_LESS "7.5")
        target_compile_definitions(lightning_tensornet_catalyst PRIVATE KOKKOS_CUDA_OPTIONS="disable_malloc_async enable_lambda")
    else()
        target_compile_definitions(lightning_tensornet_catalyst PRIVATE
            KOKKOS_CUDA_OPTIONS="disable_malloc_async enable_lambda enable_constexpr"
        )
    endif()
endif()

target_include_directories(lightning_tensornet_catalyst PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${pennylane_lightning_SOURCE_DIR}/pennylane_lightning/core/noise/catalyst
)
target_link_libraries(lightning_tensornet_catalyst PUBLIC lightning_compile_options
                                                      lightning_tensornet
                                                      lightning_noise
                                                      lightning_catalyst_core
)

set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set_target_properties(lightning_tensornet_catalyst PROPERTIES BUILD_RPATH "${SCIPY_OPENBLAS32_RUNTIME_LIB_PATH}")

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
