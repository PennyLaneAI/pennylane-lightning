cmake_minimum_required(VERSION 3.26)

project(lightning_noise LANGUAGES CXX)

# Header-only helpers of the noisy Kokkos backends (lightning_tensornet):
# dense gate matrices, observable term algebra and the native (compiled)
# noise model format.
add_library(lightning_noise INTERFACE)

target_include_directories(lightning_noise INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(lightning_noise INTERFACE lightning_compile_options
                                                lightning_external_libs
                                                lightning_gates
                                                lightning_utils
                                                )

set_property(TARGET lightning_noise PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS AND PL_NOISY_BACKEND)
    enable_testing()
    add_subdirectory("tests")
endif()
