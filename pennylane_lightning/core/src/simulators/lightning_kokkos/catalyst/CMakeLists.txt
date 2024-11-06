cmake_minimum_required(VERSION 3.20)

project(lightning_kokkos_catalyst LANGUAGES CXX)

set(LK_CATALYST_FILES LightningKokkosSimulator.cpp CACHE INTERNAL "")
add_library(lightning_kokkos_catalyst SHARED ${LK_CATALYST_FILES})

include(FetchContent)

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_catalyst.cmake")
FindCatalyst(lightning_kokkos_catalyst)

target_include_directories(lightning_kokkos_catalyst INTERFACE  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_kokkos_catalyst PUBLIC  lightning_compile_options
                                                        lightning_external_libs
                                                        lightning_base
                                                        lightning_gates
                                                        lightning_utils
                                                        lightning_kokkos
                                                        lightning_kokkos_algorithms
                                                        lightning_kokkos_gates
                                                        lightning_kokkos_measurements
                                                        lightning_kokkos_utils
)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
