cmake_minimum_required(VERSION 3.26)

set(LOGO [=[
░█░░░▀█▀░█▀▀░█░█░▀█▀░█▀█░▀█▀░█▀█░█▀▀░░░░▀█▀░█▀▀░█▀█░█▀▀░█▀█░█▀▄░█▀█░█▀▀░▀█▀
░█░░░░█░░█░█░█▀█░░█░░█░█░░█░░█░█░█░█░░░░░█░░█▀▀░█░█░▀▀█░█░█░█▀▄░█░█░█▀▀░░█░
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀░▀░▀░▀░▀▀▀░░▀░
]=])
message(${LOGO})

project(lightning_tensornet
    DESCRIPTION "PennyLane Lightning Tensornet: Kokkos matrix-product-state backend."
    LANGUAGES CXX C
)

# The MPS backend only needs Kokkos, the Lightning utilities and the shared
# noise helpers. The lightning_kokkos targets (added by
# support_simulators.cmake before this directory, without their Catalyst
# plugin and tests) provide the Kokkos installation.
if(NOT TARGET lightning_kokkos)
    message(FATAL_ERROR "lightning_tensornet requires the lightning_kokkos targets.")
endif()

add_library(lightning_tensornet INTERFACE)
target_include_directories(lightning_tensornet INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_tensornet INTERFACE lightning_compile_options
                                                    lightning_external_libs
                                                    lightning_kokkos
                                                    lightning_noise
                                                    )
target_compile_definitions(lightning_tensornet INTERFACE "_ENABLE_PLTENSORNET=1")

# Component targets expected by the top-level CMakeLists.txt
foreach(COMP observables measurements algorithms bindings)
    add_library(lightning_tensornet_${COMP} INTERFACE)
    target_link_libraries(lightning_tensornet_${COMP} INTERFACE lightning_tensornet)
endforeach()
target_include_directories(lightning_tensornet_bindings INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/bindings
                                                        ${pennylane_lightning_SOURCE_DIR}/pennylane_lightning/core/noise/bindings)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/catalyst/CMakeLists.txt")
    add_subdirectory(catalyst)
endif()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
