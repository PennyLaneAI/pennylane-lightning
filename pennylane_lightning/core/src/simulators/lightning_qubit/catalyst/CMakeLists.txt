cmake_minimum_required(VERSION 3.20)

project(lightning_qubit_catalyst LANGUAGES CXX)

set(LQ_CATALYST_FILES LightningSimulator.cpp CACHE INTERNAL "")
add_library(lightning_qubit_catalyst SHARED ${LQ_CATALYST_FILES})

include(FetchContent)

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_catalyst.cmake")
FindCatalyst(lightning_qubit_catalyst)

target_include_directories(lightning_qubit_catalyst INTERFACE  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_qubit_catalyst PUBLIC  lightning_compile_options
                                                       lightning_external_libs
                                                       lightning_base
                                                       lightning_gates
                                                       lightning_utils
                                                       lightning_qubit
                                                       lightning_qubit_algorithms
                                                       lightning_qubit_gates
                                                       lightning_qubit_measurements
                                                       lightning_qubit_observables
                                                       lightning_qubit_utils
)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
