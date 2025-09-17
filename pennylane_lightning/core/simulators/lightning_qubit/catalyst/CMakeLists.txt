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
                                                       lightning_qubit_algorithms
                                                       lightning_qubit_measurements
                                                       lightning_qubit_observables
                                                       lightning_catalyst_core
)

set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set_target_properties(lightning_qubit_catalyst PROPERTIES BUILD_RPATH "${SCIPY_OPENBLAS32_RUNTIME_LIB_PATH}")

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
