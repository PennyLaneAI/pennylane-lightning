cmake_minimum_required(VERSION 3.20)

project(TNCuda_observables LANGUAGES CXX)

set(OBSERVABLES_FILES ObservablesTNCuda.cpp CACHE INTERNAL "" FORCE)

add_library(TNCuda_observables STATIC ${OBSERVABLES_FILES})

target_include_directories(TNCuda_observables INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(TNCuda_observables PRIVATE   lightning_compile_options
                                                            lightning_external_libs
                                                            )

target_link_libraries(TNCuda_observables PUBLIC  lightning_utils
                                                        lightning_gates
                                                        lightning_observables
                                                        ${PL_BACKEND}_utils
                                                        ${PL_TENSOR}_gates
                                                        ${PL_TENSOR}
                                                        )

set_property(TARGET TNCuda_observables PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
