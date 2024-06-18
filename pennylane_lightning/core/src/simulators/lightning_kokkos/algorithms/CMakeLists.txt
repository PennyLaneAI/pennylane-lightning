cmake_minimum_required(VERSION 3.20)

project(lightning_kokkos_algorithms LANGUAGES CXX)

set(ALGORITHMS_FILES    AlgorithmsKokkos.cpp
                        CACHE INTERNAL "" FORCE)

# Compile the sources files to a target object:
add_library(LKokkos_algorithm_obj OBJECT ${ALGORITHMS_FILES})
set_property(TARGET LKokkos_algorithm_obj PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(LKokkos_algorithm_obj PRIVATE lightning_compile_options
                                                    lightning_external_libs
)

target_link_libraries(LKokkos_algorithm_obj PUBLIC  lightning_kokkos_utils
                                                    lightning_algorithms
                                                    lightning_kokkos
                                                    lightning_kokkos_observables
)

# Build THE SHARED and STATIC libraries from the target object:
add_library(lightning_kokkos_algorithms_shared SHARED $<TARGET_OBJECTS:LKokkos_algorithm_obj>)
add_library(lightning_kokkos_algorithms STATIC $<TARGET_OBJECTS:LKokkos_algorithm_obj>)

set(LKOKKOS_LIBS    lightning_kokkos_algorithms
                    lightning_kokkos_algorithms_shared
)

foreach(LK_LIB ${LKOKKOS_LIBS})
    target_link_libraries(${LK_LIB} PRIVATE lightning_external_libs)
    target_link_libraries(${LK_LIB} PUBLIC  lightning_kokkos_utils
                                            lightning_algorithms
                                            lightning_kokkos
                                            lightning_kokkos_observables
    )
    target_include_directories(${LK_LIB}  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    set_property(TARGET ${LK_LIB}  PROPERTY POSITION_INDEPENDENT_CODE ON)
endforeach()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()