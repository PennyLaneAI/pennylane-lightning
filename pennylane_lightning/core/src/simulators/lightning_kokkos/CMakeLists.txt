cmake_minimum_required(VERSION 3.20)
set(LOGO [=[
░█░░░▀█▀░█▀▀░█░█░▀█▀░█▀█░▀█▀░█▀█░█▀▀░░░░█░█░█▀█░█░█░█░█░█▀█░█▀▀
░█░░░░█░░█░█░█▀█░░█░░█░█░░█░░█░█░█░█░░░░█▀▄░█░█░█▀▄░█▀▄░█░█░▀▀█
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░░▀░▀░▀▀▀░▀░▀░▀░▀░▀▀▀░▀▀▀
]=])
message(${LOGO})

project(lightning_kokkos
    DESCRIPTION "PennyLane Lightning Kokkos C++ Backend."
    LANGUAGES CXX C
)

set(LKOKKOS_FILES   StateVectorKokkos.cpp
                    CACHE INTERNAL "" FORCE)

##########################
## Enforce C++ Standard ##
##########################
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Ensure the libraries can see additional libs at same level;
# Required for external deps when loading in Python
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)

#########################
## Set Default Options ##
#########################
option(PLKOKKOS_ENABLE_NATIVE "Enable native CPU build tuning" OFF)
option(PLKOKKOS_ENABLE_SANITIZER "Enable address sanitizer" OFF)
option(PLKOKKOS_ENABLE_WARNINGS "Enable warnings" ON)

# Include macro and functions supporting Kokkos libraries.
include("${pennylane_lightning_SOURCE_DIR}/cmake/support_kokkos.cmake")
FindKokkos(lightning_external_libs)

if(PLKOKKOS_ENABLE_SANITIZER)
    add_compile_options(-fsanitize=address)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_link_options(-fsanitize=address)
    else ()
        add_link_options(-fsanitize=address -static-libasan)
    endif()
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
ENDIF()

target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_PLKOKKOS=1")

# Compile the sources files to a target object:
add_library(obj_file OBJECT ${LKOKKOS_FILES})
set_property(TARGET obj_file PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(obj_file PUBLIC   lightning_compile_options
                                        lightning_external_libs
                                        lightning_base
                                        lightning_gates
                                        lightning_utils
                                        lightning_kokkos_utils
                                        lightning_kokkos_gates
)

if(PLKOKKOS_ENABLE_NATIVE)
    target_compile_options(obj_file  PRIVATE -march=native)
endif()

# Build THE SHARED and STATIC libraries from the target object:
add_library(lightning_kokkos_shared SHARED $<TARGET_OBJECTS:obj_file>)
add_library(lightning_kokkos STATIC $<TARGET_OBJECTS:obj_file>)

set(LKOKKOS_LIBS    lightning_kokkos
                    lightning_kokkos_shared
)

foreach(LK_LIB ${LKOKKOS_LIBS})
    target_link_libraries(${LK_LIB} PUBLIC  lightning_external_libs
                                            lightning_base
                                            lightning_gates
                                            lightning_utils
                                            lightning_kokkos_utils
                                            lightning_kokkos_gates
    )
    target_include_directories(${LK_LIB}  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    set_property(TARGET ${LK_LIB}  PROPERTY POSITION_INDEPENDENT_CODE ON)
endforeach()

###############################################################################
# Include subdirectories
###############################################################################
set(COMPONENT_SUBDIRS   algorithms
                        bindings
                        gates
                        measurements
                        observables
                        utils
)

foreach(COMP ${COMPONENT_SUBDIRS})
    add_subdirectory(${COMP})
endforeach()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
