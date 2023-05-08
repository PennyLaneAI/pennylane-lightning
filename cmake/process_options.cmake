##############################################################################
# This file processes ENABLE_WARNINGS, ENABLE_NATIVE, ENABLE_OPENMP,
# ENABLE_KOKKOS, and ENABLE_BLAS
# options and produces interface libraries
# lightning_compile_options and lightning_external_libs.
##############################################################################

# Include this file only once
include_guard()

##############################################################################

# Macro to aid in finding Kokkos with 3 potential install options:
# 1. Fully integrated Kokkos packages and CMake module files
# 2. Statically compiled libraries and headers
# 3. Not installed, so fall back to building from source.

macro(FindKokkos target_name)
    find_package(Kokkos
    HINTS   ${CMAKE_SOURCE_DIR}/kokkos
            ${CMAKE_SOURCE_DIR}/Kokkos
            ${Kokkos_Core_DIR}
            /usr
            /usr/local
            /opt
            /opt/Kokkos
    )

    find_package(KokkosKernels
    HINTS   ${CMAKE_SOURCE_DIR}/kokkos
            ${CMAKE_SOURCE_DIR}/Kokkos
            ${CMAKE_SOURCE_DIR}/kokkosKernels
            ${CMAKE_SOURCE_DIR}/KokkosKernels
            ${Kokkos_Kernels_DIR}
            /usr
            /usr/local
            /opt
            /opt/KokkosKernels
    )
    if(Kokkos_FOUND AND KokkosKernels_FOUND)
        message(STATUS "Found existing Kokkos libraries")
        target_link_libraries(${target_name} INTERFACE Kokkos::kokkos Kokkos::kokkoskernels)
        return()
    else()
        message(STATUS "Could not find existing Kokkos package. Searching for precompiled libraries and headers")

        find_library(Kokkos_core_lib
            NAME kokkoscore.a libkokkoscore.a kokkoscore.so libkokkoscore.so
            HINTS   ${CMAKE_SOURCE_DIR}/Kokkos/lib
                    ${Kokkos_Core_DIR}/lib
                    ${Kokkos_Core_DIR}/lib64
                    /usr/lib
                    /usr/lib64
                    /usr/local/lib
                    /usr/local/lib64
                    ENV LD_LIBRARY_PATH
        )
        find_library(Kokkos_Kernels_lib
            NAME kokkoskernels.a libkokkoskernels.a kokkoskernels.so libkokkoskernels.so
            HINTS   ${CMAKE_SOURCE_DIR}/Kokkos/lib
                    ${Kokkos_Kernels_DIR}/lib
                    ${Kokkos_Kernels_DIR}/lib64
                    /usr/lib
                    /usr/lib64
                    /usr/local/lib
                    /usr/local/lib64
                    ENV LD_LIBRARY_PATH
        )
        find_file(  Kokkos_core_inc
            NAMES   Kokkos_Core.hpp
            HINTS   ${Kokkos_Core_DIR}/include
                    /usr/include
                    /usr/local/include
                    ENV CPATH
        )
        find_file(  Kokkos_sparse_inc
            NAMES   KokkosSparse.hpp
            HINTS   ${Kokkos_Kernels_DIR}/include
                    /usr/include
                    /usr/local/include
                    ENV CPATH
        )
        if (Kokkos_core_lib_FOUND AND Kokkos_Kernels_lib_FOUND)
            message(STATUS "Found existing Kokkos compiled libraries")

            add_library( kokkos SHARED IMPORTED GLOBAL)
            add_library( kokkoskernels SHARED IMPORTED GLOBAL)

            cmake_path(GET Kokkos_core_inc ROOT_PATH Kokkos_INC_DIR)
            cmake_path(GET Kokkos_sparse_inc ROOT_PATH KokkosKernels_INC_DIR)

            set_target_properties( kokkos PROPERTIES IMPORTED_LOCATION ${Kokkos_core_lib})
            set_target_properties( kokkoskernels PROPERTIES IMPORTED_LOCATION ${Kokkos_Kernels_lib})
            set_target_properties( kokkos PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${Kokkos_INC_DIR}")
            set_target_properties( kokkoskernels PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${KokkosKernels_INC_DIR}")

            target_link_libraries(${target_name} PRIVATE kokkos kokkoskernels)
            return()
        else()
            message(STATUS "Building Kokkos from source. SERIAL device enabled.")

            option(Kokkos_ENABLE_SERIAL  "Enable Kokkos SERIAL device" ON)
            option(Kokkos_ENABLE_COMPLEX_ALIGN "Enable complex alignment in memory" OFF)

            set(CMAKE_POSITION_INDEPENDENT_CODE ON)
            include(FetchContent)

            FetchContent_Declare(kokkos
                                GIT_REPOSITORY https://github.com/kokkos/kokkos.git
                                GIT_TAG        4.0.01
                                GIT_SUBMODULES "" # Avoid recursively cloning all submodules
            )

            FetchContent_MakeAvailable(kokkos)

            get_target_property(kokkos_INC_DIR kokkos INTERFACE_INCLUDE_DIRECTORIES)
            set_target_properties(kokkos PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${kokkos_INC_DIR}")

            FetchContent_Declare(kokkoskernels
                                GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git
                                GIT_TAG        4.0.01
                                GIT_SUBMODULES "" # Avoid recursively cloning all submodules
            )

            FetchContent_MakeAvailable(kokkoskernels)

            get_target_property(kokkoskernels_INC_DIR kokkoskernels INTERFACE_INCLUDE_DIRECTORIES)
            set_target_properties(kokkoskernels PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${kokkoskernels_INC_DIR}")
            target_link_libraries(${target_name} INTERFACE kokkos kokkoskernels)
        endif()
    endif()
endmacro()

##############################################################################


if (WIN32)
    # Increasing maximum full-path length allowed.
  message("Setting default path length to 240 characters")
  set(CMAKE_OBJECT_PATH_MAX 240)
endif ()

# Check GCC version
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
        message(FATAL_ERROR "GCC version must be at least 10.0")
    endif()
endif()

# Set compile flags and library dependencies
add_library(lightning_compile_options INTERFACE)
add_library(lightning_external_libs INTERFACE)

# We use C++20 experimentally. As we still set CMAKE_CXX_STANDARD, the following line is not essential.
# It will be uncommented when we move to a newer set-up.
# target_compile_features(lightning_compile_options INTERFACE cxx_std_20)

# Initial attempt to find which BLAS implementation is chosen
function(get_blas_impl)
    string(FIND "${BLAS_LIBRARIES}" "mkl" FOUND_MKL)
    string(FIND "${BLAS_LIBRARIES}" "openblas" FOUND_OPENBLAS)

    if (NOT (FOUND_MKL EQUAL -1)) # MKL is found
        set(BLAS_IMPL "MKL" PARENT_SCOPE)
    elseif (NOT (FOUND_OPENBLAS EQUAL -1))
        set(BLAS_IMPL "OpenBLAS" PARENT_SCOPE)
    else()
        set(BLAS_IMPL "Unknown" PARENT_SCOPE)
    endif()
endfunction()

if(MSVC) # For M_PI
    target_compile_options(lightning_compile_options INTERFACE /D_USE_MATH_DEFINES)
endif()

# Add -fwrapv, -fno-plt in Clang
if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"))
    target_compile_options(lightning_compile_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-fwrapv;-fno-plt>)
# Add -fwrapv, -fno-plt, -pipe in GCC
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(lightning_compile_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-fwrapv;-fno-plt;-pipe>)
endif()

if(ENABLE_COVERAGE)
    message(STATUS "ENABLE_COVERAGE is ON.")
    target_compile_options(lightning_compile_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-fprofile-arcs;-ftest-coverage>)
    target_link_libraries(lightning_external_libs INTERFACE gcov)
endif()

if(ENABLE_WARNINGS)
    message(STATUS "ENABLE_WARNINGS is ON.")
    if(MSVC)
        target_compile_options(lightning_compile_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4;/WX>)
    else()
        target_compile_options(lightning_compile_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>)
    endif()
else()
    message(STATUS "ENABLE_WARNINGS is OFF.")
endif()

if(ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Using -march=native.")
    target_compile_options(lightning_compile_options INTERFACE -march=native)
endif()

if(ENABLE_OPENMP)
    message(STATUS "ENABLE_OPENMP is ON.")
    find_package(OpenMP)

    if(NOT OpenMP_CXX_FOUND)
        message(FATAL_ERROR "OpenMP is enabled but not found.\n"
            "Install OpenMP or set ENABLE_OPENMP OFF.")
    endif()

    target_link_libraries(lightning_external_libs INTERFACE OpenMP::OpenMP_CXX)
else()
    message(STATUS "ENABLE_OPENMP is OFF.")
endif()

if(ENABLE_BLAS)
    message(STATUS "ENABLE_BLAS is ON.")
    find_package(BLAS)

    if(NOT BLAS_FOUND)
        message(FATAL_ERROR "BLAS is enabled but not found.")
    endif()

    get_blas_impl()
    message(STATUS "Use ${BLAS_IMPL} for BLAS implementation. Set BLA_VENDOR variable "
                   "if you want to use a different BLAS implementation. "
                   "See https://cmake.org/cmake/help/latest/module/FindBLAS.html"
                   "#blas-lapack-vendors for available options.")

    target_link_libraries(lightning_external_libs INTERFACE "${BLAS_LIBRARIES}")
    target_link_options(lightning_external_libs INTERFACE "${BLAS_LINKER_FLAGS}")
    target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_BLAS=1")
else()
    message(STATUS "ENABLE_BLAS is OFF.")
endif()

if(ENABLE_KOKKOS)
    message(STATUS "ENABLE_KOKKOS is ON.")
    target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_KOKKOS=1")
    FindKokkos(lightning_external_libs)
else()
    message(STATUS "ENABLE_KOKKOS is OFF.")
endif()

if (UNIX AND (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(AMD64)|(X64)|(x64)|(x86_64)"))
    message(STATUS "ENABLE AVX for X64 on UNIX compatible system.")
    target_compile_options(lightning_compile_options INTERFACE -mavx)
endif()