##############################################################################
# This file processes ENABLE_WARNINGS, ENABLE_NATIVE, ENABLE_OPENMP, 
# ENABLE_KOKKOS, and ENABLE_BLAS 
# options and produces interface libraries
# lightning_compile_options and lightning_external_libs.
##############################################################################

# Include this file only once
include_guard()

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

    find_package(Kokkos
    HINTS   ${CMAKE_SOURCE_DIR}/Kokkos
            /usr
            /usr/local
            /opt
    )
    if(Kokkos_FOUND)
        message(STATUS "Found existing Kokkos library")
    endif()

    find_package(KokkosKernels
    HINTS   ${CMAKE_SOURCE_DIR}/Kokkos
            ${CMAKE_SOURCE_DIR}/KokkosKernels
            /usr
            /usr/local
            /opt
    )
    if(KokkosKernels_FOUND)
        message(STATUS "Found existing Kokkos Kernels library")
    endif()

    if (NOT (Kokkos_FOUND AND KokkosKernels_FOUND))
        # Setting the Serial device.
        option(Kokkos_ENABLE_SERIAL  "Enable Kokkos SERIAL device" ON)
        message(STATUS "KOKKOS SERIAL DEVICE ENABLED.")

        option(Kokkos_ENABLE_COMPLEX_ALIGN "Enable complex alignment in memory" OFF)

        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        include(FetchContent)

        FetchContent_Declare(kokkos
                            GIT_REPOSITORY https://github.com/kokkos/kokkos.git
                            GIT_TAG        3.6.00
                            GIT_SUBMODULES "" # Avoid recursively cloning all submodules
        )
    
        FetchContent_MakeAvailable(kokkos)

        get_target_property(kokkos_INC_DIR kokkos INTERFACE_INCLUDE_DIRECTORIES)
        set_target_properties(kokkos PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${kokkos_INC_DIR}")

        FetchContent_Declare(kokkoskernels
                            GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git
                            GIT_TAG        3.6.00
                            GIT_SUBMODULES "" # Avoid recursively cloning all submodules
        )
    
        FetchContent_MakeAvailable(kokkoskernels)
    
        get_target_property(kokkoskernels_INC_DIR kokkoskernels INTERFACE_INCLUDE_DIRECTORIES)
        set_target_properties(kokkoskernels PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${kokkoskernels_INC_DIR}")

    endif()
    target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_KOKKOS=1")
    target_link_libraries(lightning_external_libs INTERFACE Kokkos::kokkos Kokkos::kokkoskernels)
else()
    message(STATUS "ENABLE_KOKKOS is OFF.")
endif()

if (UNIX AND (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(AMD64)|(X64)|(x64)|(x86_64)"))
    message(STATUS "ENABLE AVX for X64 on UNIX compatible system.")
    target_compile_options(lightning_compile_options INTERFACE -mavx)
endif()