##############################################################################
# This file processes options:
# ENABLE_WARNINGS, ENABLE_NATIVE, ENABLE_OPENMP
# and produces interface libraries:
# lightning_compile_options and lightning_external_libs.
##############################################################################

# Include this file only once
include_guard()

option(PLLGPU_DISABLE_CUDA_SAFETY "Build without CUDA call safety checks" OFF)

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

if(ENABLE_CLANG_TIDY)
    if(NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy)
    endif()
    message(STATUS "Using CLANG_TIDY_BINARY=${CLANG_TIDY_BINARY}")
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
                             -extra-arg=-std=c++20;
    )
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


if(PLLGPU_DISABLE_CUDA_SAFETY)
    target_compile_options(lightning_compile_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-DCUDA_UNSAFE>)
endif()

if(ENABLE_OPENMP)
    message(STATUS "ENABLE_OPENMP is ON.")
    find_package(OpenMP)
    if(NOT OpenMP_CXX_FOUND)
        message(FATAL_ERROR "OpenMP is enabled but not found.\n"
            "Install OpenMP or set ENABLE_OPENMP OFF.")
    endif()
    target_compile_options(lightning_compile_options INTERFACE "-DPL_USE_OMP=1")
    target_link_libraries(lightning_external_libs INTERFACE OpenMP::OpenMP_CXX)
else()
    message(STATUS "ENABLE_OPENMP is OFF.")
endif()

if (UNIX AND (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(AMD64)|(X64)|(x64)|(x86_64)"))
    message(STATUS "ENABLE AVX for X64 on UNIX compatible system.")
    target_compile_options(lightning_compile_options INTERFACE -mavx)
endif()


if(ENABLE_LAPACK)
    if(MSVC)
        message(FATAL_ERROR "LAPACK is not supported for Windows.")
    endif()

    #LAPACK NOT FOUND
    include(ExternalProject)

    ExternalProject_Add(LAPACK
        GIT_REPOSITORY https://github.com/Reference-LAPACK/lapack.git
        GIT_TAG        "master"
        GIT_SUBMODULES ""
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/lapack -DLAPACKE=ON -DBUILD_SHARED_LIBS=ON
    )

    find_library(LAPACK_LIB
                NAMES   liblapack.so
                HINTS   ${CMAKE_CURRENT_BINARY_DIR}/lapack/lib64
                        ${CMAKE_CURRENT_BINARY_DIR}/lapack/lib
    )

    find_library(LAPACKE_LIB
                NAMES   liblapacke.so
                HINTS   ${CMAKE_CURRENT_BINARY_DIR}/lapack/lib64
                        ${CMAKE_CURRENT_BINARY_DIR}/lapack/lib
    )

    find_library(BLAS_LIB
                NAMES   libblas.so
                HINTS   ${CMAKE_CURRENT_BINARY_DIR}/lapack/lib64
                        ${CMAKE_CURRENT_BINARY_DIR}/lapack/lib
    )

    find_file(LAPACKE_INC
                NAMES   lapacke.h
                HINTS   $ENV{CRAY_LIBSCI_PREFIX_DIR}/include
    )

    add_library(lapack SHARED IMPORTED GLOBAL)
    add_library(lapacke SHARED IMPORTED GLOBAL)
    add_library(blas SHARED IMPORTED GLOBAL)

    #
    #        find_file(SCI_INC
    #            NAMES   lapacke.h
    #            HINTS   $ENV{CRAY_LIBSCI_PREFIX_DIR}/include
    #        )

    #target_include_directories(lapack INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/lapack/include)
    #target_include_directories(lapacke INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/lapack/include)

    #set_target_properties(lapack PROPERTIES
    #    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/lapack/lib64/liblapacke.so
    #)

    #set_target_properties(lapacke PROPERTIES
    #    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/lapack/lib64/liblapacke.so
    #)

    #set_target_properties(blas PROPERTIES
    #    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/lapack/lib64/libblas.so
    #)

    get_filename_component(LAPACKE_INC_DIR ${LAPACKE_INC} DIRECTORY)
    target_include_directories(lapacke INTERFACE ${LAPACKE_INC_DIR})
    set_target_properties(lapacke PROPERTIES IMPORTED_LOCATION ${LAPACKE_LIB})


    get_filename_component(LAPACK_INC_DIR ${LAPACKE_INC} DIRECTORY)
    target_include_directories(lapack INTERFACE ${LAPACK_INC_DIR})
    set_target_properties(lapack PROPERTIES IMPORTED_LOCATION ${LAPACK_LIB})

    get_filename_component(BLAS_INC_DIR ${LAPACKE_INC} DIRECTORY)
    target_include_directories(blas INTERFACE ${BLAS_INC_DIR})
    set_target_properties(blas PROPERTIES IMPORTED_LOCATION ${BLAS_LIB})

    target_link_libraries(lightning_external_libs INTERFACE lapack lapacke blas)

    target_compile_options(lightning_compile_options INTERFACE "-DPL_USE_LAPACK=1")
    
    #find_package(LAPACK REQUIRED)
    #if(LAPACK_FOUND)
    #    message(STATUS "LAPACK found.")
    #    string(FIND "${CMAKE_SYSTEM}" "cray" subStrIdx)
    #    if(NOT subStrIdx EQUAL -1)
    #        set(CRAY_LIBSCI_INC "$ENV{CRAY_LIBSCI_PREFIX_DIR}/include")
    #        set(CRAY_LIBSCI_LIB "$ENV{CRAY_LIBSCI_PREFIX_DIR}/lib")
    #
    #        find_library(SCI_LIB
    #            NAMES   libsci_gnu.so
    #            HINTS   $ENV{CRAY_LIBSCI_PREFIX_DIR}/lib
    #        )
    #
    #        find_file(SCI_INC
    #            NAMES   lapacke.h
    #            HINTS   $ENV{CRAY_LIBSCI_PREFIX_DIR}/include
    #        )
    #
    #        if(NOT SCI_LIB OR NOT SCI_INC)
    #            message(FATAL_ERROR "\nUnable to find libsci on CrayLinux. Please ensure it is correctly installed and available on path.")
    #        else()
    #            add_library(sci SHARED IMPORTED GLOBAL)
    #            get_filename_component(SCI_INC_DIR ${SCI_INC} DIRECTORY)
    #            target_include_directories(sci INTERFACE ${SCI_INC_DIR})
    #            set_target_properties(sci PROPERTIES IMPORTED_LOCATION ${SCI_LIB})
    #            target_link_libraries(lightning_external_libs INTERFACE sci)
    #        endif()
    #    else()
    #        target_link_libraries(lightning_external_libs INTERFACE LAPACK::LAPACK)
    #    endif()
    #    target_compile_options(lightning_compile_options INTERFACE "-DPL_USE_LAPACK=1")
    #else()
    #    message(STATUS "LAPACK is enabled but not found.\n")
    #endif()
    
endif()