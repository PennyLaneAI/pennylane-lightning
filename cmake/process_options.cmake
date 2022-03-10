##############################################################################
# This file processes ENABLE_WARNINGS, ENABLE_NATIVE, ENABLE_AVX, 
# ENABLE_OPENMP, ENABLE_BLAS options and produces interface libraries
# lightning_compile_options and lightning_external_libs.
##############################################################################

# Include this file only once
include_guard()

# Set compile flags and library dependencies
add_library(lightning_compile_options INTERFACE)
add_library(lightning_external_libs INTERFACE)

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

if(ENABLE_WARNINGS)
    message(STATUS "ENABLE_WARNINGS is ON.")
    if(MSVC)
        target_compile_options(lightning_compile_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4;/WX>)
    else()
        target_compile_options(lightning_compile_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>)
    endif()
else()
    message(STATUS "ENABLE_WARNINGS is OFF")
endif()

if(ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Using -march=native.")
    target_compile_options(lightning_compile_options INTERFACE -march=native)
endif()

if(ENABLE_AVX)
    message(STATUS "ENABLE_AVX is ON.")
    target_compile_options(lightning_compile_options INTERFACE -mavx)
else()
    message(STATUS "ENABLE_AVX is OFF")
endif()

if(ENABLE_AVX2)
    message(STATUS "ENABLE_AVX2 is ON.")
    target_compile_options(lightning_compile_options INTERFACE -mavx2)
else()
    message(STATUS "ENABLE_AVX2 is OFF")
endif()

if(ENABLE_AVX512)
    message(STATUS "ENABLE_AVX512 is ON.")
    target_compile_options(lightning_compile_options INTERFACE -mavx512f) # Now we only use avx512f
else()
    message(STATUS "ENABLE_AVX512 is OFF")
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
    message(STATUS "ENABLE_OPENMP is OFF")
endif()

if(ENABLE_BLAS)
    message(STATUS "ENABLE_BLAS is ON. Find BLAS.")
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
    message(STATUS "ENABLE_BLAS is OFF")
endif()
