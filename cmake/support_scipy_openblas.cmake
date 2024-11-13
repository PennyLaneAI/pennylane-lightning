###############################################################################################
# This file provides macros to process CUDA, CUDAToolkit, cuQuantum and MPI external libraries.
###############################################################################################

# Include this only once
include_guard()

# Macro to aid in finding cuStateVec lib
macro(get_scipy_openblas SCIPY_OPENBLASE_LIB_PATH)
    set(SCIPY_OPENBLAS_ENV "$ENV{SCIPY_OPENBLAS}")

    set(SCIPY_OPENBLAS_LIB_NAME "libscipy_openblas.so")

    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(SCIPY_OPENBLAS_LIB_NAME "libLAPACK.dylib")
    elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
        set(SCIPY_OPENBLAS_LIB_NAME "libscipy_openblas.dll")
    endif()

    set(ACCELEARTE_FRAMEWORK "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework")

    find_library(SCIPY_OPENBLAS_LIB
        NAMES   ${SCIPY_OPENBLAS_LIB_NAME}
        HINTS   ${SCIPY_OPENBLAS}/lib
                ${SCIPY_OPENBLAS_ENV}/lib
                ${Python_SITELIB}/scipy_openblas32/lib
                ${SCIPY_OPENBLAS}
                ${ACCELEARTE_FRAMEWORK}
                ENV LD_LIBRARY_PATH
    )

    if(NOT SCIPY_OPENBLAS_LIB)
        message(FATAL_ERROR "\nUnable to find ${SCIPY_OPENBLAS}. Please ensure it is correctly installed and available on path.")
        message(FATAL_ERROR "\nUnable to find scipy_openblas64. Please ensure it is correctly installed and available on path.")
    else()
        cmake_path(GET SCIPY_OPENBLAS_LIB PARENT_PATH SCIPY_OPENBLAS_LIB_DIR)
        set(${SCIPY_OPENBLASE_LIB_PATH} ${SCIPY_OPENBLAS_LIB_DIR})
    endif()
endmacro()
