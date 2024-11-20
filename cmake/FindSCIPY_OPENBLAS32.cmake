###############################################################################################
# This file provides macros to process CUDA, CUDAToolkit, cuQuantum and MPI external libraries.
###############################################################################################

# Include this only once
include_guard()

macro(get_scipy_openblas_libname SCIPY_OPENBLAS_LIB_NAME)
    set(LIB_SUFFIX ".so")
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(LIB_SUFFIX ".dylib")
    elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
        set(LIB_SUFFIX ".dll")
    endif()

    set(${SCIPY_OPENBLAS_LIB_NAME} "libscipy_openblas${LIB_SUFFIX}")
endmacro()

macro(find_path_to_openblas SCIPY_OPENBLASE_LIB_PATH)
    set(SCIPY_OPENBLAS_LIB_NAME "")
    get_scipy_openblas_libname(SCIPY_OPENBLAS_LIB_NAME)

    find_file(SCIPY_OPENBLAS_LIB_FILE
        NAMES   ${SCIPY_OPENBLAS_LIB_NAME}
        HINTS   ${SCIPY_OPENBLAS}/lib
                $ENV{SCIPY_OPENBLAS32}/lib
                ${Python_SITELIB}/scipy_openblas32/lib
                ${SCIPY_OPENBLAS}
                $ENV{SCIPY_OPENBLAS32}
                ENV LD_LIBRARY_PATH
    )

    if(NOT SCIPY_OPENBLAS_LIB_FILE)
        message(FATAL_ERROR "\nUnable to find ${SCIPY_OPENBLAS_LIB_NAME}. Please ensure it is correctly installed and available on path.")
    else()
        cmake_path(GET SCIPY_OPENBLAS_LIB_FILE PARENT_PATH SCIPY_OPENBLAS_LIB_DIR)
        set(${SCIPY_OPENBLASE_LIB_PATH} ${SCIPY_OPENBLAS_LIB_DIR})
    endif()
endmacro()

# Macro to aid in finding cuStateVec lib
macro(get_scipy_openblas external_libs)
    set(SCIPY_OPENBLAS_LIB_NAME "")
    get_scipy_openblas_libname(SCIPY_OPENBLAS_LIB_NAME)

    find_library(SCIPY_OPENBLAS_LIB
        NAMES   ${SCIPY_OPENBLAS_LIB_NAME}
        HINTS   ${SCIPY_OPENBLAS}/lib
                $ENV{SCIPY_OPENBLAS32}/lib
                ${Python_SITELIB}/scipy_openblas32/lib
                ${SCIPY_OPENBLAS}
                $ENV{SCIPY_OPENBLAS32}
                ENV LD_LIBRARY_PATH
    )

    if(NOT SCIPY_OPENBLAS_LIB)
        message(FATAL_ERROR "\nUnable to find ${SCIPY_OPENBLAS}. Please ensure it is correctly installed and available on path.")
    else()
        cmake_path(GET SCIPY_OPENBLAS_LIB PARENT_PATH SCIPY_OPENBLAS_LIB_DIR)
        set(${SCIPY_OPENBLASE_LIB_PATH} ${SCIPY_OPENBLAS_LIB_DIR})
    endif()
endmacro()
