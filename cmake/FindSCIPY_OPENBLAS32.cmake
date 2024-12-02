###############################################################################################
# This file provides macros to process scipy-openblas32 external libraries.
###############################################################################################

# Include this only once
include_guard()

macro(find_path_to_openblas SCIPY_OPENBLASE_LIB_PATH)
    set(SCIPY_OPENBLAS_LIB_NAME "libscipy_openblas${CMAKE_SHARED_LIBRARY_SUFFIX}")

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
        message(FATAL_ERROR "\nUnable to find ${SCIPY_OPENBLAS_LIB_NAME}. Please set a SCIPY_OPENBLAS32 env variable to provide the path to scipy_openblas32.")
    else()
        cmake_path(GET SCIPY_OPENBLAS_LIB_FILE PARENT_PATH SCIPY_OPENBLAS_LIB_DIR)
        set(${SCIPY_OPENBLASE_LIB_PATH} ${SCIPY_OPENBLAS_LIB_DIR})
    endif()
endmacro()

