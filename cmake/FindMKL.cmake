###########################################################
# Adapted from PRACE course materials at:
# https://repository.prace-ri.eu/git/CodeVault/hpc-kernels/spectral_methods/-/blob/master/cmake/Modules/FindMKL.cmake
###########################################################

# This script looks for MKL in two locations:
# - The environment variable MKLROOT, which is defined by
#    sourcing an MKL environment
# - The directory `/opt/intel/mkl`, which is a common
#    install location for MKL.
# It may be possible to install MKL using python -m pip,
# though this is not guaranteed to be found, and may
# require explicitly setting the MKLROOT variable.

###########################################################
# Stage 1: find the root directory
###########################################################

set(MKLROOT_PATH $ENV{MKLROOT})
if (NOT MKLROOT_PATH)
    if ($ENV{ONEAPI_ROOT})
        set(MKLROOT_PATH $ENV{ONEAPI_ROOT}/mkl/latest)
    endif ()
endif ()
if (NOT MKLROOT_PATH)
    foreach(P IN ITEMS "/opt/intel/oneapi/mkl/latest" "/opt/intel/mkl")
        if (EXISTS "${P}")
            set(MKLROOT_PATH "${P}")
            message(STATUS "Found MKL: ${P}")
            break()
        endif ()
    endforeach()
endif ()

###########################################################
# Stage 2: find include path and libraries
###########################################################

if (MKLROOT_PATH)

    set(EXPECT_MKL_INCPATH "${MKLROOT_PATH}/include")
    set(EXPECT_ICC_LIBPATH "$ENV{ICC_LIBPATH}")

    # MacOS will have a different path structure
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(EXPECT_MKL_LIBPATH "${MKLROOT_PATH}/lib")
    elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
        set(EXPECT_MKL_LIBPATH "${MKLROOT_PATH}/lib/intel64")
    endif()

    ###########################################################
    # Set MKL_INCLUDE and MKL_LIBRARY_DIR
    ###########################################################

    if (IS_DIRECTORY ${EXPECT_MKL_INCPATH})
        set(MKL_INCLUDE_DIR ${EXPECT_MKL_INCPATH})
    endif (IS_DIRECTORY ${EXPECT_MKL_INCPATH})

    if (IS_DIRECTORY ${EXPECT_MKL_LIBPATH})
        set(MKL_LIBRARY_DIR ${EXPECT_MKL_LIBPATH})
    endif (IS_DIRECTORY ${EXPECT_MKL_LIBPATH})

    ###########################################################
    # find specific library files
    ###########################################################

    find_library(LIB_MKL_RT NAMES mkl_rt mkl_rt.1 HINTS ${MKL_LIBRARY_DIR})
    find_library(LIB_PTHREAD NAMES pthread)

endif (MKLROOT_PATH)

set(MKL_LIBRARY "${LIB_MKL_RT};${LIB_PTHREAD}")

###########################################################
# deal with QUIET and REQUIRED argument
###########################################################

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(MKL DEFAULT_MSG
                                      MKL_LIBRARY_DIR
                                      LIB_MKL_RT
                                      LIB_PTHREAD
                                      MKL_INCLUDE_DIR)

mark_as_advanced(LIB_MKL_RT LIB_PTHREAD MKL_INCLUDE_DIR)