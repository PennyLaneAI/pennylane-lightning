###############################################################################################
# This file provides macros to process CUDA, CUDAToolkit, cuQuantum and MPI external libraries.
###############################################################################################

# Include this only once
include_guard()

# Macro to aid in finding CUDA lib
macro(findCUDATK external_libs)
    find_package(CUDAToolkit REQUIRED EXACT 12)
    if(CUDA_FOUND)
	    target_link_libraries(${external_libs} INTERFACE CUDA::cudart CUDA::cublas CUDA::cusparse)
    endif()
endmacro()

# Macro to aid in finding MPI lib
macro(findMPI external_libs)
    if(ENABLE_MPI)
        find_package(MPI REQUIRED)
        if(MPI_FOUND)
            message(STATUS "MPI found.")
        else()
            message(FATAL_ERROR "MPI is NOT found.")
        endif()

        find_library(GTL
        NAMES libmpi_gtl_cuda.so libmpi_gtl_cuda.so.0 libmpi_gtl_cuda.so.0.0.0
        HINTS ENV CRAY_LD_LIBRARY_PATH
        ENV LD_LIBRARY_PATH
        )

        string(FIND "${CMAKE_SYSTEM}" "cray" subStrIdx)
        if(NOT subStrIdx EQUAL -1)
            if(DEFINED ENV{CRAY_MPICH_VERSION} AND NOT "$ENV{CRAY_MPICH_VERSION}" STREQUAL "none")
                if(NOT GTL)
                    message(FATAL_ERROR "\nUnable to find GTL (GPU Transport Layer) library.")
                else()
                    message(STATUS "GPU Transport Layer library found.")
                    target_link_libraries(${external_libs} INTERFACE ${GTL})
                endif()
            endif()
        endif()

        target_link_libraries(${external_libs} INTERFACE MPI::MPI_CXX)
    endif()
endmacro()

# Macro to aid in finding cuStateVec lib
macro(findCustatevec external_libs)
    set(CUQUANTUM_ENV "$ENV{CUQUANTUM_SDK}")
    find_library(CUSTATEVEC_LIB
        NAMES   libcustatevec.so.1 custatevec.so.1
        HINTS   /usr/lib
            /usr/local/cuda
            /usr/local/lib
            /opt
            /opt/cuda
            lib
            lib64
            ${CUQUANTUM_SDK}/lib
            ${CUQUANTUM_SDK}/lib64
            ${CUQUANTUM_ENV}/lib
            ${CUQUANTUM_ENV}/lib64
            ${CUDAToolkit_LIBRARY_DIR}
            ${CUDA_TOOLKIT_ROOT_DIR}/lib
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64
            ${Python_SITELIB}/cuquantum/lib
            ENV LD_LIBRARY_PATH
    )

    find_file( CUSTATEVEC_INC
        NAMES   custatevec.h
        HINTS   /usr/include
            /usr/local/cuda
            /usr/local/include
            /opt
            /opt/cuda
            include
            ${CUQUANTUM_SDK}/include
            ${CUQUANTUM_ENV}/include
            ${CUDAToolkit_INCLUDE_DIRS}
            ${CUDA_TOOLKIT_ROOT_DIR}/include
            ${Python_SITELIB}/cuquantum/include
            ENV CPATH
    )

    if(NOT CUSTATEVEC_LIB OR NOT CUSTATEVEC_INC)
        message(FATAL_ERROR "\nUnable to find cuQuantum SDK installation. Please ensure it is correctly installed and available on path.")
    else()
        add_library( custatevec SHARED IMPORTED GLOBAL)

        get_filename_component(CUSTATEVEC_INC_DIR ${CUSTATEVEC_INC} DIRECTORY)
        target_include_directories(custatevec INTERFACE ${CUSTATEVEC_INC_DIR})
        set_target_properties( custatevec PROPERTIES IMPORTED_LOCATION ${CUSTATEVEC_LIB})

        target_link_libraries(${external_libs} INTERFACE custatevec)
    endif()
endmacro()
