###############################################################################################
# This file provides macros to process cuTensorNet external libraries.
###############################################################################################

# Include this only once
include_guard()

# Macro to aid in finding cuTensorNet lib
macro(findCutensornet external_libs)
    find_library(CUTENSORNET_LIB
        NAMES   libcutensornet.so.2 libcutensornet.so
        HINTS   /usr/lib
            /usr/local/cuda
            /usr/local/lib
            /opt
            /opt/cuda
            lib
            lib64
            ${CUQUANTUM_SDK}/lib
            ${CUQUANTUM_SDK}/lib64
            ${CUDAToolkit_LIBRARY_DIR}
            ${CUDA_TOOLKIT_ROOT_DIR}/lib
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64
            ${Python_SITELIB}/cuquantum/lib
            ENV LD_LIBRARY_PATH
    )

    find_file( CUTENSORNET_INC
        NAMES   cutensornet.h
        HINTS   /usr/include
            /usr/local/cuda
            /usr/local/include
            /opt
            /opt/cuda
            include
            ${CUQUANTUM_SDK}/include
            ${CUDAToolkit_INCLUDE_DIRS}
            ${CUDA_TOOLKIT_ROOT_DIR}/include
            ${Python_SITELIB}/cuquantum/include
            ENV CPATH
    )

    if(NOT CUTENSORNET_LIB OR NOT CUTENSORNET_INC)
        message(FATAL_ERROR "\nUnable to find cuQuantum SDK installation. Please ensure it is correctly installed and available on path.")
    else()
        add_library(cutensornet SHARED IMPORTED GLOBAL)

        get_filename_component(CUTENSORNET_INC_DIR ${CUTENSORNET_INC} DIRECTORY)
        target_include_directories(cutensornet INTERFACE ${CUTENSORNET_INC_DIR})
        set_target_properties(cutensornet PROPERTIES IMPORTED_LOCATION ${CUTENSORNET_LIB})

        target_link_libraries(${external_libs} INTERFACE cutensornet)
    endif()
endmacro()
