cmake_minimum_required(VERSION 3.21)
set(LOGO [=[
░█░░░▀█▀░█▀▀░█░█░▀█▀░█▀█░▀█▀░█▀█░█▀▀░░░░█▀▀░█▀█░█░█
░█░░░░█░░█░█░█▀█░░█░░█░█░░█░░█░█░█░█░░░░█░█░█▀▀░█░█
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░░▀▀▀░▀░░░▀▀▀
]=])
message(${LOGO})

project(lightning_gpu
    DESCRIPTION "Lightning-GPU bindings for PennyLane. Backed by NVIDIA cuQuantum SDK."
    LANGUAGES CXX C CUDA
)

# Include macro and functions supporting Nvidia and cuQuantum libraries.
include("${pennylane_lightning_SOURCE_DIR}/cmake/support_pllgpu.cmake")
findCUDATK(lightning_external_libs)
findCustatevec(lightning_external_libs)

if(ENABLE_MPI)
    findMPI(lightning_external_libs)
endif()

set(LGPU_FILES  StateVectorCudaManaged.cpp initSV.cu
                CACHE INTERNAL "" FORCE)

if(ENABLE_MPI)
    list(APPEND LGPU_FILES StateVectorCudaMPI.cpp)
endif()

add_library(lightning_gpu STATIC ${LGPU_FILES})

target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_PLGPU=1")

if(ENABLE_MPI)
    target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_PLGPU_MPI=1")
endif()

##########################
## Enforce C++ Standard ##
##########################
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#########################
## Set Default Options ##
#########################
option(PL_DISABLE_CUDA_SAFETY "Build without CUDA call safety checks" OFF)

target_link_libraries(lightning_gpu PUBLIC  lightning_compile_options
                                            lightning_external_libs
                                            lightning_base
                                            lightning_gates
                                            lightning_utils
                                            lightning_gpu_utils
                                            lightning_gpu_gates
                                            )
    
target_include_directories(lightning_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET lightning_gpu PROPERTY POSITION_INDEPENDENT_CODE ON)

# To avoid DSO errors on platforms preferring static linkage, uncomment the following line:
# string(REPLACE "libcudart_static.a" "libcudart.so" CUDA_SHARED_RT "${CUDA_LIBRARIES}")
set_target_properties(lightning_gpu PROPERTIES INSTALL_RPATH "$ORIGIN/../cuquantum/lib:$ORIGIN/../cuquantum/lib64:$ORIGIN/")

if(PL_DISABLE_CUDA_SAFETY)
    target_compile_options(lightning_gpu INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-DCUDA_UNSAFE>)
endif()

###############################################################################
# Include all nested sources directories
###############################################################################
set(COMPONENT_SUBDIRS      algorithms
                           bindings
                           measurements
                           gates
                           observables
                           utils
)
foreach(COMP ${COMPONENT_SUBDIRS})
    add_subdirectory(${COMP})
endforeach()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()