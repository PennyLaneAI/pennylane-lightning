cmake_minimum_required(VERSION 3.20)

set(LOGO [=[
░█░░░▀█▀░█▀▀░█░█░▀█▀░█▀█░▀█▀░█▀█░█▀▀░░░░▀█▀░█▀▀░█▀█░█▀▀░█▀█░█▀▄
░█░░░░█░░█░█░█▀█░░█░░█░█░░█░░█░█░█░█░░░░░█░░█▀▀░█░█░▀▀█░█░█░█▀▄
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀░▀
]=])
message(${LOGO})

project(${PL_BACKEND}
    DESCRIPTION "Lightning-Tensor MPS bindings for PennyLane. Backed by NVIDIA cuQuantum SDK."
    LANGUAGES CXX C
)

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_pllgpu.cmake")
include("${pennylane_lightning_SOURCE_DIR}/cmake/support_pltensortncuda.cmake")
findCUDATK(lightning_external_libs)
findCutensornet(lightning_external_libs)

set(LTENSOR_MPS_FILES  MPSTNCuda.cpp CACHE INTERNAL "" FORCE)

add_library(${PL_BACKEND} STATIC ${LTENSOR_MPS_FILES})

target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_PLTENSOR=1")

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

target_link_libraries(${PL_BACKEND} PUBLIC  lightning_compile_options
                                            lightning_external_libs
                                            lightning_utils
                                            ${PL_BACKEND}_gates
                                            ${PL_BACKEND}_tensor
                                            ${PL_BACKEND}_tensornetBase
                                            ${PL_BACKEND}_utils
                                            )
    
target_include_directories(${PL_BACKEND} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET ${PL_BACKEND} PROPERTY POSITION_INDEPENDENT_CODE ON)

# To avoid DSO errors on platforms preferring static linkage, uncomment the following line:
# string(REPLACE "libcudart_static.a" "libcudart.so" CUDA_SHARED_RT "${CUDA_LIBRARIES}")
set_target_properties(${PL_BACKEND} PROPERTIES INSTALL_RPATH "$ORIGIN/../cuquantum/lib:$ORIGIN/../cuquantum/lib64:$ORIGIN/")

if(PL_DISABLE_CUDA_SAFETY)
    target_compile_options(${PL_BACKEND} INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-DCUDA_UNSAFE>)
endif()

###############################################################################
# Include all nested sources directories
###############################################################################
set(COMPONENT_SUBDIRS      base
                           bindings
                           gates
                           measurements
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
