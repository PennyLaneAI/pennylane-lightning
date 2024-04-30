cmake_minimum_required(VERSION 3.21)

project(lightning_tensor LANGUAGES CXX)

###############################################################################
# Include all nested sources directories
###############################################################################
set(COMPONENT_SUBDIRS   utils 
                        tensor)

if("${PL_TENSOR_METHOD}" STREQUAL "mps")
    list(APPEND COMPONENT_SUBDIRS mps)
else()
    message(FATAL_ERROR "Lightning Tensor currently doesn't support ${PL_TENSOR_METHOD} method")
endif()

foreach(COMP ${COMPONENT_SUBDIRS})
    add_subdirectory(${COMP})
endforeach()
