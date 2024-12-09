cmake_minimum_required(VERSION 3.20)

project(lightning_tensor LANGUAGES CXX)

###############################################################################
# Include all nested sources directories
###############################################################################
set(COMPONENT_SUBDIRS   utils 
                        tensor)

if("${PL_TENSOR_BACKEND}" STREQUAL "cutensornet")
    list(APPEND COMPONENT_SUBDIRS tncuda)
else()
    message(FATAL_ERROR "Lightning Tensor currently doesn't support ${PL_TENSOR_BACKEND} method")
endif()

foreach(COMP ${COMPONENT_SUBDIRS})
    add_subdirectory(${COMP})
endforeach()
