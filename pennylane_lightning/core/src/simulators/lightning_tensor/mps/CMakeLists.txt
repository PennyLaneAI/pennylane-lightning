cmake_minimum_required(VERSION 3.21)

project(lightning_tensor_mps LANGUAGES CXX)

set(COMPONENT_SUBDIRS base utils)

if("${PL_TENSOR_BACKEND}" STREQUAL "cutensornet")
    list(APPEND COMPONENT_SUBDIRS cutn)
else()
    message(FATAL_ERROR "Lightning Tensor currently doesn't support ${PL_TENSOR_BACKEND} backend")
endif()

foreach(COMP ${COMPONENT_SUBDIRS})
    add_subdirectory(${COMP})
endforeach()
