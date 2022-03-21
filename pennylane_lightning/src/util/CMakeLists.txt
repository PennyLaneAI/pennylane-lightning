project(lightning_utils LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(UTIL_FILES RuntimeInfo.cpp CACHE INTERNAL "" FORCE)

add_library(lightning_utils STATIC ${UTIL_FILES})
target_include_directories(lightning_utils INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/DS>
)

set_property(TARGET lightning_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
