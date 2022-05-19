project(lightning_utils LANGUAGES CXX)

set(UTIL_FILES RuntimeInfo.cpp CACHE INTERNAL "" FORCE)

add_library(lightning_utils STATIC ${UTIL_FILES})

target_link_libraries(lightning_utils PRIVATE lightning_compile_options
                                              lightning_external_libs)
target_include_directories(lightning_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_property(TARGET lightning_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
