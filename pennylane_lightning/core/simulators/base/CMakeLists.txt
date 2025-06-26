cmake_minimum_required(VERSION 3.20)

project(lightning_base LANGUAGES CXX)

add_library(lightning_base INTERFACE)


# include_directories(../../../../perfetto/sdk)
# add_library(perfetto STATIC ../../../../perfetto/sdk/perfetto.cc)
# set_property(TARGET perfetto PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(lightning_base INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ../../../../perfetto/sdk)
target_link_libraries(lightning_base INTERFACE  lightning_compile_options
                                                lightning_external_libs
                                                perfetto
                                                )

set_property(TARGET lightning_base PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS AND NOT DEFINED PL_TENSOR)
    enable_testing()
    add_subdirectory("tests")
endif()
