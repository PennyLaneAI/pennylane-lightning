cmake_minimum_required(VERSION 3.26)

project(lightning_cliffish_catalyst LANGUAGES CXX)

set(LC_CATALYST_FILES LightningCliffishSimulator.cpp CACHE INTERNAL "")
add_library(lightning_cliffish_catalyst SHARED ${LC_CATALYST_FILES})

include(FetchContent)

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_catalyst.cmake")
FindCatalyst(lightning_cliffish_catalyst)

target_include_directories(lightning_cliffish_catalyst PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(lightning_cliffish_catalyst PUBLIC lightning_compile_options
                                                         lightning_cliffish
                                                         lightning_catalyst_core
)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
