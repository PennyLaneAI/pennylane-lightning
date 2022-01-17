project(lightning_simulator)
set(CMAKE_CXX_STANDARD 17)

set(SIMULATOR_FILES IndicesUtil.cpp CACHE INTERNAL "" FORCE)

add_library(lightning_simulator STATIC ${SIMULATOR_FILES})
target_link_libraries(lightning_simulator PRIVATE pennylane_lightning_compile_options
                                                   pennylane_lightning_external_libs)
target_include_directories(lightning_simulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_simulator PRIVATE lightning_utils)

set_property(TARGET lightning_simulator PROPERTY POSITION_INDEPENDENT_CODE ON)
