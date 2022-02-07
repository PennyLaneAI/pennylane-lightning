project(lightning_gates)
set(CMAKE_CXX_STANDARD 17)

set(SIMULATOR_FILES GateUtil.cpp CACHE INTERNAL "" FORCE)

add_library(lightning_gates STATIC ${SIMULATOR_FILES})
target_link_libraries(lightning_gates PRIVATE pennylane_lightning_compile_options
                                              pennylane_lightning_external_libs)
target_include_directories(lightning_gates PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_gates PRIVATE lightning_utils)

set_property(TARGET lightning_gates PROPERTY POSITION_INDEPENDENT_CODE ON)
