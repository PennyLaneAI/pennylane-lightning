project(lightning_algorithms LANGUAGES CXX)

set(ALGORITHM_FILES AdjointDiff.cpp JacobianTape.cpp StateVecAdjDiff.cpp CACHE INTERNAL "" FORCE)
add_library(lightning_algorithms STATIC ${ALGORITHM_FILES})

target_link_libraries(lightning_algorithms PUBLIC lightning_compile_options
                                                   lightning_external_libs
                                                   lightning_gates
                                                   lightning_simulator
                                                   lightning_utils)

target_include_directories(lightning_algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET lightning_algorithms PROPERTY POSITION_INDEPENDENT_CODE ON)
