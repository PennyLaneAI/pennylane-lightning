project(lightning_simulator)
set(CMAKE_CXX_STANDARD 17)

set(SIMULATOR_FILES DynamicDispatcher.cpp CACHE INTERNAL "" FORCE)

add_library(lightning_simulator STATIC ${SIMULATOR_FILES})

target_include_directories(lightning_simulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_simulator PRIVATE lightning_compile_options
                                                  lightning_external_libs
                                                  lightning_gates
                                                  lightning_utils)

set_property(TARGET lightning_simulator PROPERTY POSITION_INDEPENDENT_CODE ON)
