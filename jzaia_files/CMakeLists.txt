project(lightning_demo LANGUAGES CXX)
add_executable(lq_grover main.cpp)
target_include_directories(lq_grover PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lq_grover lightning_qubit
                                lightning_qubit_algorithms
                                lightning_qubit_observables
                                )
