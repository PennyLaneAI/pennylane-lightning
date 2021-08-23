project(lightning_algorithms LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(ALGORITHM_FILES AdjointDiff.hpp AdjointDiff.cpp CACHE INTERNAL "" FORCE)
add_library(lightning_algorithms STATIC ${ALGORITHM_FILES})

target_include_directories(lightning_algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(lightning_algorithms PRIVATE lightning_simulator lightning_utils)

set_property(TARGET lightning_algorithms PROPERTY POSITION_INDEPENDENT_CODE ON)

if(ENABLE_NATIVE)
    target_compile_options(lightning_algorithms PRIVATE -march=native)
endif()