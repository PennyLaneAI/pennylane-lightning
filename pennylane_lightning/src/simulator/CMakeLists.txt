project(lightning_simulator)
set(CMAKE_CXX_STANDARD 17)

set(SIMULATOR_FILES StateVector.cpp StateVector.hpp Gates.hpp CACHE INTERNAL "" FORCE)
add_library(lightning_simulator STATIC ${SIMULATOR_FILES})

find_package(OpenMP)

target_include_directories(lightning_simulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(lightning_simulator PRIVATE lightning_utils)

if(OpenMP_FOUND)
    target_link_libraries(lightning_simulator PRIVATE OpenMP::OpenMP_CXX)
endif()

set_property(TARGET lightning_simulator PROPERTY POSITION_INDEPENDENT_CODE ON)

if(ENABLE_NATIVE)
    target_compile_options(lightning_simulator PRIVATE -march=native)
endif()