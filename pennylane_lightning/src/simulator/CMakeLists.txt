project(lightning_simulator)

set(SIMULATOR_FILES StateVectorRawCPU.cpp Observables.cpp StateVectorManagedCPU.cpp Measures.cpp CACHE INTERNAL "" FORCE)

add_library(lightning_simulator STATIC ${SIMULATOR_FILES})

if (ENABLE_GATE_DISPATCHER AND UNIX AND (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(AMD64)|(X64)|(x64)|(x86_64)"))
    target_sources(lightning_simulator PRIVATE KernelMap_X64.cpp AssignKernelMap_AVX2.cpp AssignKernelMap_AVX512.cpp AssignKernelMap_Default.cpp)
else()
    target_sources(lightning_simulator PRIVATE KernelMap_Default.cpp AssignKernelMap_Default.cpp)
endif()

target_include_directories(lightning_simulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_simulator PRIVATE lightning_compile_options
                                                  lightning_external_libs
                                                  lightning_gates
                                                  lightning_utils)
set_property(TARGET lightning_simulator PROPERTY POSITION_INDEPENDENT_CODE ON)
