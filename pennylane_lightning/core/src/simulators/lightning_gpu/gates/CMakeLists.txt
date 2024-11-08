project(lightning_gpu_gates)

# We make a static library for default kernels using explicit template instantiations.
# We thus ensure that only a single instantiation exists during the build process.

add_library(lightning_gpu_gates INTERFACE)
target_include_directories(lightning_gpu_gates INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(lightning_gpu_gates INTERFACE  lightning_gpu_utils lightning_utils lightning_external_libs)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
