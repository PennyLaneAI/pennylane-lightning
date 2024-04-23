project(cuda_utils LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

findCUDATK(lightning_compile_options)

add_library(cuda_utils INTERFACE)
target_include_directories(cuda_utils INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(cuda_utils INTERFACE CUDA::cudart CUDA::cublas CUDA::cusparse lightning_compile_options)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
