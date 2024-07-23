project(lightning_gpu_utils LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

findCUDATK(lightning_compile_options)

add_library(lightning_gpu_utils INTERFACE)
target_include_directories(lightning_gpu_utils INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(lightning_gpu_utils INTERFACE CUDA::cudart CUDA::cublas CUDA::cusparse lightning_compile_options)
