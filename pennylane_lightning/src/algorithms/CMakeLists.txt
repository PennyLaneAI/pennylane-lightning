project(lightning_algorithms LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_library(lightning_algorithms INTERFACE)
target_include_directories(lightning_algorithms INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)