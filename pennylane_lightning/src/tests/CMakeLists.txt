cmake_minimum_required(VERSION 3.14)

project(pennylane_lightning_tests)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenMP)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug)
endif()


option(ENABLE_NATIVE "Enable native CPU build tuning" OFF)


include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git 
  GIT_TAG release-1.11.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

add_executable(cpptests lightning_apply_unittest.cpp
                        lightning_gates_unittest.cpp
                        lightning_util_unittest.cpp
                        "../Apply.cpp"
                        "../Gates.cpp")

target_link_libraries(cpptests PRIVATE gtest_main)
if(OpenMP_CXX_FOUND)
	target_link_libraries(cpptests PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(cpptests PRIVATE "$<$<CONFIG:DEBUG>:-Wall>")

if(ENABLE_NATIVE)
	message(STATUS "ENABLE_NATIVE is ON. Use -march=native for cpptests.")
	target_compile_options(cpptests PRIVATE -march=native)
endif()


include(GoogleTest)
gtest_discover_tests(cpptests)
