cmake_minimum_required(VERSION 3.14)

project(pennylane_lightning_tests)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenMP)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
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

include(GoogleTest)
gtest_discover_tests(cpptests)
