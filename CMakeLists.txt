cmake_minimum_required(VERSION 3.16)

set(LOGO [=[
░█░░░▀█▀░█▀▀░█░█░▀█▀░█▀█░▀█▀░█▀█░█▀▀░░░░▄▀▄░█░█░█▀▄░▀█▀░▀█▀
░█░░░░█░░█░█░█▀█░░█░░█░█░░█░░█░█░█░█░░░░█\█░█░█░█▀▄░░█░░░█░
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░░░▀\░▀▀▀░▀▀░░▀▀▀░░▀░
]=])
message(${LOGO})

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum OS X deployment version")

set(CMAKE_CXX_STANDARD 20) # At least C++20 is required

project(pennylane_lightning
    DESCRIPTION "C++ state-vector simulator bindings for PennyLane. "
    LANGUAGES CXX
)

# Read and set pennylane_lightning version
function(set_pennylane_lightning_version VERSION_FILE_PATH)
    file(STRINGS ${VERSION_FILE_PATH} VERSION_FILE_STR)
    foreach (LINE IN LISTS VERSION_FILE_STR)
    if("${LINE}" MATCHES "__version__.*")
        set(VERSION_LINE_STR "${LINE}")
    endif()
    endforeach()

    string(REGEX REPLACE "__version__ = \"(.*)\"" "\\1" VERSION_STRING ${VERSION_LINE_STR})
    set(VERSION_STRING ${VERSION_STRING} PARENT_SCOPE)
endfunction()

set_pennylane_lightning_version(${PROJECT_SOURCE_DIR}/pennylane_lightning/_version.py)

message(STATUS "pennylane_lightning version ${VERSION_STRING}")
set(PROJECT_VERSION ${VERSION_STRING})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Clang-tidy
option(ENABLE_CLANG_TIDY "Enable clang-tidy build checks" OFF)

# Compile options
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_WARNINGS "Enable warnings" ON)
option(ENABLE_NATIVE "Enable native CPU build tuning" OFF)
option(ENABLE_OPENMP "Enable OpenMP" ON)
option(ENABLE_KOKKOS "Enable Kokkos" ON)
option(ENABLE_BLAS "Enable BLAS" OFF)

# Other build options
option(BUILD_TESTS "Build cpp tests" OFF)
option(BUILD_BENCHMARKS "Enable cpp benchmarks" OFF)

# Process compile options
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/process_options.cmake")

set(CMAKE_POLICY_DEFAULT_CMP0127 NEW) # To suppress pybind11 CMP0127 warning
# Add pybind11
include(FetchContent)
FetchContent_Declare(pybind11
                     GIT_REPOSITORY https://github.com/pybind/pybind11.git
                     GIT_TAG        v2.6.2
)
FetchContent_MakeAvailable(pybind11)


# All CMakeLists.txt in subdirectories use pennylane_lightning_compile_options and pennylane_lightning_external_libs
add_subdirectory(pennylane_lightning/src)

#####################################################
# Maintain for dependent external package development
#####################################################
add_library(pennylane_lightning INTERFACE)

target_link_libraries(pennylane_lightning INTERFACE     lightning_utils
                                                        lightning_simulator
                                                        lightning_algorithms)
target_include_directories(pennylane_lightning INTERFACE "pennylane_lightning/src")

#####################################################

pybind11_add_module(lightning_qubit_ops "pennylane_lightning/src/bindings/Bindings.cpp")
target_link_libraries(lightning_qubit_ops PRIVATE lightning_algorithms
                                                  lightning_gates
                                                  lightning_simulator
                                                  lightning_utils)

target_link_libraries(lightning_qubit_ops PRIVATE lightning_compile_options
                                                  lightning_external_libs)
set_target_properties(lightning_qubit_ops PROPERTIES CXX_VISIBILITY_PRESET hidden)

target_compile_definitions(lightning_qubit_ops PRIVATE VERSION_INFO=${VERSION_STRING})

if (BUILD_TESTS)
    enable_testing()
endif()
