##########################
## Set Project version
##########################
cmake_minimum_required(VERSION 3.14)
set(LIGHTNING_LOGO "
░█░░░▀█▀░█▀▀░█░█░▀█▀░█▀█░▀█▀░█▀█░█▀▀░
░█░░░░█░░█░█░█▀█░░█░░█░█░░█░░█░█░█░█░
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀▀░
")
message(${LIGHTNING_LOGO})

project(pennylane_lightning
    DESCRIPTION "A C++ backed statevector device for PennyLane"
    LANGUAGES CXX C
)

##########################
## Utility methods
##########################

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

##########################
## Enfore C++ Standard
##########################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

##########################
## Set Default Options
##########################
# Compiler options
option(ENABLE_NATIVE "Enable native CPU build tuning" OFF)
option(BUILD_TESTS "Build cpp tests" OFF)
option(ENABLE_WARNINGS "Enable warnings" ON)
option(ENABLE_CLANG_TIDY "Enable clang-tidy build checks" OFF)
option(ENABLE_CUDA "Build with CUDA GPU support" OFF)
option(DISABLE_CUDA_SAFETY "Build without CUDA call safety checks" OFF)

# Build options
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(ENABLE_CLANG_TIDY)
    if (NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy)
    endif()
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
                            -extra-arg=-std=c++17;
    )
endif()

##########################
## Fetch dependencies
##########################
# Add pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.6.2
)
FetchContent_MakeAvailable(pybind11)

if(ENABLE_CUDA)
    SET(CUDA_SEPARABLE_COMPILATION ON)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60)
    endif()

    find_package(CUDA REQUIRED)
endif()

############################
## Create Lightning bindings
############################
add_subdirectory(pennylane_lightning/src)
add_library(pennylane_lightning INTERFACE)
target_link_libraries(pennylane_lightning INTERFACE     lightning_utils 
                                                        lightning_simulator 
                                                        lightning_algorithms)
target_include_directories(pennylane_lightning INTERFACE "pennylane_lightning/src")

# Create binding module
add_library(external_dependency INTERFACE)
pybind11_add_module(lightning_qubit_ops     "pennylane_lightning/src/bindings/Bindings.cpp")
target_link_libraries(lightning_qubit_ops PRIVATE pennylane_lightning external_dependency)
set_target_properties(lightning_qubit_ops PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_options(lightning_qubit_ops PRIVATE "$<$<CONFIG:RELEASE>:-W>")
target_compile_definitions(lightning_qubit_ops PRIVATE VERSION_INFO=${VERSION_STRING})

##########################
## Compile options
##########################

if(ENABLE_WARNINGS)
    target_compile_options(pennylane_lightning INTERFACE 
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
    )
endif()

if(ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Using -march=native")
    add_compile_options(-march=native)
    target_compile_options(pennylane_lightning INTERFACE -march=native)
    target_compile_options(lightning_qubit_ops PRIVATE -march=native)
endif()

if (ENABLE_CUDA)
    # Avoid DSO errors on platforms preferring static linkage
    string(REPLACE "libcudart_static.a" "libcudart.so" CUDA_SHARED_RT "${CUDA_LIBRARIES}")
    target_include_directories(pennylane_lightning INTERFACE ${CUDA_TOOLKIT_ROOT_DIR}/include)
    target_link_libraries(pennylane_lightning INTERFACE ${CUDA_SHARED_RT})
    if(DISABLE_CUDA_SAFETY)
        target_compile_options(pennylane_lightning INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-DCUDA_UNSAFE>)
    endif()
endif()

if (BUILD_TESTS)
    enable_testing()
endif()
