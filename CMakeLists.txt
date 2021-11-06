cmake_minimum_required(VERSION 3.14)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING "Minimum OS X deployment version")

project(pennylane_lightning)

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

set(CMAKE_CXX_STANDARD 17) # At least C++17 is required

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Clang-tidy
option(ENABLE_CLANG_TIDY "Enable clang-tidy build checks" OFF)

# Compile options
option(ENABLE_WARNINGS "Enable warnings" ON)
option(ENABLE_NATIVE "Enable native CPU build tuning" OFF)
option(ENABLE_OPENMP "Enable OpenMP" ON)
option(ENABLE_BLAS "Enable BLAS" ON)

# Other build options
option(BUILD_TESTS "Build cpp tests" OFF)

if(ENABLE_CLANG_TIDY)
    if(NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy)
    endif()
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
                            -extra-arg=-std=c++17;
    )
endif()

# Add pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.6.2
)
FetchContent_MakeAvailable(pybind11)

add_subdirectory(pennylane_lightning/src)

add_library(pennylane_lightning INTERFACE)
target_link_libraries(pennylane_lightning INTERFACE     lightning_utils 
                                                        lightning_simulator 
                                                        lightning_algorithms)
target_include_directories(pennylane_lightning INTERFACE "pennylane_lightning/src")

#add_library(external_dependency INTERFACE)

pybind11_add_module(lightning_qubit_ops "pennylane_lightning/src/bindings/Bindings.cpp")
target_link_libraries(lightning_qubit_ops PRIVATE pennylane_lightning) #external_dependency)
set_target_properties(lightning_qubit_ops PROPERTIES CXX_VISIBILITY_PRESET hidden)

target_compile_options(lightning_qubit_ops PRIVATE "$<$<CONFIG:RELEASE>:-W>")
target_compile_definitions(lightning_qubit_ops PRIVATE VERSION_INFO=${VERSION_STRING})

if(ENABLE_WARNINGS)
  target_compile_options(pennylane_lightning INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>)
endif()


if(ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Using -march=native.")
    add_compile_options(-march=native)
    target_compile_options(pennylane_lightning INTERFACE -march=native)
    target_compile_options(lightning_qubit_ops PRIVATE -march=native)
endif()

if(ENABLE_OPENMP)
	message(STATUS "ENABLE_OPENMP is ON. Using OpenMP.")
	find_package(OpenMP)

	if(NOT OpenMP_CXX_FOUND)
		message(FATAL_ERROR "OpenMP is enabled but not found.\n"
			"Install OpenMP or set ENABLE_OPENMP OFF.")
	endif()

    target_link_libraries(pennylane_lightning INTERFACE OpenMP::OpenMP_CXX)
    target_link_libraries(lightning_qubit_ops PRIVATE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_BLAS)
	message(STATUS "ENABLE_OPENBLAS is On. Find BLAS.")
	find_package(BLAS)

	if(BLAS_FOUND)
		message(STATUS "Use BLAS from ${BLA_VENDOR}.")
	else()
		message(FATAL_ERROR "BLAS is enabled but not found.")
	endif()

	message(STATUS "BLAS_LINKER_FLAGS=${BLAS_LINKER_FLAGS}")
	message(STATUS "BLAS_LIBRARIES=${BLAS_LIBRARIES}")

	target_link_libraries(lightning_qubit_ops PRIVATE "${BLAS_LIBRARIES}")
	target_link_options(lightning_qubit_ops PRIVATE "${BLAS_LINKER_FLAGS}")
	target_compile_options(lightning_qubit_ops PRIVATE "-DOPENBLAS=1")
endif()

if(BUILD_TESTS)
    enable_testing()
endif()
