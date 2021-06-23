cmake_minimum_required(VERSION 3.14)

project(pennylane_lightning)

# Read and set pennylane_lightning version
execute_process(COMMAND "/usr/bin/env" "python"  "-c"
	"with open('${CMAKE_SOURCE_DIR}/pennylane_lightning/_version.py') as f: print(f.readlines()[-1].split()[-1].strip(\"\\\"'\"), end='')"
	OUTPUT_VARIABLE VERSION_STRING)
message(STATUS "pennylane_lightning version ${VERSION_STRING}")
set(PROJECT_VERSION ${VERSION_STRING})

set(CMAKE_CXX_STANDARD 11) # At least C++11 is required
find_package(OpenMP REQUIRED) # find OpenMP

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -W -DVERSION_INFO=${VERSION_STRING}")


# Add pybind11
include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        stable 
)
FetchContent_MakeAvailable(pybind11)


add_library(external_dependency INTERFACE)


if (("$ENV{USE_MKL}" OR "${USE_MKL}") AND (DEFINED ENV{MKLROOT}))
	message(STATUS "Found MKL")
	target_link_libraries(external_dependency INTERFACE mkl_intel_lp64 mkl_gnu_thread 
		mkl_core gomp pthread m dl)
	target_include_directories(external_dependency INTERFACE "$ENV{MKLROOT}/include")
	target_compile_options(external_dependency INTERFACE "-DMKL=1")
elseif ("$ENV{USE_LAPACK}" OR "${USE_LAPACK}")
    message(STATUS "Use LAPACKE")
	target_link_libraries(external_dependency INTERFACE lapacke)
	target_compile_options(external_dependency INTERFACE "-DLAPACKE=1")
elseif ("$ENV{USE_OPENBLAS}" OR "${USE_OPENBLAS}")
	message(STATUS "Use OPENBLAS")
	target_link_libraries(external_dependency INTERFACE openblas)
	target_compile_options(external_dependency INTERFACE "-DOPENBLAS=1")
endif()

pybind11_add_module(lightning_qubit_ops "pennylane_lightning/src/Apply.cpp"
	"pennylane_lightning/src/Gates.cpp" "pennylane_lightning/src/Bindings.cpp"
	"pennylane_lightning/src/StateVector.cpp")
target_link_libraries(lightning_qubit_ops PRIVATE external_dependency)
set_target_properties(lightning_qubit_ops PROPERTIES CXX_VISIBILITY_PRESET hidden)

target_link_libraries(lightning_qubit_ops PRIVATE OpenMP::OpenMP_CXX)

if (${BUILD_TESTS})
	enable_testing()
	add_subdirectory("pennylane_lightning/src/tests" "tests")
endif()
