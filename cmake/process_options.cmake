# Set compile flags and library dependencies
add_library(pennylane_lightning_compile_options INTERFACE)
add_library(pennylane_lightning_external_libs INTERFACE)

if(MSVC) # For M_PI
    target_compile_options(pennylane_lightning_compile_options INTERFACE /D_USE_MATH_DEFINES)
endif()

# Add -fwrapv, -fno-plt in Clang
if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"))
    target_compile_options(pennylane_lightning_compile_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-fwrapv;-fno-plt>)
# Add -fwrapv, -fno-plt, -pipe in GCC
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(pennylane_lightning_compile_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-fwrapv;-fno-plt;-pipe>)
endif()

if(ENABLE_WARNINGS)
    message(STATUS "ENABLE_WARNINGS is ON.")
    if(MSVC)
        target_compile_options(pennylane_lightning_compile_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4;/WX>)
    else()
        target_compile_options(pennylane_lightning_compile_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>)
    endif()
endif()

if(ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Using -march=native.")
    target_compile_options(pennylane_lightning_compile_options INTERFACE -march=native)
endif()

if(ENABLE_AVX)
    message(STATUS "ENABLE_AVX is ON.")
    target_compile_options(pennylane_lightning_compile_options INTERFACE -mavx)
endif()

if(ENABLE_OPENMP)
    message(STATUS "ENABLE_OPENMP is ON. Using OpenMP.")
    find_package(OpenMP)

    if(NOT OpenMP_CXX_FOUND)
        message(FATAL_ERROR "OpenMP is enabled but not found.\n"
            "Install OpenMP or set ENABLE_OPENMP OFF.")
    endif()

    target_link_libraries(pennylane_lightning_external_libs INTERFACE OpenMP::OpenMP_CXX)
endif()


if(ENABLE_BLAS)
    message(STATUS "ENABLE_BLAS is ON. Find BLAS.")
    find_package(BLAS)

    if(NOT BLAS_FOUND)
        message(FATAL_ERROR "BLAS is enabled but not found.")
    endif()

    target_link_libraries(pennylane_lightning_external_libs INTERFACE "${BLAS_LIBRARIES}")
    target_link_options(pennylane_lightning_external_libs INTERFACE "${BLAS_LINKER_FLAGS}")
    target_compile_options(pennylane_lightning_compile_options INTERFACE "-D_ENABLE_BLAS=1")
endif()


