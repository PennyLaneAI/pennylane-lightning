cmake_minimum_required(VERSION 3.20)

project(lightning_kokkos_catalyst LANGUAGES CXX)

set(LK_CATALYST_FILES LightningKokkosSimulator.cpp CACHE INTERNAL "")
add_library(lightning_kokkos_catalyst SHARED ${LK_CATALYST_FILES})

include(FetchContent)

if(LIGHTNING_CATALYST_SRC_PATH)
    if(NOT IS_ABSOLUTE ${LIGHTNING_CATALYST_SRC_PATH})
        message(FATAL_ERROR " LIGHTNING_CATALYST_SRC_PATH=${LIGHTNING_CATALYST_SRC_PATH} must be set to an absolute path")
    endif()
    if(CATALYST_GIT_TAG)
        message(WARN " Setting `LIGHTNING_CATALYST_SRC_PATH=${LIGHTNING_CATALYST_SRC_PATH}` overrides `CATALYST_GIT_TAG=${CATALYST_GIT_TAG}`")
    endif()

    # Acquire local git hash and use for CATALYST_GIT_TAG
    execute_process(COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${LIGHTNING_CATALYST_SRC_PATH}
        OUTPUT_VARIABLE CATALYST_GIT_TAG
    )
    message(INFO " Building against local Catalyst - path: ${LIGHTNING_CATALYST_SRC_PATH} - GIT TAG: ${CATALYST_GIT_TAG}")

    target_include_directories(lightning_kokkos_catalyst PUBLIC ${LIGHTNING_CATALYST_SRC_PATH}/runtime/lib/backend/common)
    target_include_directories(lightning_kokkos_catalyst PUBLIC ${LIGHTNING_CATALYST_SRC_PATH}/runtime/include)

else()
    if(NOT CATALYST_GIT_TAG)
        set(CATALYST_GIT_TAG "main" CACHE STRING "GIT_TAG value to build Catalyst")
    endif()
    message(INFO " Building against Catalyst GIT TAG ${CATALYST_GIT_TAG}")

    # Fetching /lib/backend/common hpp headers
    set(LIB_BACKEND_COMMON_HEADERS  CacheManager.hpp
                                    QubitManager.hpp
                                    Utils.hpp
    )

    foreach(HEADER ${LIB_BACKEND_COMMON_HEADERS})
        string(REGEX REPLACE "\\.[^.]*$" "" HEADER_NAME ${HEADER})
        FetchContent_Declare(
            ${HEADER_NAME}
            URL                 https://raw.githubusercontent.com/PennyLaneAI/catalyst/${CATALYST_GIT_TAG}/runtime/lib/backend/common/${HEADER}
            DOWNLOAD_NO_EXTRACT True
            SOURCE_DIR          include
        )

        FetchContent_MakeAvailable(${HEADER_NAME})
    endforeach()

    # Fetching include hpp headers
    set(INCLUDE_HEADERS Exception.hpp
                        DataView.hpp
                        QuantumDevice.hpp
                        RuntimeCAPI.h
                        Types.h
    )

    foreach(HEADER ${INCLUDE_HEADERS})
        string(REGEX REPLACE "\\.[^.]*$" "" HEADER_NAME ${HEADER})
        FetchContent_Declare(
            ${HEADER_NAME}
            URL                 https://raw.githubusercontent.com/PennyLaneAI/catalyst/${CATALYST_GIT_TAG}/runtime/include/${HEADER}
            DOWNLOAD_NO_EXTRACT True
            SOURCE_DIR          include
        )

        FetchContent_MakeAvailable(${HEADER_NAME})
    endforeach()

    target_include_directories(lightning_kokkos_catalyst PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)

endif()

if(Kokkos_ENABLE_CUDA)
    message(STATUS "Kokkos_ENABLE_CUDA is ON")

    # KOKKOS_CUDA_OPTIONS:
    # - disable_malloc_async: Disable asynchronous memory allocation to ensure
    #   that memory operations complete before proceeding.
    # - enable_lambda: Allow the use of lambda expressions in parallel
    #   execution policies, used in LK kernel functors, and this flag is
    #   required when calling those functors from an external library.
    #   It's supported starting from NVCC v7.0.
    # - enable_constexpr: Enable compile-time evaluations for constants.
    #   There are some undefined behaviour when executing end-to-end Catalyst
    #   programs related to the value of `constexpr`s in the definition of gate kernels.
    #   This flag should be enforced for NVCC v7.5+.
    if(NVCC_VERSION VERSION_LESS "7.0")
        message(WARNING "Building lightning_kokkos_catalyst without lambda and constexpr support.")
        target_compile_definitions(lightning_kokkos_catalyst PRIVATE KOKKOS_CUDA_OPTIONS="disable_malloc_async")
    elseif(NVCC_VERSION VERSION_LESS "7.5")
        message(WARNING "Building lightning_kokkos_catalyst without constexpr support.")
        target_compile_definitions(lightning_kokkos_catalyst PRIVATE KOKKOS_CUDA_OPTIONS="disable_malloc_async enable_lambda")
    else()
        target_compile_definitions(lightning_kokkos_catalyst PRIVATE
            KOKKOS_CUDA_OPTIONS="disable_malloc_async enable_lambda enable_constexpr"
        )
    endif()
endif()

target_include_directories(lightning_kokkos_catalyst INTERFACE  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_kokkos_catalyst PUBLIC  lightning_compile_options
                                                        lightning_kokkos_algorithms
                                                        lightning_kokkos_measurements
)


if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
