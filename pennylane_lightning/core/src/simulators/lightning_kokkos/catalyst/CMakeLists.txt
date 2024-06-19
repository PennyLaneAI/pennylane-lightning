cmake_minimum_required(VERSION 3.20)

project(lightning_kokkos_catalyst LANGUAGES CXX)

set(CATALYST_FILES LightningKokkosSimulator.cpp CACHE INTERNAL "" FORCE)
add_library(lightning_kokkos_catalyst SHARED ${CATALYST_FILES})

include(FetchContent)

if(CATALYST_SRC_PATH)
    if(NOT IS_ABSOLUTE ${CATALYST_SRC_PATH})
        message(FATAL_ERROR " CATALYST_SRC_PATH=${CATALYST_SRC_PATH} must be set to an absolute path")
    endif()
    if(CATALYST_GIT_TAG)
        message(WARN " Setting `CATALYST_SRC_PATH=${CATALYST_SRC_PATH}` overrides `CATALYST_GIT_TAG=${CATALYST_GIT_TAG}`")
    endif()

    # Acquire local git hash and use for CATALYST_GIT_TAG
    execute_process(COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CATALYST_SRC_PATH}
        OUTPUT_VARIABLE CATALYST_GIT_TAG
    )
    message(INFO " Building against local Catalyst - path: ${CATALYST_SRC_PATH} - GIT TAG: ${CATALYST_GIT_TAG}")

    target_include_directories(lightning_kokkos_catalyst PRIVATE ${CATALYST_SRC_PATH}/runtime/lib/backend/common)
    target_include_directories(lightning_kokkos_catalyst PRIVATE ${CATALYST_SRC_PATH}/runtime/include)

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
    set(INCLUDE_HEADERS DataView.hpp
                        Exception.hpp
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

    target_include_directories(lightning_kokkos_catalyst PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

endif()

target_link_libraries(lightning_kokkos_catalyst PUBLIC  lightning_compile_options
                                                        lightning_external_libs
                                                        lightning_base
                                                        lightning_gates
                                                        lightning_utils
                                                        lightning_kokkos
                                                        lightning_kokkos_algorithms
                                                        lightning_kokkos_gates
                                                        lightning_kokkos_measurements
                                                        lightning_kokkos_utils
)

# if (BUILD_TESTS)
#     enable_testing()
#     add_subdirectory("tests")
# endif()