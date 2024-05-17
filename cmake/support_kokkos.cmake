####################################################################################
# This file provides macros to process Kokkos and Kokkos Kernels external libraries.
####################################################################################

# Include this file only once
include_guard()

set(KOKKOS_VERSION 4.3.01)

# Macro to aid in finding Kokkos with 3 potential install options:
# 1. Fully integrated Kokkos packages and CMake module files
# 2. Statically compiled libraries and headers
# 3. Not installed, so fall back to building from source.
macro(FindKokkos target_name)
    find_package(Kokkos
    HINTS   ${pennylane_lightning_SOURCE_DIR}/kokkos
            ${pennylane_lightning_SOURCE_DIR}/Kokkos
            ${Kokkos_Core_DIR}
            /usr
            /usr/local
            /opt
            /opt/Kokkos
    )

    if(Kokkos_FOUND)
        message(STATUS "Found existing Kokkos library.")
        target_link_libraries(${target_name} INTERFACE Kokkos::kokkos)
    else()
        message(STATUS "Could not find existing Kokkos package. Searching for precompiled libraries and headers...")

        find_library(Kokkos_core_lib
            NAME kokkoscore.a libkokkoscore.a kokkoscore.so libkokkoscore.so
            HINTS   ${pennylane_lightning_SOURCE_DIR}/Kokkos/lib
                    ${Kokkos_Core_DIR}/lib
                    ${Kokkos_Core_DIR}/lib64
                    /usr/lib
                    /usr/lib64
                    /usr/local/lib
                    /usr/local/lib64
                    ENV LD_LIBRARY_PATH
        )
        find_file(  Kokkos_core_inc
            NAMES   Kokkos_Core.hpp
            HINTS   ${Kokkos_Core_DIR}/include
                    /usr/include
                    /usr/local/include
                    ENV CPATH
        )

        if (Kokkos_core_lib_FOUND)
            message(STATUS "Found existing Kokkos compiled libraries.")

            add_library( kokkos SHARED IMPORTED GLOBAL)

            cmake_path(GET Kokkos_core_inc ROOT_PATH Kokkos_INC_DIR)

            set_target_properties( kokkos PROPERTIES IMPORTED_LOCATION ${Kokkos_core_lib})
            set_target_properties( kokkos PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${Kokkos_INC_DIR}")

            target_link_libraries(${target_name} PRIVATE kokkos)
        else()
            message(STATUS "Building Kokkos from source. SERIAL device enabled.")
            message(STATUS "Requested Kokkos library version: ${KOKKOS_VERSION}")

            # option(Kokkos_ENABLE_SERIAL  "Enable Kokkos SERIAL device" ON)
            option(Kokkos_ENABLE_SERIAL  "Enable Kokkos SERIAL device" ON)
            option(Kokkos_ENABLE_COMPLEX_ALIGN "Enable complex alignment in memory" OFF)

            set(CMAKE_POSITION_INDEPENDENT_CODE ON)
            include(FetchContent)

            FetchContent_Declare(kokkos
                                GIT_REPOSITORY https://github.com/kokkos/kokkos.git
                                GIT_TAG        ${KOKKOS_VERSION}
                                GIT_SUBMODULES "" # Avoid recursively cloning all submodules
            )

            FetchContent_MakeAvailable(kokkos)

            get_target_property(kokkos_INC_DIR kokkos INTERFACE_INCLUDE_DIRECTORIES)
            set_target_properties(kokkos PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${kokkos_INC_DIR}")

            target_link_libraries(${target_name} INTERFACE kokkos)
        endif()
    endif()
endmacro()