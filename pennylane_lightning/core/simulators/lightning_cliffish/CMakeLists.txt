cmake_minimum_required(VERSION 3.26)

project(lightning_cliffish
    DESCRIPTION "PennyLane Lightning Cliffish C++ Backend."
    LANGUAGES CXX
)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include("${pennylane_lightning_SOURCE_DIR}/cmake/support_kokkos.cmake")
FindKokkos(lightning_external_libs)

###############################################################################
# Simulator core. Its components are header-only libraries under this
# directory; the core library collects them.
###############################################################################
add_library(lightning_cliffish INTERFACE)

target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_PLCLIFFISH=1")

add_subdirectory(pauli)

target_link_libraries(lightning_cliffish INTERFACE lightning_compile_options
                                                   lightning_external_libs
                                                   lightning_utils
                                                   lightning_cliffish_pauli
                                                   )

target_include_directories(lightning_cliffish INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

###############################################################################
# Catalyst device. lightning.cliffish has no Python module: Catalyst loads this
# library directly.
###############################################################################
add_subdirectory(catalyst)
