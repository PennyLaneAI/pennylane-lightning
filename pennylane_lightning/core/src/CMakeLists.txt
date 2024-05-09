cmake_minimum_required(VERSION 3.20)

project(lightning_components LANGUAGES CXX)

###############################################################################
# Include all nested sources directories
###############################################################################
set(COMPONENT_SUBDIRS   algorithms
                        bindings
                        gates
                        measurements
                        observables
                        simulators
                        utils
)

if("${PL_BACKEND}" STREQUAL "lightning_tensor")
    add_subdirectory(gates)
    add_subdirectory(simulators)
    add_subdirectory(utils)
else()
    foreach(COMP ${COMPONENT_SUBDIRS})
        add_subdirectory(${COMP})
    endforeach()
endif()

if (BUILD_TESTS)
    # Include macros supporting tests.
    include("${pennylane_lightning_SOURCE_DIR}/cmake/support_tests.cmake")
    FetchAndIncludeCatch()

    include(CTest)
endif()