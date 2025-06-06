cmake_minimum_required(VERSION 3.20)

project(lightning_bindings LANGUAGES CXX)

add_library(lightning_bindings INTERFACE)

target_include_directories(lightning_bindings INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(lightning_bindings INTERFACE lightning_utils lightning_gates lightning_compile_options)

set_property(TARGET lightning_bindings PROPERTY POSITION_INDEPENDENT_CODE ON)

# Add nanobind-specific configuration
if(ENABLE_NANOBIND)
    # Find nanobind package
    find_package(nanobind REQUIRED)
    
    # Add nanobind-specific source files
    set(NANOBIND_SOURCES
        Bindings_nb.cpp
    )
    
    # Create nanobind module target
    nanobind_add_module(
        lightning_core_nb
        ${NANOBIND_SOURCES}
    )
    
    # Link against required libraries
    target_link_libraries(lightning_core_nb PRIVATE 
        lightning_bindings
        lightning_utils
        lightning_gates
    )
    
    # Install the nanobind module
    install(TARGETS lightning_core_nb DESTINATION ${PYTHON_MODULE_PATH})
endif()
