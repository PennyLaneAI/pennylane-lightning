####################################################################################
# This file provides macros to support the test suite.
####################################################################################

# Include this file only once
include_guard()

# This macro fetch Catch2 from its Github repository.
# After that Catch2 is configured and included.
macro(FetchAndIncludeCatch)
    Include(FetchContent)

    FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.10
    )

    FetchContent_MakeAvailable(Catch2)

    get_target_property(CATCH2_IID Catch2 INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(Catch2 PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${CATCH2_IID}")

    # Required for catch_discover_tests() and include(Catch)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

    include(Catch)
endmacro()

# Process ENABLE_WARNINGS, and ENABLE_NATIVE options.
macro(ProcessTestOptions target_name)
    if(ENABLE_WARNINGS)
        message(STATUS "ENABLE_WARNINGS is ON.")
        if(MSVC)
            target_compile_options(${target_name} INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4;/WX>)
        else()
            target_compile_options(${target_name} INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>)
        endif()
    else()
        if(MSVC)
            target_compile_options(${target_name} INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
        else()
            target_compile_options(${target_name} INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wno-unused>)
        endif()
    endif()

    if(ENABLE_NATIVE)
        message(STATUS "ENABLE_NATIVE is ON. Use -march=native for cpptests.")
        target_compile_options(${target_name} INTERFACE -march=native)
    endif()
endmacro()