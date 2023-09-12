####################################################################################
# This file provides macros to support PennyLane Lightning simulators,
# and to process the PL_BACKEND variable.
####################################################################################

# Include this file only once
include_guard()

# All simulators have their own directory in "simulators"
# This macro will extract this list of directories.
MACRO(FIND_SIMULATORS_LIST RESULT)
    set(SIMULATORS_DIR ${pennylane_lightning_SOURCE_DIR}/pennylane_lightning/core/src/simulators)
    FILE(GLOB FULL_LIST RELATIVE ${SIMULATORS_DIR} ${SIMULATORS_DIR}/*)
    SET(${RESULT} "")
    FOREACH(ITEM ${FULL_LIST})
        IF(IS_DIRECTORY ${SIMULATORS_DIR}/${ITEM})
            LIST(APPEND ${RESULT} ${ITEM})
        ENDIF()
    ENDFOREACH()
ENDMACRO()

# Checking if the chosen simulator (or Backend) is valid.
# If valid: its directory will be added to the building process.
# If invalid: A error message, with a list of valid simulators (backends), will be printed out.
MACRO(FIND_AND_ADD_SIMULATOR)
    # Finding the list of simulators:
    FIND_SIMULATORS_LIST(SIMULATORS_LIST)

    FOREACH(BACKEND ${PL_BACKEND})
        if (${BACKEND} IN_LIST SIMULATORS_LIST)
            add_subdirectory(${BACKEND})
        else()
            message("Could not find the requested backend. Options found are:")
            FOREACH(SIMULATOR ${SIMULATORS_LIST})
                message("      * " ${SIMULATOR})
            ENDFOREACH()
            message(FATAL_ERROR "Building process will not proceed. Failed to find backend.")
        endif()
    ENDFOREACH()
ENDMACRO()
