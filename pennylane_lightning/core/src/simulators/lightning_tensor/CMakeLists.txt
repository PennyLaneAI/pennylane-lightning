cmake_minimum_required(VERSION 3.21)

project(lightning_tensor_components LANGUAGES CXX)

###############################################################################
# Include all nested sources directories
###############################################################################
set(COMPONENT_SUBDIRS   base
                        mps                
                        utils
)

foreach(COMP ${COMPONENT_SUBDIRS})
    add_subdirectory(${COMP})
endforeach()
