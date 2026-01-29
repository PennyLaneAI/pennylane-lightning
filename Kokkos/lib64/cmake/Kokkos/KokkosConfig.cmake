# No need for policy push/pop. CMake also manages a new entry for scripts
# loaded by include() and find_package() commands except when invoked with
# the NO_POLICY_SCOPE option
# CMP0057 + NEW -> IN_LIST operator in IF(...)
cmake_policy(SET CMP0057 NEW)

# Compute paths

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was KokkosConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

#Find dependencies
include(CMakeFindDependencyMacro)

#This needs to go above the KokkosTargets in case
#the Kokkos targets depend in some way on the TPL imports
IF(NOT DEFINED rocthrust_DIR)
  SET(rocthrust_DIR  /opt/rocm/lib/cmake/rocthrust)
ENDIF()
FIND_DEPENDENCY(rocthrust)

get_filename_component(Kokkos_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
include("${Kokkos_CMAKE_DIR}/KokkosTargets.cmake")
include("${Kokkos_CMAKE_DIR}/KokkosConfigCommon.cmake")
unset(Kokkos_CMAKE_DIR)

# check for conflicts
if("launch_compiler" IN_LIST Kokkos_FIND_COMPONENTS AND "separable_compilation" IN_LIST Kokkos_FIND_COMPONENTS)
  message(STATUS "'launch_compiler' implies global redirection of targets depending on Kokkos to appropriate compiler.")
  message(
    STATUS
      "'separable_compilation' implies explicitly defining where redirection occurs via 'kokkos_compilation(PROJECT|TARGET|SOURCE|DIRECTORY ...)'"
  )
  message(FATAL_ERROR "Conflicting COMPONENTS: 'launch_compiler' and 'separable_compilation'")
endif()

if("launch_compiler" IN_LIST Kokkos_FIND_COMPONENTS)
  #
  # if find_package(Kokkos COMPONENTS launch_compiler) then rely on the
  # RULE_LAUNCH_COMPILE and RULE_LAUNCH_LINK to always redirect to the
  # appropriate compiler for Kokkos
  #

  message(
    STATUS
      "kokkos_launch_compiler is enabled globally. C++ compiler commands with -DKOKKOS_DEPENDENCE will be redirected to the appropriate compiler for Kokkos"
  )
  kokkos_compilation(GLOBAL CHECK_CUDA_COMPILES)

elseif(OFF AND (NOT CXX STREQUAL CUDA OR Kokkos_ENABLE_MULTIPLE_CMAKE_LANGUAGES)
       AND NOT "separable_compilation" IN_LIST Kokkos_FIND_COMPONENTS
)
  #
  # if CUDA was enabled, the compilation language was not set to CUDA, and separable compilation was not
  # specified, then set the RULE_LAUNCH_COMPILE and RULE_LAUNCH_LINK globally and
  # kokkos_launch_compiler will re-direct to the compiler used to compile CUDA code during installation.
  # kokkos_launch_compiler will re-direct if ${CMAKE_CXX_COMPILER} and -DKOKKOS_DEPENDENCE is present,
  # otherwise, the original command will be executed
  #

  # run test to see if CMAKE_CXX_COMPILER=nvcc_wrapper
  kokkos_compiler_is_nvcc(IS_NVCC ${CMAKE_CXX_COMPILER})

  # if not nvcc_wrapper and Kokkos_LAUNCH_COMPILER was not set to OFF
  if(NOT IS_NVCC AND (NOT DEFINED Kokkos_LAUNCH_COMPILER OR Kokkos_LAUNCH_COMPILER))
    message(
      STATUS
        "kokkos_launch_compiler is enabled globally. C++ compiler commands with -DKOKKOS_DEPENDENCE will be redirected to the appropriate compiler for Kokkos"
    )
    kokkos_compilation(GLOBAL)
  endif()

  # be mindful of the environment, pollution is bad
  unset(IS_NVCC)
endif()

set(Kokkos_COMPILE_LANGUAGE CXX)
