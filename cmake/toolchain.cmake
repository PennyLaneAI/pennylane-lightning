set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm64)

set(triple aarch64-linux-gnu)

set(CMAKE_C_COMPILER clang-14)
set(CMAKE_C_COMPILER_TARGET ${triple})
set(CMAKE_CXX_COMPILER clang++-14)
set(CMAKE_CXX_COMPILER_TARGET ${triple})

#set(CMAKE_C_FLAGS "-march=armv8")
#set(CMAKE_CXX_FLAGS "-march=armv8")

#set(CMAKE_C_FLAGS "-fms-extensions")
#set(CMAKE_CXX_FLAGS "-fms-extensions")

#set(CMAKE_C_FLAGS "-Wno-error=extra-tokens")
#set(CMAKE_CXX_FLAGS "-Wno-error=extra-tokens")
