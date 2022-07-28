#!/usr/bin/bash/
cmake -DCMAKE_TOOLCHAIN_FILE=./cmake/toolchain.cmake  -DPYTHON_EXECUTABLE=/usr/bin/python3.10 -S. -B build
