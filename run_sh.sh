#!/bin/bash

#cmake -B build -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug -DENABLE_PYTHON=OFF -DPL_BACKEND=lightning_qubit 
#cmake --build build 
#rm -rf build
#cmake -B build -DBUILD_TESTS=ON -DENABLE_LAPACK=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DPL_BACKEND=lightning_kokkos
#cmake -B build -DCUQUANTUM_SDK=$HOME/venv/lib/python3.10/site-packages/cuquantum -DBUILD_TESTS=ON -DENABLE_MPI=ON -DPL_BACKEND=lightning_gpu
#cmake -B build -DBUILD_TESTS=ON -DENABLE_MPI=ON -DPL_BACKEND=lightning_gpu
#cmake -B build -DBUILD_TESTS=ON -DENABLE_MPI=OFF -DPL_BACKEND=lightning_kokkos
#cmake -B build -DBUILD_TESTS=ON -DENABLE_OPENMP=OFF -DPL_BACKEND=lightning_kokkos

#cmake -B build -DBUILD_TESTS=ON -DENABLE_LAPACK=ON -DENABLE_OPENMP=OFF -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_SERIAL=ON -DPL_BACKEND=lightning_kokkos
cmake -B build -DBUILD_TESTS=ON -DENABLE_LAPACK=ON -DENABLE_PYTHON=OFF -DPL_BACKEND=lightning_qubit
cmake --build build

# Specify the directory where the executables are located
directory="./build"

# Use a loop to find and run all executables with "_runner" in their names
#for file in "$directory"/*_runner_mpi; do
#    if [ -x "$file" ]; then
#        echo "Running $file"
#        mpirun -np 2 $file
#    else
#        echo "Error: $file is not executable."
#    fi
#done

echo "*************************"#
echo "Single GPU backend tests!"
echo "*************************"

for file in "$directory"/*_runner; do
    if [ -x "$file" ]; then
        echo "Running $file"
        $file
    else
        echo "Error: $file is not executable."
    fi
done

#cmake -S. -B build -DBUILD_TESTS=ON -DENABLE_PYTHON=OFF
#cmake --build build
