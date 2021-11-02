#!/bin/bash

crt_dir=$(pwd)

# Export env variables in case cmake reverts to default values
export CXX=$(which clang++)
export CMAKE_CXX_FLAGS=-O3 # Flags passed via bash only for ease of use; would normally set in CMakeLists.txt

# Compiler version & optimization
compiler_file_name=compiler_info.txt
path_to_compiler_file=$crt_dir/$compiler_file_name
echo "Creating $path_to_compiler_file"
$CXX --version | head -n 1 > $path_to_compiler_file
echo $CMAKE_CXX_FLAGS >> $path_to_compiler_file

# CMake & make
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS" .. && make
cd ..

# Parameter initialization
min_num_qubits=6
max_num_qubits=22
num_qubits_increment=2
num_gate_reps=1
num_runs=3

# Creating data file
data_file_name=gate_benchmark.csv
binary_dir=$crt_dir/build
binary_name=gate_benchmark
path_to_binary=$binary_dir/$binary_name
path_to_csv=$crt_dir/$data_file_name
echo "Creating $path_to_csv"
echo "Num Gate Reptitions, Num Qubits, Time (milliseconds)" > $path_to_csv

# Generate data
for ((num_qubits=$min_num_qubits; num_qubits<$max_num_qubits+1; num_qubits+=$num_qubits_increment)); do
    for ((run=0; run<$num_runs; run++)); do
        printf "Run number %1d with %2d qubits \n" "$run" "$num_qubits"
        $path_to_binary ${num_gate_reps} ${num_qubits} >> $path_to_csv
    done
done

# Plot results
python_path=$(which python3)
echo "Plotting results"
$python_path gate_benchmark_plotter.py $path_to_csv $path_to_compiler_file