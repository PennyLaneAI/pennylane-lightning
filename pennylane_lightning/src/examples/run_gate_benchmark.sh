#!/bin/bash

crt_dir=$(pwd)

# Export env variables in case cmake reverts to default values
export CXX=$1
if [ $# -eq 0 ]; then
    echo "Usage: bash $0 CXX_COMPILER"
    exit 1
fi

# Compiler version & optimization
compiler_file_name=compiler_info.txt
path_to_compiler_file=$crt_dir/$compiler_file_name
echo "Creating $path_to_compiler_file"
$CXX --version | head -n 1 > $path_to_compiler_file

# CMake & make
mkdir build
pushd ./build
cmake -DCMAKE_CXX_COMPILER=$CXX .. && make
popd

# Parameter initialization
min_num_qubits=6
max_num_qubits=22
num_qubits_increment=2
num_gate_reps=3

# Creating data file
data_file_name=gate_benchmark.csv
binary_dir=$crt_dir/build
binary_name=gate_benchmark
path_to_binary=$binary_dir/$binary_name
path_to_csv=$crt_dir/$data_file_name
echo "Creating $path_to_csv"
echo "Num Qubits, Time (milliseconds)" > $path_to_csv

# Generate data
for ((num_qubits=$min_num_qubits; num_qubits<$max_num_qubits+1; num_qubits+=$num_qubits_increment)); do
    printf "Run with %1d gate repitions and %2d qubits \n" "$num_gate_reps" "$num_qubits"
    $path_to_binary ${num_gate_reps} ${num_qubits} >> $path_to_csv
done

# Plot results
python_path=$(which python3)
echo "Plotting results"
$python_path gate_benchmark_plotter.py $path_to_csv $path_to_compiler_file