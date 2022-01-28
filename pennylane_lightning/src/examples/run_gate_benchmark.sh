#!/bin/bash

currdir=$(pwd)

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 Kernel Gate"
	exit 1
fi

# Parameter initialization
min_num_qubits=8
max_num_qubits=24
num_qubits_increment=2
num_gate_reps=1000
kernel="$1"
gate="$2"

compiler_info=$(<compiler_info.txt)

# Creating data file
binary_name="./gate_benchmark_oplist"
path_to_binary="$currdir/$binary_name"

resdir="$currdir/res_${compiler_info}"
mkdir -p $resdir
data_file_name="benchmark_${kernel}_${gate}.csv"
path_to_csv="$resdir/$data_file_name"
echo "Creating $path_to_csv"
echo "Num Qubits, Time (milliseconds)" > $path_to_csv

# Generate data
for ((num_qubits=$min_num_qubits; num_qubits<$max_num_qubits+1; num_qubits+=$num_qubits_increment)); do
    echo "Gate repetition=$num_gate_reps, num_qubits=$num_qubits, kernel=$kernel, gate=$gate"
    $path_to_binary ${num_gate_reps} ${num_qubits} ${kernel} ${gate} >> $path_to_csv
done

# Plot results
# python_path=$(which python3)
# echo "Plotting results"
# $python_path gate_benchmark_plotter.py $path_to_csv $path_to_compiler_file
