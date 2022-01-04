#!/bin/bash

currdir=$(pwd)

# Parameter initialization
min_num_qubits=6
max_num_qubits=22
num_qubits_increment=2
num_gate_reps=10000
kernel="LM"
gate="PauliX"

# Creating data file
data_file_name="benchmark_res_$kernel_$gate.csv"
binary_name="./gate_benchmark_oplist"
path_to_binary=$currdir/$binary_name
path_to_csv=$currdir/$data_file_name
echo "Creating $path_to_csv"
echo "Num Qubits, Time (milliseconds)" > $path_to_csv

# Generate data
for ((num_qubits=$min_num_qubits; num_qubits<$max_num_qubits+1; num_qubits+=$num_qubits_increment)); do
    echo "Gate repetition=$num_gate_reps, num_qubits=$num_qubits, kernel="$kernel", gate=$gate"
    $path_to_binary ${num_gate_reps} ${num_qubits} ${kernel} ${gate} >> $path_to_csv
done

# Plot results
# python_path=$(which python3)
# echo "Plotting results"
# $python_path gate_benchmark_plotter.py $path_to_csv $path_to_compiler_file
