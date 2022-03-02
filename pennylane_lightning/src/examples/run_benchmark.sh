#!/bin/bash

currdir=$(pwd)

if [ "$#" -lt 2 ]; then
	echo "Usage: $0 Kernel Gate [Number of wires (for MultiRZ)]"
	exit 1
fi

# Parameter initialization
min_num_qubits=8
max_num_qubits=24
num_qubits_increment=2
num_gate_reps=1000
kernel="$1"
gate="$2"
path_to_binary="./benchmark_gate"

compiler_info=$(<compiler_info.txt)

resdir="$currdir/res_${compiler_info}"
mkdir -p $resdir
data_file_name="benchmark_${kernel}_${gate}.csv"
path_to_csv="$resdir/$data_file_name"
echo "Creating $path_to_csv"
echo "Num Qubits, Time (milliseconds)" > $path_to_csv

if [[ "$#" -eq 3 ]]; then
	command_format="$path_to_binary ${num_gate_reps} %d ${kernel} ${gate} $3"
else
	command_format="$path_to_binary ${num_gate_reps} %d ${kernel} ${gate}"
fi

# Generate data
for ((num_qubits=$min_num_qubits; num_qubits<$max_num_qubits+1; num_qubits+=$num_qubits_increment)); do
	echo "Gate repetition=$num_gate_reps, num_qubits=$num_qubits, kernel=$kernel, gate=$gate"
	command=$(printf "$command_format" "$num_qubits")
	$command >> $path_to_csv
done
