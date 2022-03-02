#!/usr/bin/env python3
import csv
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

import re

plt.rc("font", family="sans-serif")


def parse_result_csv(filepath):
    n_qubits = []
    times = []
    with filepath.open() as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # ignore the first line
        for row in reader:
            n_qubits.append(int(row[0]))
            times.append(float(row[1]))

    return n_qubits, times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot benchmark results",
        epilog="Example: {} res_GNU_9.3.0 PauliX".format(sys.argv[0]),
    )
    parser.add_argument("path", help="Path to the result directory")
    parser.add_argument("gate_name", help="Name of gate to plot")

    args = parser.parse_args()

    res_dir = Path(args.path)
    gate_name = args.gate_name

    filename_rgx = re.compile(f"^benchmark_(.*?)_{gate_name}.csv$")

    res_files = []
    for file in res_dir.glob("*.csv"):
        m = filename_rgx.match(file.name)
        if m is not None:
            res_files.append((m.group(1), file))

    res_files.sort()

    if len(res_files) == 0:
        print(
            f"Cannot find results for {gate_name}. Check the results are obtained from run_gate_benchmark.sh."
        )
        sys.exit(1)
    num_kernels = len(res_files)

    total_num_qubits = set()

    for kernel_idx, (kernel_name, res_file) in enumerate(res_files):
        n_qubits, times = parse_result_csv(res_file)
        total_num_qubits |= set(n_qubits)
        n_qubits = np.array(n_qubits, dtype=float)
        plt.bar(n_qubits + 0.8 * (kernel_idx - num_kernels / 2 + 1 / 2), times, label=kernel_name)

    total_num_qubits = list(total_num_qubits)
    total_num_qubits.sort()
    plt.xticks(total_num_qubits)
    plt.legend()
    plt.title("{} ({})".format(gate_name, res_dir.name[4:]))
    plt.yscale("log")
    plt.xlabel("Number of qubits")
    plt.ylabel("Average time (ms)")

    plt.savefig(f"plot_{gate_name}.png")
