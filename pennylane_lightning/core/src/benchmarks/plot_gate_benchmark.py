#!/usr/bin/env python3
import json
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

from re import compile as re_compile

plt.rc("font", family="sans-serif")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")


class BenchmarkDataProcessor:
    def __init__(self, precision=None):
        if precision not in [None, "float", "double"]:
            raise ValueError("Argument precision must be one of None, float, or double")
        self.name_rgx = re_compile(r"^(\w+)<(\w+)>/(\w+)/(\d+)(/(\d+))?".format(precision))
        self.precision = precision

    def parse_result_json(self, filepath):
        parsed_data = defaultdict(list)
        with filepath.open("r") as jsonfile:
            all_data = json.load(jsonfile)

            for d in all_data["benchmarks"]:
                m = self.name_rgx.match(d["name"])
                op_name, precision, kernel, num_qubits, _, num_wires = m.groups()

                if self.precision and precision != self.precision:
                    print("Ignore {} as the floating point type".format(d["name"]))
                    continue

                if self.precision is None:
                    kernel = f"{kernel}({precision})"

                time = d["real_time"] / 1e6  # to ms
                if num_wires:  # multiqubit gates
                    op_name = f"{op_name}_{num_wires}"
                    timing_data = parsed_data[(op_name, kernel)]
                else:
                    timing_data = parsed_data[(op_name, kernel)]
                timing_data.append([int(num_qubits), time])

        for k in parsed_data.keys():
            parsed_data[k].sort()
            parsed_data[k] = np.array(parsed_data[k])
        self.parsed_data = parsed_data

    def all_ops(self):
        return set(op_name for op_name, _ in self.parsed_data.keys())

    def get_data_for_op(self, op_name):
        keys = [k for k in self.parsed_data.keys() if k[0] == op_name]
        return {k[1]: self.parsed_data[k] for k in keys}

    def plot_to_file(self, filepath, op_name):
        fig, ax = plt.subplots()
        data = self.get_data_for_op(op_name)

        num_kernels = len(data)
        width = min(1.6 / num_kernels, 0.8)

        for kernel_idx, kernel_name in enumerate(data.keys()):
            to_plot = data[kernel_name]
            num_qubits = to_plot[:, 0]
            times = to_plot[:, 1]
            ax.bar(
                num_qubits + width * (kernel_idx - num_kernels / 2 + 1 / 2),
                times,
                width=width,
                label=kernel_name,
            )

        ax.legend()
        ax.set_xticks(num_qubits)
        ax.set_yscale("log")
        ax.set_xlabel("Number of qubits")
        ax.set_ylabel("Average time (ms)")

        ax.set_title("{}".format(op_name.replace("_", r"\_")))
        plt.savefig(filepath)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot benchmark results",
        epilog="Example: {} res.json float".format(sys.argv[0]),
    )
    parser.add_argument("res_file", help="Path to the result json file", type=Path)
    parser.add_argument(
        "--precision",
        help="Floating point precision to filter",
        choices=["float", "double"],
        default=None,
    )
    parser.add_argument(
        "--plot_dir", help="Output directory for plots", default="plots", metavar="DIR"
    )

    args = parser.parse_args()

    data_processor = BenchmarkDataProcessor(args.precision)
    data_processor.parse_result_json(args.res_file)

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for op_name in data_processor.all_ops():
        data_processor.plot_to_file(plot_dir.joinpath(f"{op_name}.png"), f"{op_name}")
