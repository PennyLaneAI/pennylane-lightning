import subprocess
import argparse
import json
from pathlib import Path
from typing import final
import abc

MIN_NUM_QUBITS = 8
MAX_NUM_QUBITS = 24
STEP_NUM_QUBITS = 2
NUM_GATE_REPS = 1000


class BenchmarkRunner:
    def __init__(self, kernel, operation):
        self.kernel = kernel
        self.operation = operation

    @final
    def benchmark(self, res_path):
        result = []
        ext_info = self.external_info()
        if ext_info:
            result.append(ext_info)
        try:
            for num_qubit in range(MIN_NUM_QUBITS, MAX_NUM_QUBITS + 1, STEP_NUM_QUBITS):
                cmd = self.command(num_qubit)
                print(f"Run N={num_qubit}, {self.kernel}, {self.operation}")
                output = subprocess.run([str(c) for c in cmd], capture_output=True, check=True)
                time = output.stdout.decode("utf-8").strip().split(",")[1]
                result.append({"N": num_qubit, "time": time})
        except subprocess.CalledProcessError as err:
            print("Error from subprocess call. Message:")
            print(err.stderr.decode("utf-8"))
        except KeyboardInterrupt:
            pass

        res_path = Path(res_path)
        if not res_path.exists():
            res_path.mkdir(parents=True)

        with res_path.joinpath(self.filename()).open("w") as f:
            json.dump(result, f, indent=4)

    @abc.abstractmethod
    def command(self, num_qubits):
        pass

    @abc.abstractmethod
    def external_info(self):
        pass

    @abc.abstractmethod
    def filename(self):
        pass


class MatrixBenchmarkRunner(BenchmarkRunner):
    def __init__(self, kernel, operation, num_wires):
        super().__init__(kernel, operation)
        self.num_wires = num_wires

    def command(self, num_qubits):
        return ["./benchmark_matrix", NUM_GATE_REPS, num_qubits, self.kernel, self.num_wires]

    def external_info(self):
        return {"num_wires": self.num_wires}

    def filename(self):
        return f"Matrix_{self.kernel}_{self.num_wires}.json"


class GateBenchmarkRunner(BenchmarkRunner):
    def __init__(self, kernel, operation, num_wires=None):
        super().__init__(kernel, operation)
        self.num_wires = num_wires

    def command(self, num_qubits):
        cmd = ["./benchmark_gate", NUM_GATE_REPS, num_qubits, self.kernel, self.operation]
        if self.num_wires:
            cmd.append(self.num_wires)
        return cmd

    def external_info(self):
        if self.num_wires:
            return {"num_wires": self.num_wires}
        return None

    def filename(self):
        if self.num_wires:
            return f"{self.operation}_{self.kernel}_{self.num_wires}.json"
        return f"{self.operation}_{self.kernel}.json"


class GeneratorBenchmarkRunner(BenchmarkRunner):
    def __init__(self, kernel, operation, num_wires=None):
        super().__init__(kernel, operation)
        self.num_wires = num_wires

    def command(self, num_qubits):
        cmd = ["./benchmark_generator", NUM_GATE_REPS, num_qubits, self.kernel, self.operation[9:]]
        if self.num_wires is not None:
            cmd.append(self.num_wires)
        return cmd

    def external_info(self):
        if self.num_wires:
            return {"num_wires": self.num_wires}
        return None

    def filename(self):
        if self.num_wires:
            return f"{self.operation}_{self.kernel}_{self.num_wires}.json"
        return f"{self.operation}_{self.kernel}.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run C++ benchmarks")
    parser.add_argument("kernel", help="Kernel to benchmark")
    parser.add_argument("operation", help="Operation to benchmark.")

    parser.add_argument(
        "num_wires",
        help="Number of wires (optional for multi-qubit operations).",
        nargs="?",
        default=None,
        type=int,
    )

    args = parser.parse_args()

    compiler_info_file = "compiler_info.txt"

    try:
        with open(compiler_info_file, "r") as f:
            res_path = "res_" + f.readline().strip()
    except OSError:
        print("Encountered an error while opening '{}'".format(compiler_info_file))
        sys.exit(1)

    if args.operation == "Matrix":
        if args.num_wires == 0:
            raise ValueError(
                "Parameter num_wires must be provided and larger than 0 for matrix benchmark."
            )
        runner = MatrixBenchmarkRunner(args.kernel, args.operation, args.num_wires)
    elif args.operation.startswith("Generator"):
        runner = GeneratorBenchmarkRunner(args.kernel, args.operation, args.num_wires)
    else:
        runner = GateBenchmarkRunner(args.kernel, args.operation, args.num_wires)

    runner.benchmark(res_path)
