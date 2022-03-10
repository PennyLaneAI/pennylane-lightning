#include <vector>

#include "Constant.hpp"
#include "StateVectorManaged.hpp"

#include <benchmark/benchmark.h>

static void applyOperation_HCNOT(benchmark::State &state,
                                 Pennylane::Gates::KernelType kernel) {
    auto num_qubits = static_cast<size_t>(state.range(0));
    for (auto _ : state) {
        Pennylane::StateVectorManaged<double> sv{num_qubits};

        sv.applyOperation(kernel, "Hadamard", {0}, false);
        for (size_t i = 0; i < num_qubits - 1; i++) {
            sv.applyOperation(kernel, "CNOT", {i, i + 1}, false);
        }

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}

BENCHMARK_CAPTURE(applyOperation_HCNOT, kernel_LM,
                  Pennylane::Gates::KernelType::LM)
    ->RangeMultiplier(1l << 1)
    ->Range(2l, 26l);

BENCHMARK_CAPTURE(applyOperation_HCNOT, kernel_PI,
                  Pennylane::Gates::KernelType::PI)
    ->RangeMultiplier(2l)
    ->Range(2l, 26l);
