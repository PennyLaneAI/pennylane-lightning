#include <algorithm>
#include <random>
#include <vector>

#include "Constant.hpp"
#include "StateVectorManaged.hpp"

#include <benchmark/benchmark.h>

template <typename RandomEngine>
static inline auto generateDistinctWires(RandomEngine &eng, size_t num_qubits,
                                         size_t num_wires)
    -> std::vector<size_t> {
    std::vector<size_t> v(num_qubits, 0);
    std::iota(v.begin(), v.end(), 0);
    shuffle(v.begin(), v.end(), eng);
    return {v.begin(), v.begin() + num_wires};
}

static void applyOperation_MultiRZ(benchmark::State &state,
                                   const Pennylane::Gates::KernelType kernel) {
    size_t num_gates = state.range(0);
    size_t num_qubits = state.range(1);
    size_t num_wires = state.range(2);

    if (!num_gates) {
        state.SkipWithError("Invalid number of gates.");
    }

    if (!num_qubits) {
        state.SkipWithError("Invalid number of qubits.");
    }

    if (!num_wires) {
        state.SkipWithError("Invalid number of wires.");
    }

    std::random_device rd;
    std::mt19937_64 eng(rd());

    std::uniform_real_distribution<double> param_distr(-M_PI, M_PI);

    std::vector<std::vector<size_t>> wires;
    std::vector<double> params;

    wires.reserve(num_gates);
    params.reserve(num_gates);

    for (size_t i = 0; i < num_gates; i++) {
        wires.emplace_back(generateDistinctWires(eng, num_qubits, num_wires));
        params.emplace_back(param_distr(eng));
    }

    for (auto _ : state) {
        Pennylane::StateVectorManaged<double> sv{num_qubits};

        for (size_t g = 0; g < num_gates; g++) {
            sv.applyOperation(kernel, "MultiRZ", wires[g], false, {params[g]});
        }

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}

BENCHMARK_CAPTURE(applyOperation_MultiRZ, kernel_LM,
                  Pennylane::Gates::KernelType::LM)
    ->RangeMultiplier(2l)
    ->Ranges({{8, 64}, {4, 24}, {2, 2}});

BENCHMARK_CAPTURE(applyOperation_MultiRZ, kernel_PI,
                  Pennylane::Gates::KernelType::PI)
    ->RangeMultiplier(2l)
    ->Ranges({{8, 64}, {4, 24}, {2, 2}});
