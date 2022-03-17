// Copyright 2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <algorithm>
#include <random>
#include <vector>

#include "Constant.hpp"
#include "StateVectorManaged.hpp"

#include "Bench_Utils.hpp"

using Kernel = Pennylane::Gates::KernelType;

/**
 * @brief Generate distinct wires.
 *
 * @tparam RandomEngine Random number generator engine.
 * @param num_qubits Number of qubits.
 * @param num_wires Number of wires.
 *
 * @return std::vector<size_t>
 */
template <typename RandomEngine>
static inline auto generateDistinctWires(RandomEngine &eng, size_t num_qubits,
                                         size_t num_wires)
    -> std::vector<size_t> {
    std::vector<size_t> v(num_qubits, 0);
    std::iota(v.begin(), v.end(), 0);
    shuffle(v.begin(), v.end(), eng);
    return {v.begin(), v.begin() + num_wires};
}

//***********************************************************************//
//                            applyOperation
//***********************************************************************//

/**
 * @brief Benchmark applyOperation for "MultiRZ" in PennyLane-Lightning.
 *
 * @tparam T Floating point precision type.
 * @param kernel Pennylane::Gates::KernelType.
 */
template <class T>
static void applyOperation_MultiRZ(benchmark::State &state, Kernel kernel) {
    const size_t num_gates = state.range(0);
    const size_t num_qubits = state.range(1);
    const size_t num_wires = state.range(2);

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

    std::uniform_real_distribution<T> param_distr(-M_PI, M_PI);

    std::vector<std::vector<size_t>> wires;
    std::vector<T> params;

    wires.reserve(num_gates);
    params.reserve(num_gates);

    for (size_t i = 0; i < num_gates; i++) {
        wires.emplace_back(generateDistinctWires(eng, num_qubits, num_wires));
        params.emplace_back(param_distr(eng));
    }

    for (auto _ : state) {
        Pennylane::StateVectorManaged<T> sv{num_qubits};

        for (size_t g = 0; g < num_gates; g++) {
            sv.applyOperation(kernel, "MultiRZ", wires[g], false, {params[g]});
        }

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}

BENCHMARK_APPLYOPS(applyOperation_MultiRZ, float, LM, Kernel::LM)
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
        benchmark::CreateDenseRange(2, 4, /*step=*/2),  // num_wires
    });

BENCHMARK_APPLYOPS(applyOperation_MultiRZ, float, PI, Kernel::PI)
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
        benchmark::CreateDenseRange(2, 4, /*step=*/2),  // num_wires
    });

BENCHMARK_APPLYOPS(applyOperation_MultiRZ, double, LM, Kernel::LM)
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
        benchmark::CreateDenseRange(2, 4, /*step=*/2),  // num_wires
    });

BENCHMARK_APPLYOPS(applyOperation_MultiRZ, double, PI, Kernel::PI)
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
        benchmark::CreateDenseRange(2, 4, /*step=*/2),  // num_wires
    });
