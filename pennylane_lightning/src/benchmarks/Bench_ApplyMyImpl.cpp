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
/*
template <typename RandomEngine>
static inline auto generateDistinctWires(RandomEngine &eng, size_t num_qubits,
                                         size_t num_wires)
    -> std::vector<size_t> {
    std::vector<size_t> v(num_qubits, 0);
    std::iota(v.begin(), v.end(), 0);
    shuffle(v.begin(), v.end(), eng);
    std::vector<size_t> vo(1,0);
    vo.begin() = v.begin();
    //return {v.begin(), v.begin() + num_wires};
    return vo;
}
*/
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
static void applyOperation_MyImplPauliX(benchmark::State &state, Kernel kernel) {
    const size_t num_qubits = state.range(0);
    const size_t num_wires = state.range(1);

    if (!num_qubits) {
        state.SkipWithError("Invalid number of qubits.");
    }
    if (!num_wires) {
        state.SkipWithError("Invalid number of wires.");
    }

    for (auto _ : state) {
        Pennylane::StateVectorManaged<T> sv{num_qubits};

        sv.applyOperation(kernel, "PauliX", {num_wires}, false);

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}


template <class T>
static void applyOperation_MyImplPauliZ(benchmark::State &state, Kernel kernel) {
    const size_t num_qubits = state.range(0);
    const size_t num_wires = state.range(1);

    if (!num_qubits) {
        state.SkipWithError("Invalid number of qubits.");
    }
    if (!num_wires) {
        state.SkipWithError("Invalid number of wires.");
    }

    for (auto _ : state) {
        Pennylane::StateVectorManaged<T> sv{num_qubits};

        sv.applyOperation(kernel, "PauliZ", {num_wires}, false);

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}

template <class T>
static void applyOperation_PauliXLM(benchmark::State &state, Kernel kernel) {
    const size_t num_qubits = state.range(0);
    const size_t num_wires = state.range(1);

    if (!num_qubits) {
        state.SkipWithError("Invalid number of qubits.");
    }
    if (!num_wires) {
        state.SkipWithError("Invalid number of wires.");
    }

    for (auto _ : state) {
        Pennylane::StateVectorManaged<T> sv{num_qubits};

        sv.applyOperation(kernel, "PauliX", {num_wires}, false);

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}

template <class T>
static void applyOperation_PauliZLM(benchmark::State &state, Kernel kernel) {
    const size_t num_qubits = state.range(0);
    const size_t num_wires = state.range(1);

    if (!num_qubits) {
        state.SkipWithError("Invalid number of qubits.");
    }
    if (!num_wires) {
        state.SkipWithError("Invalid number of wires.");
    }

    for (auto _ : state) {
        Pennylane::StateVectorManaged<T> sv{num_qubits};

        sv.applyOperation(kernel, "PauliZ", {num_wires}, false);

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}

BENCHMARK_APPLYOPS(applyOperation_PauliXLM, float, LM, Kernel::LM)
    ->ArgsProduct({
        benchmark::CreateDenseRange(6, 30, /*step=*/6), // num_qubits
        benchmark::CreateDenseRange(2, 4, /*step=*/1),  // num_wires
    });

BENCHMARK_APPLYOPS(applyOperation_MyImplPauliX, float, MyKernel, Kernel::MyKernel)
    ->ArgsProduct({
        benchmark::CreateDenseRange(6, 30, /*step=*/6), // num_qubits
        benchmark::CreateDenseRange(2, 4, /*step=*/1),  // num_wires
    });

BENCHMARK_APPLYOPS(applyOperation_PauliZLM, float, LM, Kernel::LM)
    ->ArgsProduct({
        benchmark::CreateDenseRange(6, 30, /*step=*/6), // num_qubits
        benchmark::CreateDenseRange(2, 4, /*step=*/1),  // num_wires
    });

BENCHMARK_APPLYOPS(applyOperation_MyImplPauliZ, float, MyKernel, Kernel::MyKernel)
    ->ArgsProduct({
        benchmark::CreateDenseRange(6, 30, /*step=*/6), // num_qubits
        benchmark::CreateDenseRange(2, 4, /*step=*/1),  // num_wires
    });
