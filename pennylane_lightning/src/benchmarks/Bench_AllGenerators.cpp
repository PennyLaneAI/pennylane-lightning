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

using namespace Pennylane;
using Kernel = Pennylane::Gates::KernelType;

//***********************************************************************//
//                            Generators
//***********************************************************************//

/**
 * @brief Benchmark function for gate operation with a fixed number of wires.
 */
template <class T>
static void applyOperation_GntrOp(benchmark::State &state, Kernel kernel,
                                  Gates::GeneratorOperation gntr_op) {
    const size_t num_gates = state.range(0);
    const size_t num_qubits = state.range(1);

    const auto num_wires = Util::lookup(Gates::Constant::gate_wires, gate_op);
    const auto gate_name =
        std::string{Util::lookup(Gates::Constant::gate_names, gate_op)};

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

    std::vector<std::vector<size_t>> wires;
    std::vector<std::vector<T>> params;

    wires.reserve(num_gates);
    params.reserve(num_gates);

    for (size_t i = 0; i < num_gates; i++) {
        wires.emplace_back(generateDistinctWires(eng, num_qubits, num_wires));
    }

    for (auto _ : state) {
        Pennylane::StateVectorManaged<T> sv{num_qubits};

        for (size_t g = 0; g < num_gates; g++) {
            sv.applyGenerator(kernel, gate_name, wires[g], false, params[g]);
        }

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}

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

template <class T, class GateImplementation> void registerNonMultiQubitGates() {
    for (const auto gate_op : GateImplementation::implemented_gates) {
        if (Util::array_has_elt(Gates::Constant::multi_qubit_gates, gate_op)) {
            continue;
        }
        const auto gate_name =
            std::string(Util::lookup(Gates::Constant::gate_names, gate_op));
        const std::string name = std::string("applyOperation_") + gate_name +
                                 "<" + std::string(precision_to_str<T>) + ">/" +
                                 std::string(GateImplementation::name);
        benchmark::RegisterBenchmark(name.c_str(), applyOperation_GateOp<T>,
                                     GateImplementation::kernel_id, gate_op)
            ->ArgsProduct({
                benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
                benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
            });
    }
}

template <typename TypeList, std::size_t... Is>
void registerBenchmarkForAllKernelsHelper(std::index_sequence<Is...>) {
    (registerNonMultiQubitGates<
         float, typename Util::getNthType<TypeList, Is>::Type>(),
     ...);
    (registerNonMultiQubitGates<
         double, typename Util::getNthType<TypeList, Is>::Type>(),
     ...);
}
void registerBenchmarkForAllKernels() {
    registerBenchmarkForAllKernelsHelper<AvailableKernels>(
        std::make_index_sequence<Util::length<AvailableKernels>()>());
}

int main(int argc, char **argv) {
    registerBenchmarkForAllKernels();

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
