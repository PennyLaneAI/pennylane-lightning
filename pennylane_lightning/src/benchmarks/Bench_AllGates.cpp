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

constexpr size_t num_gates = 32;

//***********************************************************************//
//                            Gates
//***********************************************************************//
template <class ParamT, class RandomEngine>
auto createParams(RandomEngine &re, size_t num_params) -> std::vector<ParamT> {
    std::uniform_real_distribution<ParamT> param_distr(-M_PI, M_PI);
    std::vector<ParamT> res;
    res.reserve(num_params);
    for (size_t i = 0; i < num_params; i++) {
        res.push_back(param_distr(re));
    }
    return res;
}

/**
 * @brief Benchmark function for gate operation with a fixed number of wires.
 *
 * @tparam T Floating point precision type.
 */
template <class T>
static void applyOperation_GateOp(benchmark::State &state, Kernel kernel,
                                  Gates::GateOperation gate_op) {
    const size_t num_qubits = state.range(0);

    const auto num_params =
        Util::lookup(Gates::Constant::gate_num_params, gate_op);
    const auto gate_name =
        std::string{Util::lookup(Gates::Constant::gate_names, gate_op)};
    const auto num_wires = [&]() {
        if (Util::array_has_elt(Gates::Constant::multi_qubit_gates, gate_op)) {
            return static_cast<size_t>(state.range(1));
        } else {
            return Util::lookup(Gates::Constant::gate_wires, gate_op);
        }
    }();

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

    for (size_t i = 0; i < num_gates; i++) {
        params.emplace_back(createParams<T>(eng, num_params));
    }

    for (auto _ : state) {
        Pennylane::StateVectorManaged<T> sv{num_qubits};

        for (size_t g = 0; g < num_gates; g++) {
            sv.applyOperation(kernel, gate_name, wires[g], false, params[g]);
        }

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}

template <class T, class GateImplementation> void registerNonMultiQubitGates() {
    for (const auto gate_op : GateImplementation::implemented_gates) {
        const auto gate_name =
            std::string(Util::lookup(Gates::Constant::gate_names, gate_op));
        const std::string name = std::string("applyOperation_") + gate_name +
                                 "<" + std::string(precision_to_str<T>) + ">/" +
                                 std::string(GateImplementation::name);
        if (Util::array_has_elt(Gates::Constant::multi_qubit_gates, gate_op)) {
            benchmark::RegisterBenchmark(name.c_str(), applyOperation_GateOp<T>,
                                         GateImplementation::kernel_id, gate_op)
                ->ArgsProduct({
                    benchmark::CreateDenseRange(6, 24,
                                                /*step=*/2), // num_qubits
                    benchmark::CreateRange(2, 4, /*mul=*/2), // num_wires
                });
        } else {
            benchmark::RegisterBenchmark(name.c_str(), applyOperation_GateOp<T>,
                                         GateImplementation::kernel_id, gate_op)
                ->ArgsProduct({
                    benchmark::CreateDenseRange(6, 24,
                                                /*step=*/2), // num_qubits
                });
        }
    }
}

template <typename TypeList, std::size_t... Is>
void registerBenchmarkForAllKernelsHelper(std::index_sequence<Is...>) {
    (registerNonMultiQubitGates<float, Util::getNthType<TypeList, Is>>(), ...);
    (registerNonMultiQubitGates<double, Util::getNthType<TypeList, Is>>(), ...);
}

void registerBenchmarkForAllKernels() {
    registerBenchmarkForAllKernelsHelper<AvailableKernels>(
        std::make_index_sequence<Util::length<AvailableKernels>()>());
}

int main(int argc, char **argv) {
    addCompileInfo();
    addRuntimeInfo();
    registerBenchmarkForAllKernels();

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
