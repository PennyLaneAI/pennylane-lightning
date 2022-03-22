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

constexpr size_t num_gntrs = 32;

//***********************************************************************//
//                            Generators
//***********************************************************************//

/**
 * @brief Benchmark function for gate operation with a fixed number of wires.
 */
template <class T>
static void applyGenerator_GntrOp(benchmark::State &state, Kernel kernel,
                                  Gates::GeneratorOperation gntr_op) {
    const size_t num_qubits = state.range(0);
    const auto gntr_name =
        std::string{Util::lookup(Gates::Constant::generator_names, gntr_op)};

    const auto num_wires = [&]() {
        if (Util::array_has_elt(Gates::Constant::multi_qubit_generators, gntr_op)) {
            return static_cast<size_t>(state.range(1));
        } else {
            return Util::lookup(Gates::Constant::generator_wires, gntr_op);
        }
    }();

    if (!num_qubits) {
        state.SkipWithError("Invalid number of qubits.");
    }

    if (!num_wires) {
        state.SkipWithError("Invalid number of wires.");
    }

    std::random_device rd;
    std::mt19937_64 eng(rd());

    std::vector<std::vector<size_t>> wires;

    wires.reserve(num_gntrs);

    for (size_t i = 0; i < num_gntrs; i++) {
        wires.emplace_back(generateDistinctWires(eng, num_qubits, num_wires));
    }

    const auto gntr_name_without_suffix = gntr_name.substr(9);

    for (auto _ : state) {
        Pennylane::StateVectorManaged<T> sv{num_qubits};

        for (size_t g = 0; g < num_gntrs; g++) {
            [[maybe_unused]] const auto scale = 
                sv.applyGenerator(kernel, gntr_name_without_suffix, wires[g], false);
        }

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}

template <class T, class GateImplementation> void registerAllGenerators() {
    for (const auto gntr_op : GateImplementation::implemented_generators) {
        const auto gntr_name =
            std::string(Util::lookup(Gates::Constant::generator_names, gntr_op));
        const std::string name = std::string("applyGenerator_") + gntr_name +
                                 "<" + std::string(precision_to_str<T>) + ">/" +
                                 std::string(GateImplementation::name);
        if (Util::array_has_elt(Gates::Constant::multi_qubit_generators, gntr_op)) {
            benchmark::RegisterBenchmark(name.c_str(), applyGenerator_GntrOp<T>,
                                         GateImplementation::kernel_id, gntr_op)
                ->ArgsProduct({
                    benchmark::CreateDenseRange(6, 24,
                                                /*step=*/2), // num_qubits
                    benchmark::CreateRange(2, 4, /*mul=*/2), // num_wires
                });
        } else {
            benchmark::RegisterBenchmark(name.c_str(), applyGenerator_GntrOp<T>,
                                         GateImplementation::kernel_id, gntr_op)
                ->ArgsProduct({
                    benchmark::CreateDenseRange(6, 24,
                                                /*step=*/2), // num_qubits
                });
        }
    }
}

template <typename TypeList, std::size_t... Is>
void registerBenchmarkForAllKernelsHelper(std::index_sequence<Is...>) {
    (registerAllGenerators<float, Util::getNthType<TypeList, Is>>(), ...);
    (registerAllGenerators<double, Util::getNthType<TypeList, Is>>(), ...);
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
