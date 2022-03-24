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
#include "AllOperations.hpp"

using namespace Pennylane;

template <class T, class GateImplementation> void registerAllGenerators() {
    for (const auto gntr_op : GateImplementation::implemented_generators) {
        const auto gntr_name = std::string(
            Util::lookup(Gates::Constant::generator_names, gntr_op));
        const std::string name = gntr_name + "<" +
                                 std::string(precision_to_str<T>) + ">/" +
                                 std::string(GateImplementation::name);
        if (Util::array_has_elt(Gates::Constant::multi_qubit_generators,
                                gntr_op)) {
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

template <class T, class GateImplementation> void registerAllGates() {
    for (const auto gate_op : GateImplementation::implemented_gates) {
        const auto gate_name =
            std::string(Util::lookup(Gates::Constant::gate_names, gate_op));
        const std::string name = gate_name + "<" +
                                 std::string(precision_to_str<T>) + ">/" +
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

template <class T, class GateImplementation> void registerAllMatrices() {
    if constexpr (Util::array_has_elt(GateImplementation::implemented_matrices,
                                      Gates::MatrixOperation::SingleQubitOp)) {
        std::string name = "SingleQubitOp<" + std::string(precision_to_str<T>) +
                           ">/" + std::string(GateImplementation::name);
        benchmark::RegisterBenchmark(name.c_str(), applyMatrix<T>,
                                     GateImplementation::kernel_id)
            ->ArgsProduct(
                {benchmark::CreateDenseRange(6, 24,
                                             /*step=*/2), // num_qubits
                 {1}});
    }
    if constexpr (Util::array_has_elt(GateImplementation::implemented_matrices,
                                      Gates::MatrixOperation::TwoQubitOp)) {
        std::string name = "TwoQubitOp<" + std::string(precision_to_str<T>) +
                           ">/" + std::string(GateImplementation::name);
        benchmark::RegisterBenchmark(name.c_str(), applyMatrix<T>,
                                     GateImplementation::kernel_id)
            ->ArgsProduct(
                {benchmark::CreateDenseRange(6, 24,
                                             /*step=*/2), // num_qubits
                 {2}});
    }
    if constexpr (Util::array_has_elt(GateImplementation::implemented_matrices,
                                      Gates::MatrixOperation::MultiQubitOp)) {
        std::string name = "MultiQubitOp<" + std::string(precision_to_str<T>) +
                           ">/" + std::string(GateImplementation::name);
        benchmark::RegisterBenchmark(name.c_str(), applyMatrix<T>,
                                     GateImplementation::kernel_id)
            ->ArgsProduct(
                {benchmark::CreateDenseRange(6, 24,
                                             /*step=*/2), // num_qubits
                 {3, 4, 5}});
    }
}

template <typename TypeList, std::size_t... Is>
void registerAllKernelsHelper(std::index_sequence<Is...>) {
    /* Gates */
    (registerAllGates<float, Util::getNthType<TypeList, Is>>(), ...);
    (registerAllGates<double, Util::getNthType<TypeList, Is>>(), ...);
    /* Generators */
    (registerAllGenerators<float, Util::getNthType<TypeList, Is>>(), ...);
    (registerAllGenerators<double, Util::getNthType<TypeList, Is>>(), ...);
    /* Matrices */
    (registerAllMatrices<float, Util::getNthType<TypeList, Is>>(), ...);
    (registerAllMatrices<double, Util::getNthType<TypeList, Is>>(), ...);
}

void registerAllKernels() {
    registerAllKernelsHelper<AvailableKernels>(
        std::make_index_sequence<Util::length<AvailableKernels>()>());
}

int main(int argc, char **argv) {
    addCompileInfo();
    addRuntimeInfo();
    registerAllKernels();

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
