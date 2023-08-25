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
#include "DynamicDispatcher.hpp"
#include "KernelType.hpp"

using namespace Pennylane;

template <class T> void registerAllGenerators(Gates::KernelType kernel) {
    const auto &dispatcher = DynamicDispatcher<T>::getInstance();
    for (const auto gntr_op :
         dispatcher.registeredGeneratorsForKernel(kernel)) {
        const auto gntr_name = std::string(
            Util::lookup(Gates::Constant::generator_names, gntr_op));
        const std::string name = gntr_name + "<" +
                                 std::string(precision_to_str<T>) + ">/" +
                                 dispatcher.getKernelName(kernel);
        if (Util::array_has_elt(Gates::Constant::multi_qubit_generators,
                                gntr_op)) {
            benchmark::RegisterBenchmark(name.c_str(), applyGenerator_GntrOp<T>,
                                         kernel, gntr_op)
                ->ArgsProduct({
                    benchmark::CreateDenseRange(6, 24,
                                                /*step=*/2), // num_qubits
                    benchmark::CreateRange(2, 4, /*mul=*/2), // num_wires
                });
        } else {
            benchmark::RegisterBenchmark(name.c_str(), applyGenerator_GntrOp<T>,
                                         kernel, gntr_op)
                ->ArgsProduct({
                    benchmark::CreateDenseRange(6, 24,
                                                /*step=*/2), // num_qubits
                });
        }
    }
}

template <class T> void registerAllGates(Gates::KernelType kernel) {
    const auto &dispatcher = DynamicDispatcher<T>::getInstance();
    for (const auto gate_op : dispatcher.registeredGatesForKernel(kernel)) {
        const auto gate_name =
            std::string(Util::lookup(Gates::Constant::gate_names, gate_op));
        const std::string name = gate_name + "<" +
                                 std::string(precision_to_str<T>) + ">/" +
                                 dispatcher.getKernelName(kernel);
        if (Util::array_has_elt(Gates::Constant::multi_qubit_gates, gate_op)) {
            benchmark::RegisterBenchmark(name.c_str(), applyOperation_GateOp<T>,
                                         kernel, gate_op)
                ->ArgsProduct({
                    benchmark::CreateDenseRange(6, 24,
                                                /*step=*/2), // num_qubits
                    benchmark::CreateRange(2, 4, /*mul=*/2), // num_wires
                });
        } else {
            benchmark::RegisterBenchmark(name.c_str(), applyOperation_GateOp<T>,
                                         kernel, gate_op)
                ->ArgsProduct({
                    benchmark::CreateDenseRange(6, 24,
                                                /*step=*/2), // num_qubits
                });
        }
    }
}

template <class T> void registerAllMatrices(Gates::KernelType kernel) {
    const auto &dispatcher = DynamicDispatcher<T>::getInstance();
    auto registeredMatrices = dispatcher.registeredMatricesForKernel(kernel);

    if (registeredMatrices.contains(Gates::MatrixOperation::SingleQubitOp)) {
        std::string name = "SingleQubitOp<" + std::string(precision_to_str<T>) +
                           ">/" + dispatcher.getKernelName(kernel);
        benchmark::RegisterBenchmark(name.c_str(), applyMatrix<T>, kernel)
            ->ArgsProduct(
                {benchmark::CreateDenseRange(6, 24,
                                             /*step=*/2), // num_qubits
                 {1}});
    }
    if (registeredMatrices.contains(Gates::MatrixOperation::TwoQubitOp)) {
        std::string name = "TwoQubitOp<" + std::string(precision_to_str<T>) +
                           ">/" + dispatcher.getKernelName(kernel);
        benchmark::RegisterBenchmark(name.c_str(), applyMatrix<T>, kernel)
            ->ArgsProduct(
                {benchmark::CreateDenseRange(6, 24,
                                             /*step=*/2), // num_qubits
                 {2}});
    }
    if (registeredMatrices.contains(Gates::MatrixOperation::MultiQubitOp)) {
        std::string name = "MultiQubitOp<" + std::string(precision_to_str<T>) +
                           ">/" + dispatcher.getKernelName(kernel);
        benchmark::RegisterBenchmark(name.c_str(), applyMatrix<T>, kernel)
            ->ArgsProduct(
                {benchmark::CreateDenseRange(6, 24,
                                             /*step=*/2), // num_qubits
                 {3, 4, 5}});
    }
}

template <typename T> void registerAllKernels() {
    const auto &dispatcher = DynamicDispatcher<T>::getInstance();
    for (const auto kernel : dispatcher.registeredKernels()) {
        registerAllGenerators<T>(kernel);
        registerAllGates<T>(kernel);
        registerAllMatrices<T>(kernel);
    }
}

int main(int argc, char **argv) {
    addCompileInfo();
    addRuntimeInfo();
    registerAllKernels<float>();
    registerAllKernels<double>();

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
