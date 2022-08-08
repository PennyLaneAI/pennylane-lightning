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
#pragma once
#include "Bench_Utils.hpp"

#include "Constant.hpp"
#include "DynamicDispatcher.hpp"
#include "LinearAlgebra.hpp"
#include "StateVectorManagedCPU.hpp"

#include <algorithm>
#include <random>
#include <vector>

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
 * @brief Benchmark function for gate operations
 *
 * @tparam T Floating point precision type.
 */
template <class T, size_t num_gates = 32>
static void applyOperation_GateOp(benchmark::State &state,
                                  Pennylane::Gates::KernelType kernel,
                                  Pennylane::Gates::GateOperation gate_op) {
    using namespace Pennylane;
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
        Pennylane::StateVectorManagedCPU<T> sv{num_qubits};

        for (size_t g = 0; g < num_gates; g++) {
            sv.applyOperation(kernel, gate_name, wires[g], false, params[g]);
        }

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}

//***********************************************************************//
//                            Generators
//***********************************************************************//

/**
 * @brief Benchmark function for generator operations
 *
 * @tparam T Floating point precision type.
 */
template <class T, size_t num_gntrs = 32>
static void
applyGenerator_GntrOp(benchmark::State &state,
                      Pennylane::Gates::KernelType kernel,
                      Pennylane::Gates::GeneratorOperation gntr_op) {
    using namespace Pennylane;
    const size_t num_qubits = state.range(0);
    const auto gntr_name =
        std::string{Util::lookup(Gates::Constant::generator_names, gntr_op)};

    const auto num_wires = [&]() {
        if (Util::array_has_elt(Gates::Constant::multi_qubit_generators,
                                gntr_op)) {
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
        Pennylane::StateVectorManagedCPU<T> sv{num_qubits};

        for (size_t g = 0; g < num_gntrs; g++) {
            [[maybe_unused]] const auto scale = sv.applyGenerator(
                kernel, gntr_name_without_suffix, wires[g], false);
        }

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}

//***********************************************************************//
//                            Matrices
//***********************************************************************//

/**
 * @brief Benchmark function for matrices
 *
 * @tparam T Floating point precision type.
 */
template <class T, size_t num_matrices = 32>
static void applyMatrix(benchmark::State &state,
                        Pennylane::Gates::KernelType kernel) {
    using namespace Pennylane;
    const size_t num_qubits = state.range(0);
    const auto num_wires = static_cast<size_t>(state.range(1));

    if (!num_qubits) {
        state.SkipWithError("Invalid number of qubits.");
    }

    if (!num_wires) {
        state.SkipWithError("Invalid number of wires.");
    }

    std::random_device rd;
    std::mt19937_64 eng(rd());

    std::vector<std::vector<size_t>> wires;
    std::vector<std::vector<std::complex<T>>> matrices;
    wires.reserve(num_matrices);
    matrices.reserve(num_matrices);

    for (size_t i = 0; i < num_matrices; i++) {
        wires.emplace_back(generateDistinctWires(eng, num_qubits, num_wires));
        matrices.emplace_back(Util::randomUnitary<T>(eng, num_wires));
    }

    for (auto _ : state) {
        Pennylane::StateVectorManagedCPU<T> sv{num_qubits};

        for (size_t g = 0; g < num_matrices; g++) {
            sv.applyMatrix(kernel, matrices[g], wires[g], false);
        }

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}
