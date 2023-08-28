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
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Constant.hpp"
#include "StateVectorManagedCPU.hpp"

#include "Bench_Utils.hpp"

using Kernel = Pennylane::Gates::KernelType;

using OpName = std::string;
using NumWires = size_t;
using NumParams = size_t;
using SerializedOp = std::tuple<OpName, NumWires, NumParams>;

/**
 * @brief Get all gates except multi-qubit gates.
 *
 * @return std::unordered_map<OpName, std::pair<NumWires, NumParams>>
 */
static inline auto getLightningGates()
    -> std::unordered_map<OpName, std::pair<NumWires, NumParams>> {
    namespace Constant = Pennylane::Gates::Constant;

    std::unordered_map<OpName, std::pair<NumWires, NumParams>> all_gates;
    for (const auto &[gate_op, gate_name] : Constant::gate_names) {
        if (!Pennylane::Util::array_has_elt(Constant::multi_qubit_gates,
                                            gate_op)) {
            all_gates.emplace(
                gate_name,
                std::make_pair(static_cast<size_t>(Pennylane::Util::lookup(
                                   Constant::gate_wires, gate_op)),
                               static_cast<size_t>(Pennylane::Util::lookup(
                                   Constant::gate_num_params, gate_op))));
        }
    }
    return all_gates;
}

/**
 * @brief Benchmark getLightningGates.
 */
static void availableLightningGates(benchmark::State &state) {
    for (auto _ : state) {
        auto res = getLightningGates();
        benchmark::DoNotOptimize(res.size());
    }
}
BENCHMARK(availableLightningGates);

/**
 * @brief Serialize a list of operations.
 *
 * @return std::vector<std::tuple<OpName, NumWires, NumParams>>
 */
static inline auto
serializeOperationsFromStrings(benchmark::State &state,
                               const std::vector<OpName> op_names) {
    std::vector<SerializedOp> ops;
    const auto all_gates = getLightningGates();
    for (auto &name : op_names) {
        auto gate_info_pair = all_gates.find(name);
        if (gate_info_pair == all_gates.end()) {
            state.SkipWithError("Invalid operation name.");
        }
        ops.emplace_back(std::make_tuple(std::move(name),
                                         (*gate_info_pair).second.first,
                                         (*gate_info_pair).second.second));
    }
    return ops;
}

/**
 * @brief Benchmark serializeOperationsFromStrings.
 *
 * @param op_names List of operations' names.
 */
static void serializeOps(benchmark::State &state,
                         const std::vector<OpName> op_names) {
    for (auto _ : state) {
        const auto res = serializeOperationsFromStrings(state, op_names);
        benchmark::DoNotOptimize(res.size());
    }
}
BENCHMARK_CAPTURE(serializeOps, ops_RXYZ, {"RX", "RY", "RZ"});
BENCHMARK_CAPTURE(serializeOps, ops_CRXYZ, {"CRX", "CRY", "CRZ"});
BENCHMARK_CAPTURE(serializeOps, ops_PauliXYZ, {"PauliX", "PauliY", "PauliZ"});
BENCHMARK_CAPTURE(serializeOps, ops_all, {"PauliX",     "PauliY",
                                          "PauliZ",     "Hadamard",
                                          "S",          "T",
                                          "RX",         "RY",
                                          "RZ",         "Rot",
                                          "PhaseShift", "CNOT",
                                          "SWAP",       "ControlledPhaseShift",
                                          "CRX",        "CRY",
                                          "CRZ",        "CRot",
                                          "Toffoli",    "CSWAP"});

//***********************************************************************//
//                            applyOperation
//***********************************************************************//

/**
 * @brief Benchmark applyOperation in PennyLane-Lightning .
 *
 * @tparam T Floating point precision type.
 * @param kernel Pennylane::Gates::KernelType.
 * @param op_names List of operations' names.
 */
template <class T>
static void applyOperations_RandOps(benchmark::State &state,
                                    const Kernel kernel,
                                    const std::vector<OpName> op_names) {
    if (op_names.empty()) {
        state.SkipWithError("Invalid list of operations.");
    }

    const size_t num_gates = state.range(0);
    const size_t num_qubits = state.range(1);

    if (!num_gates) {
        state.SkipWithError("Invalid number of gates.");
    }

    if (!num_qubits) {
        state.SkipWithError("Invalid number of qubits.");
    }

    const auto ops = serializeOperationsFromStrings(state, op_names);

    std::random_device rd;
    std::mt19937_64 eng(rd());

    std::uniform_int_distribution<size_t> gate_distr(0, ops.size() - 1);
    std::uniform_int_distribution<size_t> inv_distr(0, 1);
    std::uniform_real_distribution<T> param_distr(0.0, 2 * M_PI);
    std::uniform_int_distribution<size_t> wire_distr(0, num_qubits - 1);

    auto param_generator = [&param_distr, &eng]() { return param_distr(eng); };

    std::vector<std::string_view> rand_gate_names;
    std::vector<std::vector<size_t>> rand_gate_wires;
    std::vector<std::vector<T>> rand_gate_params;

    std::vector<bool> rand_inverses;

    for (size_t i = 0; i < num_gates; i++) {
        size_t wire_start_idx = wire_distr(eng);
        const auto &[op_name, n_wires, n_params] = ops[gate_distr(eng)];

        rand_gate_names.emplace_back(op_name);
        rand_gate_wires.emplace_back(
            generateNeighboringWires(wire_start_idx, num_qubits, n_wires));

        std::vector<T> params(n_params);
        std::generate(params.begin(), params.end(), param_generator);
        rand_gate_params.emplace_back(std::move(params));

        rand_inverses.emplace_back(static_cast<bool>(inv_distr(eng)));
    }

    for (auto _ : state) {
        Pennylane::StateVectorManagedCPU<T> sv{num_qubits};

        for (size_t g = 0; g < num_gates; g++) {
            sv.applyOperation(kernel, OpName(rand_gate_names[g]),
                              rand_gate_wires[g], rand_inverses[g],
                              rand_gate_params[g]);
        }

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}

/* RXYZ */
BENCHMARK_APPLYOPS(applyOperations_RandOps, float, LM_RXYZ, Kernel::LM,
                   {"RX", "RY", "RZ"})
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
    });

BENCHMARK_APPLYOPS(applyOperations_RandOps, float, PI_RXYZ, Kernel::PI,
                   {"RX", "RY", "RZ"})
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
    });

BENCHMARK_APPLYOPS(applyOperations_RandOps, double, LM_RXYZ, Kernel::LM,
                   {"RX", "RY", "RZ"})
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
    });

BENCHMARK_APPLYOPS(applyOperations_RandOps, double, PI_RXYZ, Kernel::PI,
                   {"RX", "RY", "RZ"})
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
    });

/* PauliXYZ */
BENCHMARK_APPLYOPS(applyOperations_RandOps, double, LM_PauliXYZ, Kernel::LM,
                   {"PauliX", "PauliY", "PauliZ"})
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
    });

BENCHMARK_APPLYOPS(applyOperations_RandOps, double, PI_PauliXYZ, Kernel::PI,
                   {"PauliX", "PauliY", "PauliZ"})
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
    });

BENCHMARK_APPLYOPS(applyOperations_RandOps, float, LM_PauliXYZ, Kernel::LM,
                   {"PauliX", "PauliY", "PauliZ"})
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
    });

BENCHMARK_APPLYOPS(applyOperations_RandOps, float, PI_PauliXYZ, Kernel::PI,
                   {"PauliX", "PauliY", "PauliZ"})
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
    });

/* From All Gates */
BENCHMARK_APPLYOPS(applyOperations_RandOps, float, LM_all, Kernel::LM,
                   {"PauliX", "PauliY", "PauliZ", "Hadamard", "S", "T", "RX",
                    "RY", "RZ", "Rot", "PhaseShift", "CNOT", "SWAP",
                    "ControlledPhaseShift", "CRX", "CRY", "CRZ"})
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
    });

BENCHMARK_APPLYOPS(applyOperations_RandOps, float, PI_all, Kernel::PI,
                   {"PauliX", "PauliY", "PauliZ", "Hadamard", "S", "T", "RX",
                    "RY", "RZ", "Rot", "PhaseShift", "CNOT", "SWAP",
                    "ControlledPhaseShift", "CRX", "CRY", "CRZ"})
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
    });

BENCHMARK_APPLYOPS(applyOperations_RandOps, double, LM_all, Kernel::LM,
                   {"PauliX", "PauliY", "PauliZ", "Hadamard", "S", "T", "RX",
                    "RY", "RZ", "Rot", "PhaseShift", "CNOT", "SWAP",
                    "ControlledPhaseShift", "CRX", "CRY", "CRZ"})
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
    });

BENCHMARK_APPLYOPS(applyOperations_RandOps, double, PI_all, Kernel::PI,
                   {"PauliX", "PauliY", "PauliZ", "Hadamard", "S", "T", "RX",
                    "RY", "RZ", "Rot", "PhaseShift", "CNOT", "SWAP",
                    "ControlledPhaseShift", "CRX", "CRY", "CRZ"})
    ->ArgsProduct({
        benchmark::CreateRange(8, 64, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 24, /*step=*/2), // num_qubits
    });
