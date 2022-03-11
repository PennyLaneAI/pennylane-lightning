#include <map>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "Constant.hpp"
#include "StateVectorManaged.hpp"

#include <benchmark/benchmark.h>

using OpName = std::string;
using NumWires = size_t;
using NumParams = size_t;

using SerializedOp = std::tuple<OpName, NumWires, NumParams>;

static inline auto getLightningGates()
    -> std::map<OpName, std::pair<NumWires, NumParams>> {
    // All gates except multi-qubit-gates
    namespace Constant = Pennylane::Gates::Constant;
    std::map<OpName, std::pair<NumWires, NumParams>> all_gates;
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

static void availableLightningGates(benchmark::State &state) {
    for (auto _ : state) {
        auto res = getLightningGates();
        benchmark::DoNotOptimize(res.size());
    }
}
BENCHMARK(availableLightningGates);

static inline auto
serializeOperationsFromStrings(benchmark::State &state,
                               const std::vector<OpName> op_names) {
    std::vector<SerializedOp> ops;
    auto all_gates = getLightningGates();
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

static void serializeOps(benchmark::State &state,
                         const std::vector<OpName> op_names) {
    for (auto _ : state) {
        auto res = serializeOperationsFromStrings(state, op_names);
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

static inline auto generateNeighboringWires(size_t start_idx, size_t num_qubits,
                                            size_t num_wires)
    -> std::vector<size_t> {
    std::vector<size_t> v;
    v.reserve(num_wires);
    for (size_t k = 0; k < num_wires; k++) {
        v.emplace_back((start_idx + k) % num_qubits);
    }
    return v;
}

static void applyOperationsFromRandOps(benchmark::State &state,
                                       Pennylane::Gates::KernelType kernel,
                                       const std::vector<OpName> op_names) {
    if (op_names.empty()) {
        state.SkipWithError("Invalid list of operations.");
    }

    size_t num_gates = state.range(0);
    size_t num_qubits = state.range(1);

    if (!num_gates) {
        state.SkipWithError("Invalid number of gates.");
    }

    if (!num_qubits) {
        state.SkipWithError("Invalid number of qubits.");
    }

    auto ops = serializeOperationsFromStrings(state, op_names);

    std::random_device rd;
    std::mt19937_64 eng(rd());

    std::uniform_int_distribution<size_t> gate_distr(0, ops.size() - 1);
    std::uniform_int_distribution<size_t> inv_distr(0, 1);
    std::uniform_real_distribution<double> param_distr(0.0, 2 * M_PI);
    std::uniform_int_distribution<size_t> wire_distr(0, num_qubits - 1);

    auto param_generator = [&param_distr, &eng]() { return param_distr(eng); };

    std::vector<std::string_view> rand_gate_names;
    std::vector<std::vector<size_t>> rand_gate_wires;
    std::vector<std::vector<double>> rand_gate_params;

    std::vector<bool> rand_inverses;

    for (size_t i = 0; i < num_gates; i++) {
        size_t wire_start_idx = wire_distr(eng);
        const auto &[op_name, n_wires, n_params] = ops[gate_distr(eng)];

        rand_gate_names.emplace_back(op_name);
        rand_gate_wires.emplace_back(
            generateNeighboringWires(wire_start_idx, num_qubits, n_wires));

        std::vector<double> params(n_params);
        std::generate(params.begin(), params.end(), param_generator);
        rand_gate_params.emplace_back(std::move(params));

        rand_inverses.emplace_back(static_cast<bool>(inv_distr(eng)));
    }

    for (auto _ : state) {
        Pennylane::StateVectorManaged<double> sv{num_qubits};

        for (size_t g = 0; g < num_gates; g++) {
            sv.applyOperation(kernel, OpName(rand_gate_names[g]),
                              rand_gate_wires[g], rand_inverses[g],
                              rand_gate_params[g]);
        }

        benchmark::DoNotOptimize(sv.getDataVector()[0]);
        benchmark::DoNotOptimize(sv.getDataVector()[(1 << num_qubits) - 1]);
    }
}
BENCHMARK_CAPTURE(applyOperationsFromRandOps, LM_RXYZ,
                  Pennylane::Gates::KernelType::LM, {"RX", "RY", "RZ"})
    ->RangeMultiplier(2l)
    ->Ranges({{8, 64}, {4, 24}});

BENCHMARK_CAPTURE(applyOperationsFromRandOps, PI_RXYZ,
                  Pennylane::Gates::KernelType::PI, {"RX", "RY", "RZ"})
    ->RangeMultiplier(2l)
    ->Ranges({{8, 64}, {4, 24}});

BENCHMARK_CAPTURE(applyOperationsFromRandOps, LM_PauliXYZ,
                  Pennylane::Gates::KernelType::LM,
                  {"PauliX", "PauliY", "PauliZ"})
    ->RangeMultiplier(2l)
    ->Ranges({{8, 64}, {4, 24}});

BENCHMARK_CAPTURE(applyOperationsFromRandOps, PI_PauliXYZ,
                  Pennylane::Gates::KernelType::PI,
                  {"PauliX", "PauliY", "PauliZ"})
    ->RangeMultiplier(2l)
    ->Ranges({{8, 64}, {4, 24}});

BENCHMARK_CAPTURE(applyOperationsFromRandOps, LM_all,
                  Pennylane::Gates::KernelType::LM,
                  {"PauliX", "PauliY", "PauliZ", "Hadamard", "S", "T", "RX",
                   "RY", "RZ", "Rot", "PhaseShift", "CNOT", "SWAP",
                   "ControlledPhaseShift", "CRX", "CRY", "CRZ"})
    ->RangeMultiplier(2l)
    ->Ranges({{8, 64}, {4, 24}});

BENCHMARK_CAPTURE(applyOperationsFromRandOps, PI_all,
                  Pennylane::Gates::KernelType::PI,
                  {"PauliX", "PauliY", "PauliZ", "Hadamard", "S", "T", "RX",
                   "RY", "RZ", "Rot", "PhaseShift", "CNOT", "SWAP",
                   "ControlledPhaseShift", "CRX", "CRY", "CRZ"})
    ->RangeMultiplier(2l)
    ->Ranges({{8, 64}, {4, 24}});
