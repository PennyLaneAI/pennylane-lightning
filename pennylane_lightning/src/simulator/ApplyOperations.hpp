// Copyright 2021 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file
 * Defines 
 */

#pragma once

#include <complex>
#include <string>
#include <unordered_map>
#include <vector>
#include <variant>
#include <functional>


#define PENNYLANE_GATEOP_PAIR_PARAMS0(GATE_NAME)                                           \
    {#GATE_NAME, [](CFP_t* data, size_t num_qubits, const std::vector<size_t>& wires,      \
                bool inverse, const std::vector<fp_t>& params) {                           \
            assert(params.empty());                                                        \
            (void)params;                                                                  \
            GateOperationType<fp_t>::apply##GATE_NAME(data, num_qubits, wires, inverse);   \
        }}
#define PENNYLANE_GATEOP_PAIR_PARAMS1(GATE_NAME)                                           \
    {#GATE_NAME, [](CFP_t* data, size_t num_qubits, const std::vector<size_t>& wires,      \
                bool inverse, const std::vector<fp_t>& params) {                           \
            assert(params.size() == 1);                                                    \
            (void)params;                                                                  \
            GateOperationType<fp_t>::apply##GATE_NAME(data, num_qubits, wires, inverse,    \
                    params[0]);                                                            \
        }}
#define PENNYLANE_GATEOP_PAIR_PARAMS2(GATE_NAME)                                           \
    {#GATE_NAME, [](CFP_t* data, size_t num_qubits, const std::vector<size_t>& wires,      \
                bool inverse, const std::vector<fp_t>& params) {                           \
            assert(params.size() == 2);                                                    \
            (void)params;                                                                  \
            GateOperationType<fp_t>::apply##GATE_NAME(data, num_qubits, wires, inverse,    \
                    params[0], params[1]);                                                 \
        }}
#define PENNYLANE_GATEOP_PAIR_PARAMS3(GATE_NAME)                                           \
    {#GATE_NAME, [](CFP_t* data, size_t num_qubits, const std::vector<size_t>& wires,      \
                bool inverse, const std::vector<fp_t>& params) {                           \
            assert(params.size() == 3);                                                    \
            (void)params;                                                                  \
            GateOperationType<fp_t>::apply##GATE_NAME(data, num_qubits, wires, inverse,    \
                    params[0], params[1], params[2]);                                      \
        }}
#define PENNYLANE_GATEOP_PAIR(GATE_NAME, NUM_PARAMS) \
    PENNYLANE_GATEOP_PAIR_PARAMS##NUM_PARAMS(GATE_NAME)

namespace Pennylane {

/* forward declaration */
template<typename fp_t, template<class> class GateOperationType, class Derived>
class StateVectorBase;

template<typename fp_t, template <class> class GateOperationType>
class ApplyOperations {
  public:
    using scalar_type_t = fp_t;
    using CFP_t = std::complex<scalar_type_t>;

  private:
    using Func = std::function<
        void(CFP_t* /*data*/, size_t /*num_qubits*/, const std::vector<size_t>& /*wires*/,
            bool /*inverse*/, const std::vector<fp_t>& /*params*/)>;
    const std::unordered_map<std::string, size_t> gate_wires_{
        {"PauliX", 1},   {"PauliY", 1},     {"PauliZ", 1},
        {"Hadamard", 1}, {"T", 1},          {"S", 1},
        {"RX", 1},       {"RY", 1},         {"RZ", 1},
        {"Rot", 1},      {"PhaseShift", 1}, {"ControlledPhaseShift", 2},
        {"CNOT", 2},     {"SWAP", 2},       {"CZ", 2},
        {"CRX", 2},      {"CRY", 2},        {"CRZ", 2},
        {"CRot", 2},     {"CSWAP", 3},      {"Toffoli", 3}};

    std::unordered_map<std::string, Func> gates_;
    ApplyOperations() {
        // single-qubit gates
        gates_.emplace(PENNYLANE_GATEOP_PAIR(PauliX, 0));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(PauliY, 0));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(PauliZ, 0));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(Hadamard, 0));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(T, 0));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(S, 0));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(RX, 1));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(RY, 1));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(RZ, 1));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(PhaseShift, 1));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(Rot, 3));
        // two-qubit gates
        gates_.emplace(PENNYLANE_GATEOP_PAIR(ControlledPhaseShift, 1));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(CNOT, 0));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(SWAP, 0));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(CZ, 0));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(CRX, 1));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(CRY, 1));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(CRZ, 1));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(CRot, 3));
        // three-qubit gates
        gates_.emplace(PENNYLANE_GATEOP_PAIR(CSWAP, 0));
        gates_.emplace(PENNYLANE_GATEOP_PAIR(Toffoli, 0));
    }

  public:
    static ApplyOperations& getInstance() {
        static ApplyOperations singleton;
        return singleton;
    }
    
    template<typename FunctionType>
    void registerGateOperation(std::string opName, FunctionType&& func) {
        gates_.emplace(std::move(opName), func);
    }

    /**
     * @berief call the corresponding operation function from GateOperations
     */
    template<class Derived>
    void applyOperation(StateVectorBase<fp_t, GateOperationType, Derived>& sv,
            const std::string& opName, const std::vector<size_t>& wires,
            bool inverse, const std::vector<fp_t>& params) {

        const auto iter = gates_.find(opName);
        if (iter == gates_.end()) {
            throw std::invalid_argument("Cannot find a gate with a given name \"" 
                    + opName + "\".");
        }

        if (const auto requiredWires = gate_wires_.at(opName); requiredWires != wires.size())
        {
            throw std::invalid_argument(std::string("The supplied gate requires ") +
                                        std::to_string(requiredWires) + " wires, but " +
                                        std::to_string(wires.size()) +
                                        " were supplied.");
        }
        (iter->second)(sv.getData(), sv.getNumQubits(), wires, inverse, params);
    }

    template<class Derived>
    void applyOperations(StateVectorBase<fp_t, GateOperationType, Derived>& sv,
                         const std::vector<std::string> &ops,
                         const std::vector<std::vector<size_t>> &wires,
                         const std::vector<bool> &inverse,
                         const std::vector<std::vector<fp_t>> &params) {
        const size_t numOperations = ops.size();
        if (numOperations != wires.size() || numOperations != params.size()) {
            throw std::invalid_argument(
                "Invalid arguments: number of operations, wires, and "
                "parameters must all be equal");
        }

        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(sv, ops[i], wires[i], inverse[i], params[i]);
        }
    }

    template<class Derived>
    void applyOperations(StateVectorBase<fp_t, GateOperationType, Derived>& sv,
                         const std::vector<std::string> &ops,
                         const std::vector<std::vector<size_t>> &wires,
                         const std::vector<bool> &inverse) {
        const size_t numOperations = ops.size();
        if (numOperations != wires.size()) {
            throw std::invalid_argument(
                "Invalid arguments: number of operations, wires, and "
                "parameters must all be equal");
        }

        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(sv, ops[i], wires[i], inverse[i], {});
        }
    }
};
} // namespace Pennylane
