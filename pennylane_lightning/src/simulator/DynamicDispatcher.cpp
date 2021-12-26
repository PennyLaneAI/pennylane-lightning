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
#include "DynamicDispatcher.hpp"

#define PENNYLANE_REGISTER_GATE_OP_PARAMS0(GATE_NAME, KERNEL_TYPE)                          \
    dispatcher.registerGateOperation(#GATE_NAME, KERNEL_TYPE,                               \
            [](CFP_t* data, size_t num_qubits, const std::vector<size_t>& wires,            \
               bool inverse, const std::vector<fp_t>& params) {                             \
               assert(params.empty());                                                      \
               SelectGateOps<fp_t, KERNEL_TYPE>::apply##GATE_NAME(data, num_qubits, wires,  \
                                                                  inverse);                 \
            });
#define PENNYLANE_REGISTER_GATE_OP_PARAMS1(GATE_NAME, KERNEL_TYPE)                          \
    dispatcher.registerGateOperation(#GATE_NAME, KERNEL_TYPE,                               \
            [](CFP_t* data, size_t num_qubits, const std::vector<size_t>& wires,            \
               bool inverse, const std::vector<fp_t>& params) {                             \
               assert(params.size() == 1);                                                  \
               SelectGateOps<fp_t, KERNEL_TYPE>::apply##GATE_NAME(data, num_qubits, wires,  \
                                                                  inverse, params[0]);      \
            });
#define PENNYLANE_REGISTER_GATE_OP_PARAMS3(GATE_NAME, KERNEL_TYPE)                          \
    dispatcher.registerGateOperation(#GATE_NAME, KERNEL_TYPE,                               \
            [](CFP_t* data, size_t num_qubits, const std::vector<size_t>& wires,            \
               bool inverse, const std::vector<fp_t>& params) {                             \
               assert(params.size() == 1);                                                  \
               SelectGateOps<fp_t, KERNEL_TYPE>::apply##GATE_NAME(data, num_qubits, wires,  \
                                                                  inverse, params[0],       \
                                                                  params[1], params[2]);    \
            });

#define PENNYLANE_REGISTER_GATE_OP(GATE_NAME, KERNEL_TYPE, NUM_PARAMS) \
        PENNYLANE_REGISTER_GATE_OP_PARAMS##NUM_PARAMS(GATE_NAME, KERNEL_TYPE)


namespace Pennylane {

template<class fp_t, KernelType kernel_type>
int registerAllGateOps() {
    using CFP_t = std::complex<fp_t>;
    auto& dispatcher = DynamicDispatcher<fp_t>::getInstance();

    /* Single-qubit gates */
    PENNYLANE_REGISTER_GATE_OP(PauliX, kernel_type, 0);
    PENNYLANE_REGISTER_GATE_OP(PauliY, kernel_type, 0);
    PENNYLANE_REGISTER_GATE_OP(PauliZ, kernel_type, 0);
    PENNYLANE_REGISTER_GATE_OP(Hadamard, kernel_type, 0);
    PENNYLANE_REGISTER_GATE_OP(S, kernel_type, 0);
    PENNYLANE_REGISTER_GATE_OP(T, kernel_type, 0);
    PENNYLANE_REGISTER_GATE_OP(RX, kernel_type, 1);
    PENNYLANE_REGISTER_GATE_OP(RY, kernel_type, 1);
    PENNYLANE_REGISTER_GATE_OP(RZ, kernel_type, 1);
    PENNYLANE_REGISTER_GATE_OP(PhaseShift, kernel_type, 1);
    PENNYLANE_REGISTER_GATE_OP(Rot, kernel_type, 3);
    /* Two-qubit gates */
    PENNYLANE_REGISTER_GATE_OP(ControlledPhaseShift, kernel_type, 1);
    PENNYLANE_REGISTER_GATE_OP(CNOT, kernel_type, 0);
    PENNYLANE_REGISTER_GATE_OP(CZ, kernel_type, 0);
    PENNYLANE_REGISTER_GATE_OP(SWAP, kernel_type, 0);
    PENNYLANE_REGISTER_GATE_OP(CRX, kernel_type, 1);
    PENNYLANE_REGISTER_GATE_OP(CRY, kernel_type, 1);
    PENNYLANE_REGISTER_GATE_OP(CRZ, kernel_type, 1);
    PENNYLANE_REGISTER_GATE_OP(CRot, kernel_type, 3);
    /* Three-qubit gates */
    PENNYLANE_REGISTER_GATE_OP(Toffoli, kernel_type, 0);
    PENNYLANE_REGISTER_GATE_OP(CSWAP, kernel_type, 0);

    return 0;
}

template<class fp_t, KernelType kernel_type>
struct registerBeforeMain {
    static const int dummy;
};

template<>
const int registerBeforeMain<float, KernelType::PI>::dummy = 
          registerAllGateOps<float, KernelType::PI>();

template<>
const int registerBeforeMain<float, KernelType::LM>::dummy = 
          registerAllGateOps<float, KernelType::LM>();

template<>
const int registerBeforeMain<double, KernelType::PI>::dummy = 
          registerAllGateOps<double, KernelType::PI>();

template<>
const int registerBeforeMain<double, KernelType::LM>::dummy = 
          registerAllGateOps<double, KernelType::LM>();
}
